import numpy as np
import requests
from openvino import AsyncInferQueue, CompiledModel
from transformers import AutoTokenizer


MAX_RETRY = 2


class AsyncTokenizerRunner:
    """Callable drop-in for a ``CompiledModel`` that batches a known corpus of inputs
    through an ``AsyncInferQueue`` on first use.

    The (de)tokenizer tests parametrize over a fixed, module-level list of strings and
    invoke the compiled model once per item with a synchronous, depth-1 call. That
    pattern cannot exploit the THROUGHPUT performance hint, which only pays off with
    many requests in flight across CPU streams. Because the corpus is known up front,
    the first call primes the whole corpus through one async queue and memoizes the
    per-item results; every later call is an O(1) lookup.

    The corpus elements are the exact objects the tests pass to the model: strings for
    tokenizers, or token-id arrays for detokenizers. Inputs that are not part of the
    primed corpus (multi-string batches, chat strings built at runtime, pair inputs)
    fall through to a plain synchronous call, so correctness and the dict-by-output-name
    access used by the tests are preserved.
    """

    def __init__(self, compiled_model: CompiledModel, corpus: list) -> None:
        self._model = compiled_model
        self._corpus = corpus
        self._cache: dict = {}
        self._primed = False

    @staticmethod
    def _key(inputs):
        if isinstance(inputs, str):
            return (inputs,)
        if isinstance(inputs, np.ndarray):
            return (inputs.shape, inputs.dtype.str, inputs.tobytes())
        if isinstance(inputs, (list, tuple)) and all(isinstance(item, str) for item in inputs):
            return tuple(inputs)
        return None

    def _results_to_dict(self, request) -> dict:
        results = {}
        for output in self._model.outputs:
            tensor = request.results[output].copy()
            for name in output.get_names():
                results[name] = tensor
        return results

    def _model_input(self, item):
        return [item] if isinstance(item, str) else item

    def _prime(self) -> None:
        self._primed = True

        queue = AsyncInferQueue(self._model)

        def callback(request, key) -> None:
            self._cache[key] = self._results_to_dict(request)

        queue.set_callback(callback)
        try:
            for item in self._corpus:
                queue.start_async(self._model_input(item), self._key(item))
            queue.wait_all()
        except Exception:
            pass

    def __call__(self, inputs):
        if not self._primed:
            self._prime()

        key = self._key(inputs)
        if key is not None and key in self._cache:
            return self._cache[key]

        # Cache miss (multi-string / pair / chat input): fall back to a synchronous call.
        return self._model(inputs)


def get_hf_tokenizer(request, fast_tokenizer=True, trust_remote_code=False, left_padding=None):
    kwargs = {}
    if left_padding is not None:
        kwargs["padding_side"] = "left" if left_padding else "right"
        kwargs["truncation_side"] = "left" if left_padding else "right"

    for retry in range(1, MAX_RETRY + 1):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                request.param, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code, **kwargs
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id or getattr(tokenizer, "eod_id", None) or 0
            return tokenizer
        except requests.ReadTimeout:
            if retry == MAX_RETRY:
                raise
