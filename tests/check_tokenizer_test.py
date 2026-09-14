import sys
from types import SimpleNamespace

import numpy as np
from openvino_tokenizers.cli_tools import check_tokenizer


def test_genai_checker_forwards_runtime_options(monkeypatch):
    encode_calls = []

    class FakeGenAITokenizer:
        def __init__(self, *_args):
            pass

        def encode(self, value, **kwargs):
            encode_calls.append(kwargs)
            size = len(value) if isinstance(value, list) else 1
            return SimpleNamespace(input_ids=SimpleNamespace(data=np.ones((size, 1), dtype=np.int64)))

        def decode(self, *_args, **_kwargs):
            return ["decoded"]

    class FakeHFTokenizer:
        def __call__(self, value, **_kwargs):
            return {"input_ids": np.ones((len(value), 1), dtype=np.int64)}

        def decode(self, *_args, **_kwargs):
            return "decoded"

    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(Tokenizer=FakeGenAITokenizer))
    monkeypatch.setattr(check_tokenizer, "ALL_TEST_STRINGS", ["sample"])

    assert check_tokenizer.step_test_genai(FakeHFTokenizer(), "unused", False, True, 32, True) == 0
    assert encode_calls == [
        {"pad_to_max_length": True, "truncation": True, "max_length": 32, "add_special_tokens": True},
        {"pad_to_max_length": True, "truncation": True, "max_length": 32, "add_special_tokens": False},
    ]
