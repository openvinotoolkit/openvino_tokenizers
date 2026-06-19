import requests
from transformers import AutoTokenizer


MAX_RETRY = 2


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
