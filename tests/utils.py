import requests
from transformers import AutoTokenizer


def get_hf_tokenizer(request, fast_tokenizer=True, trust_remote_code=False, left_padding=None):
    kwargs = {}
    if left_padding is not None:
        kwargs["padding_side"] = "left" if left_padding else "right"
        kwargs["truncation_side"] = "left" if left_padding else "right"

    for retry in range(2):
        try:
            return AutoTokenizer.from_pretrained(
                request.param, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code, **kwargs
            )
        except requests.ReadTimeout:
            pass
