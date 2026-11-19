# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
from openvino import Core
from openvino_tokenizers import convert_tokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


core = Core()

test_strings = [
    "Hello world",
    "Hello, world!",
    "a  b",
    " leading space",
    "trailing space ",
    "a",
]


def build_bpe_metaspace_tokenizer(prepend_scheme: str, split: bool) -> PreTrainedTokenizerFast:
    # Small BPE model with a Metaspace pre-tokenizer and decoder, as in NLLB or Reformer tokenizer.json files
    chars = sorted(set("".join(test_strings).replace(" ", "")))
    vocab = {"<unk>": 0, "▁": 1}
    merges = []
    for char in chars:
        vocab[char] = len(vocab)
    for char in chars:
        vocab["▁" + char] = len(vocab)
        merges.append(("▁", char))

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme=prepend_scheme, split=split)
    tokenizer.decoder = decoders.Metaspace(prepend_scheme=prepend_scheme, split=split)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")


@pytest.mark.parametrize(
    "prepend_scheme, split",
    [
        ("always", True),
        ("always", False),
        ("first", False),
        ("never", True),
        ("never", False),
    ],
)
def test_bpe_metaspace_detokenizer(prepend_scheme, split):
    hf_tokenizer = build_bpe_metaspace_tokenizer(prepend_scheme, split)
    _, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    ov_detokenizer = core.compile_model(ov_detokenizer)

    for test_string in test_strings:
        token_ids = hf_tokenizer(test_string).input_ids
        hf_output = hf_tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        ov_output = ov_detokenizer([token_ids])["string_output"].tolist()
        assert ov_output == [hf_output], test_string
