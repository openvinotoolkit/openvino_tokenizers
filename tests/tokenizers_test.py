# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import difflib
import os
import sys
from collections import namedtuple
from dataclasses import fields
from typing import Any, Optional, Union

import numpy as np
import pytest
from openvino import Core, Model, Type, properties, save_model
from openvino_tokenizers import convert_tokenizer
from openvino_tokenizers._testing_data import (
    ALL_TEST_STRINGS as single_string_corpus,
)
from openvino_tokenizers._testing_data import (
    CHAT_HISTORIES as chat_messages,
)
from openvino_tokenizers._testing_data import (
    EMOJI_TEST_STRINGS as emoji_test_strings,
)
from openvino_tokenizers._testing_data import (
    ENGLISH_TEST_STRINGS as eng_test_strings,
)
from openvino_tokenizers._testing_data import (
    MISC_TEST_STRINGS as misc_strings,
)
from openvino_tokenizers._testing_data import (
    MULTILINGUAL_TEST_STRINGS as multilingual_test_strings,
)
from openvino_tokenizers.constants import ORIGINAL_TOKENIZER_CLASS_NAME, rt_info_to_hf_attribute_map
from openvino_tokenizers.utils import TokenzierConversionParams, get_hf_tokenizer_attribute
from transformers import AutoTokenizer

from tests.utils import AsyncTokenizerRunner, get_hf_tokenizer


if os.environ.get("OV_TOKENIZERS_TESTS_PRINT_WHOLE_DIFF"):
    np.set_printoptions(threshold=sys.maxsize)

core = Core()

wordpiece_models = [
    "bert-base-multilingual-cased",
    "cointegrated/rubert-tiny2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/mobilebert-uncased",
    "rasa/LaBSE",
]
bpe_models = [
    "Xenova/gpt-4o",
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "koalajun/Gemma-2-9b-it-Ko-Crypto-Translate",
    "roberta-base",
    "deepseek-ai/DeepSeek-V3-0324",
    "Qwen/Qwen3-Reranker-0.6B",
    "facebook/galactica-120b",
    "microsoft/deberta-base",
    "bigscience/bloom",
    "deepseek-ai/deepseek-coder-6.7b-instruct",  # sentencepiece tokenizer without .model file fallback to fast BPE
    "answerdotai/ModernBERT-base",
    "tiiuae/Falcon3-7B-Instruct",
    "LiquidAI/LFM2-350M",
]
sentencepiece_models = [
    # bpe
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "NousResearch/Llama-2-13b-hf",
    "microsoft/Phi-3-mini-128k-instruct",
    "mlx-community/quantized-gemma-7b-it",
    # unigram
    "camembert-base",
    "google/flan-t5-xxl",
    "BAAI/bge-reranker-v2-m3",
    "microsoft/deberta-v3-base",  # byte fallback
    "facebook/musicgen-small",
    "rinna/bilingual-gpt-neox-4b",  # t5-tokenizer
    # chars
    "microsoft/speecht5_tts",
]
tiktiken_models = [
    "Qwen/Qwen-14B-Chat",
    "THUDM/glm-4-9b-chat",
]


# THROUGHPUT hint lets AsyncTokenizerRunner spread the primed corpus across CPU streams.
THROUGHPUT_CONFIG = {properties.hint.performance_mode(): properties.hint.PerformanceMode.THROUGHPUT}


def get_tokenizer(
    hf_tokenizer, add_special_tokens=True, use_max_padding=False, use_sentencepiece_backend=False, truncation=False
):
    ov_tokenizer = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=False,
        add_special_tokens=add_special_tokens,
        use_max_padding=use_max_padding,
        truncation=truncation,
        use_sentencepiece_backend=use_sentencepiece_backend,
    )
    compiled_tokenizer = core.compile_model(ov_tokenizer, "CPU", THROUGHPUT_CONFIG)
    return hf_tokenizer, AsyncTokenizerRunner(compiled_tokenizer, single_string_corpus)


def build_detokenizer_corpus(hf_tokenizer) -> list:
    return [
        hf_tokenizer(test_string, return_tensors="np", padding=True).input_ids.astype("int32")
        for test_string in single_string_corpus
    ]


def get_tokenizer_detokenizer(
    hf_tokenizer,
    streaming_detokenizer=False,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=None,
    use_sentencepiece_backend=False,
):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=True,
        streaming_detokenizer=streaming_detokenizer,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        use_sentencepiece_backend=use_sentencepiece_backend,
    )
    compiled_tokenizer = core.compile_model(ov_tokenizer, "CPU", THROUGHPUT_CONFIG)
    compiled_detokenizer = core.compile_model(ov_detokenizer, "CPU", THROUGHPUT_CONFIG)
    if not streaming_detokenizer:
        compiled_detokenizer = AsyncTokenizerRunner(compiled_detokenizer, build_detokenizer_corpus(hf_tokenizer))
    return hf_tokenizer, compiled_tokenizer, compiled_detokenizer


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_left: "left_pad" if is_left else "right_pad")
def use_left_padding(request):
    return request.param


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_max: "max_pad" if is_max else "min_pad")
def use_max_padding(request):
    return request.param


@pytest.fixture(scope="session", params=wordpiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_wordpiece_tokenizers(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session", params=wordpiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_wordpiece_tokenizers_with_padding_sides(request, use_left_padding):
    return get_hf_tokenizer(request, left_padding=use_left_padding)


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_fast: "Fast" if is_fast else "Slow")
def is_fast_tokenizer(request):
    return request.param


@pytest.fixture(scope="session", params=[True, False], ids=lambda to_add: "add_tokens" if to_add else "no_add_tokens")
def do_add_special_tokens(request):
    return request.param


@pytest.fixture(
    scope="session", params=[True, False], ids=lambda do_skip: "skip_tokens" if do_skip else "no_skip_tokens"
)
def do_skip_special_tokens(request):
    return request.param


@pytest.fixture(
    scope="session", params=[True, False], ids=lambda truncation: "truncation" if truncation else "no_truncation"
)
def truncation(request):
    return request.param


@pytest.fixture(
    scope="session", params=[True, False], ids=lambda do_clean: "clean_spaces" if do_clean else "no_clean_spaces"
)
def do_clean_up_tokenization_spaces(request):
    return request.param


@pytest.fixture(scope="session", params=[True, False], ids=lambda do_clean: "sp_backend" if do_clean else "")
def is_sentencepiece_backend(request):
    return request.param


@pytest.fixture(scope="session", params=sentencepiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_sentencepiece_tokenizers(request, is_fast_tokenizer, is_sentencepiece_backend):
    if not is_fast_tokenizer and not is_sentencepiece_backend:
        pytest.skip("Legacy tokenizer must use Sentencepiece backend.")

    hf_tokenizer = get_hf_tokenizer(request, fast_tokenizer=is_fast_tokenizer, trust_remote_code=True)
    if not hf_tokenizer.is_fast and is_fast_tokenizer:
        pytest.skip("Fast tokenizer should use Rust backend.")

    return hf_tokenizer


@pytest.fixture(scope="session", params=sentencepiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_sentencepiece_tokenizers_with_padding_sides(
    request, use_left_padding, is_fast_tokenizer, is_sentencepiece_backend
):
    if not is_fast_tokenizer and not is_sentencepiece_backend:
        pytest.skip("Legacy tokenizer must use sentencepiece backend.")

    hf_tokenizer = get_hf_tokenizer(request, left_padding=use_left_padding, trust_remote_code=True)
    if not hf_tokenizer.is_fast and is_fast_tokenizer:
        pytest.skip("Fast tokenizer should use Rust backend.")

    return hf_tokenizer


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_bpe_tokenizers(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_bpe_tokenizers_with_padding_sides(request, use_left_padding):
    hf_tokenizer = get_hf_tokenizer(request, left_padding=use_left_padding)
    return hf_tokenizer


@pytest.fixture(scope="session", params=tiktiken_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_tiktoken_tokenizers(request):
    return get_hf_tokenizer(request, trust_remote_code=True)


@pytest.fixture(scope="session", params=tiktiken_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_tiktoken_tokenizers_with_padding_sides(request, use_left_padding):
    hf_tokenizer = get_hf_tokenizer(request, trust_remote_code=True, left_padding=use_left_padding)
    return hf_tokenizer


@pytest.fixture(scope="session")
def wordpiece_tokenizers(hf_wordpiece_tokenizers, do_add_special_tokens):
    return get_tokenizer(hf_wordpiece_tokenizers, add_special_tokens=do_add_special_tokens)


@pytest.fixture(scope="session")
def wordpiece_tokenizers_with_padding_options(
    hf_wordpiece_tokenizers_with_padding_sides, do_add_special_tokens, use_max_padding
):
    if use_max_padding and getattr(hf_wordpiece_tokenizers_with_padding_sides, "model_max_length") > 2**31:
        pytest.skip("Cannot test max_padding=True for tokenizer without max length.")

    return get_tokenizer(
        hf_wordpiece_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
        use_max_padding=use_max_padding,
    )


@pytest.fixture(scope="session")
def wordpiece_tokenizers_detokenizers(
    hf_wordpiece_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    return get_tokenizer_detokenizer(
        hf_wordpiece_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )


@pytest.fixture(scope="session")
def bpe_tokenizers(hf_bpe_tokenizers, do_add_special_tokens):
    return get_tokenizer(hf_bpe_tokenizers, add_special_tokens=do_add_special_tokens)


@pytest.fixture(scope="session")
def bpe_tokenizers_with_padding_options(
    hf_bpe_tokenizers_with_padding_sides, do_add_special_tokens, use_max_padding, truncation
):
    if use_max_padding and getattr(hf_bpe_tokenizers_with_padding_sides, "model_max_length") > 2**31:
        pytest.skip("Cannot test max_padding=True for tokenizer without max length.")

    return get_tokenizer(
        hf_bpe_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
        use_max_padding=use_max_padding,
        truncation=truncation,
    )


@pytest.fixture(scope="session")
def bpe_tokenizers_detokenizers(hf_bpe_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces):
    return get_tokenizer_detokenizer(
        hf_bpe_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )


@pytest.fixture(scope="session")
def sentencepice_tokenizers(hf_sentencepiece_tokenizers, do_add_special_tokens, is_sentencepiece_backend):
    return get_tokenizer(
        hf_sentencepiece_tokenizers,
        add_special_tokens=do_add_special_tokens,
        use_sentencepiece_backend=is_sentencepiece_backend,
    )


@pytest.fixture(scope="session")
def sentencepiece_tokenizers_with_padding_options(
    hf_sentencepiece_tokenizers_with_padding_sides, do_add_special_tokens, use_left_padding, is_sentencepiece_backend
):
    if (
        hf_sentencepiece_tokenizers_with_padding_sides.name_or_path in ("THUDM/chatglm2-6b", "THUDM/chatglm3-6b")
        and not use_left_padding
    ):
        pytest.skip("chatglm supports left padding only")
    if hf_sentencepiece_tokenizers_with_padding_sides.name_or_path == "THUDM/chatglm2-6b" and do_add_special_tokens:
        pytest.skip("chatglm2 never adds special tokens")
    if (
        hf_sentencepiece_tokenizers_with_padding_sides.name_or_path == "THUDM/chatglm3-6b"
        and not do_add_special_tokens
    ):
        pytest.skip("chatglm3 always adds special tokens")

    return get_tokenizer(
        hf_sentencepiece_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
    )


@pytest.fixture(scope="session")
def sentencepice_tokenizers_detokenizers(
    hf_sentencepiece_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces, is_sentencepiece_backend
):
    # chatglm2 always skips special tokens, chatglam3 always not skip
    if hf_sentencepiece_tokenizers.name_or_path == "THUDM/chatglm2-6b" and not do_skip_special_tokens:
        pytest.skip("chatglm2 always skips special tokens")
    if hf_sentencepiece_tokenizers.name_or_path == "THUDM/chatglm3-6b" and do_skip_special_tokens:
        pytest.skip("chatglm3 always adds special tokens")

    return get_tokenizer_detokenizer(
        hf_sentencepiece_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
        use_sentencepiece_backend=is_sentencepiece_backend,
    )


@pytest.fixture(scope="session")
def tiktoken_tokenizers(hf_tiktoken_tokenizers, do_add_special_tokens):
    return get_tokenizer(hf_tiktoken_tokenizers, add_special_tokens=do_add_special_tokens)


@pytest.fixture(scope="session")
def tiktoken_tokenizers_with_padding_options(
    hf_tiktoken_tokenizers_with_padding_sides, do_add_special_tokens, use_max_padding, use_left_padding
):
    if use_max_padding and getattr(hf_tiktoken_tokenizers_with_padding_sides, "model_max_length") > 2**31:
        pytest.skip("Cannot test max_padding=True for tokenizer without max length.")
    if not use_left_padding and hf_tiktoken_tokenizers_with_padding_sides.name_or_path == "THUDM/glm-4-9b":
        pytest.skip("chatglm supports left padding only")
    return get_tokenizer(
        hf_tiktoken_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
        use_max_padding=use_max_padding,
    )


@pytest.fixture(scope="session")
def tiktoken_tokenizers_detokenizers(hf_tiktoken_tokenizers, do_skip_special_tokens):
    return get_tokenizer_detokenizer(
        hf_tiktoken_tokenizers, skip_special_tokens=do_skip_special_tokens, clean_up_tokenization_spaces=False
    )


@pytest.fixture(
    scope="session", params=["openlm-research/open_llama_3b_v2"], ids=lambda checkpoint: checkpoint.split("/")[-1]
)
def hf_tokenizers_for_streaming(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session")
def sentencepiece_streaming_tokenizers(hf_tokenizers_for_streaming):
    return get_tokenizer_detokenizer(
        hf_tokenizers_for_streaming, streaming_detokenizer=True, use_sentencepiece_backend=True
    )


def print_diff(left, right) -> str:
    left = str(left.reshape(-1)).split("\n")
    right = str(right.reshape(-1)).split("\n")

    diff = "\n".join(difflib.ndiff(left, right))
    return f"\n{diff}"


def convert_hf_object_array_to_dense(hf_result: np.ndarray, ov_result: np.ndarray, output_name: str, hf_tokenizer):
    if hf_result.dtype != object or len(hf_result.shape) != 1 or len(ov_result.shape) != 2:
        return hf_result

    if output_name == "input_ids":
        pad_value = hf_tokenizer.pad_token_id or 0
    else:
        pad_value = 0

    dense_rows = []
    target_length = ov_result.shape[1]
    for row in hf_result:
        row = np.asarray(row, dtype=ov_result.dtype)
        pad_width = target_length - row.shape[0]
        if pad_width < 0:
            return hf_result

        padding = np.full(pad_width, pad_value, dtype=ov_result.dtype)
        if getattr(hf_tokenizer, "padding_side", "right") == "left":
            row = np.concatenate([padding, row])
        else:
            row = np.concatenate([row, padding])
        dense_rows.append(row)

    return np.stack(dense_rows)


def check_tokenizer_output(
    tokenizers: tuple,
    test_string: Union[str, list[str]],
    skip_missing_outputs: bool = False,
    hf_tokenizer_kwargs: Optional[dict[str, Any]] = None,
    calculate_diff: bool = False,
) -> tuple[bool, str]:
    hf_tokenizer, ov_tokenizer = tokenizers
    hf_tokenizer_kwargs = {} if hf_tokenizer_kwargs is None else hf_tokenizer_kwargs

    if isinstance(test_string, str):
        test_string = [test_string]

    test_string_ov = test_string
    if isinstance(test_string, list) and len(test_string) == 2 and isinstance(test_string[0], list):
        if len(test_string[0]) == 1 and len(test_string[1]) == 1:
            test_string_hf = [[test_string[0][0], test_string[1][0]]]
        else:
            # broadcast ([N], [1]) and ([1], [N]) to ([N], [N]) for HF
            if len(test_string[0]) > len(test_string[1]):
                test_string_hf = [[test_string[0][i], test_string[1][0]] for i in range(len(test_string[0]))]
            else:
                test_string_hf = [[test_string[0][0], test_string[1][i]] for i in range(len(test_string[1]))]
            test_string_ov = tuple(test_string)
    else:
        test_string_hf = test_string

    hf_tokenized = hf_tokenizer(test_string_hf, return_tensors="np", **hf_tokenizer_kwargs)
    ov_tokenized = ov_tokenizer(test_string_ov)

    hf_padding = hf_tokenizer_kwargs.get("padding", False)
    if hf_padding is False:
        hf_padding = "do_not_pad"
    elif hf_padding is True:
        hf_padding = "longest"

    for output_name, hf_result in hf_tokenized.items():
        if output_name not in ov_tokenized and skip_missing_outputs:
            continue

        assert output_name in ov_tokenized, f"OV Tokenizer missing output: {output_name}"
        ov_result = ov_tokenized[output_name]

        # hf_result can be object if the tokenizer returns a ragged array, which is not supported by OV.
        # This can happen only when padding is set to max_length and truncation is False.
        # In that case, before comparison convert ragged array from HF to a dense.
        if hf_padding == "max_length" and hf_tokenizer_kwargs.get("truncation", False) is False:
            hf_result = convert_hf_object_array_to_dense(hf_result, ov_result, output_name, hf_tokenizer)

        outputs = f"\nHF: {hf_result}\nOV: {ov_result}"
        diff = print_diff(hf_result, ov_result) if calculate_diff and ov_result.shape != hf_result.shape else outputs
        if ov_result.shape != hf_result.shape:
            return False, diff

        if not np.all(ov_result == hf_result):
            return False, outputs

        return True, ""


def check_detokenizer_output(
    detokenizers: tuple,
    test_string: Union[str, list[str]],
    hf_detokenizer_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    hf_tokenizer, _, ov_detokenizer = detokenizers
    hf_detokenizer_kwargs = {} if hf_detokenizer_kwargs is None else hf_detokenizer_kwargs

    token_ids = hf_tokenizer(test_string, return_tensors="np", padding=True).input_ids
    hf_output = hf_tokenizer.batch_decode(token_ids, **hf_detokenizer_kwargs)
    ov_output = ov_detokenizer(token_ids.astype("int32"))["string_output"].tolist()

    assert ov_output == hf_output


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_wordpiece_tokenizers(wordpiece_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    result, diff = check_tokenizer_output(
        wordpiece_tokenizers,
        test_string=test_string,
        skip_missing_outputs=False,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_wordpiece_tokenizers_multiple_strings(
    wordpiece_tokenizers_with_padding_options, test_string, do_add_special_tokens, use_max_padding
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": "max_length" if use_max_padding else True,
    }
    result, diff = check_tokenizer_output(
        wordpiece_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=False,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_wordpiece_model_detokenizer(
    wordpiece_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        wordpiece_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_sentencepiece_model_tokenizer(sentencepice_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    result, diff = check_tokenizer_output(
        sentencepice_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_chat",
    chat_messages,
)
def test_sentencepiece_model_tokenizer_chat(sentencepice_tokenizers, test_chat, do_add_special_tokens):
    hf_tokenizer, ov_tokenizer = sentencepice_tokenizers
    if hf_tokenizer.chat_template is None:
        pytest.skip("No chat template")

    from jinja2 import TemplateError

    try:
        test_string = hf_tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
    except TemplateError:
        # filter system message
        test_string = hf_tokenizer.apply_chat_template(test_chat[1:], tokenize=False, add_generation_prompt=True)

    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    result, diff = check_tokenizer_output(
        sentencepice_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_sentencepiece_tokenizers_multiple_strings(
    sentencepiece_tokenizers_with_padding_options, test_string, do_add_special_tokens
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": True,
    }
    result, diff = check_tokenizer_output(
        sentencepiece_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_sentencepiece_model_detokenizer(
    sentencepice_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        sentencepice_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_bpe_tokenizers_outputs(bpe_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    result, diff = check_tokenizer_output(
        bpe_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_chat",
    chat_messages,
)
def test_bpe_model_tokenizer_chat(bpe_tokenizers, test_chat, do_add_special_tokens):
    hf_tokenizer, ov_tokenizer = bpe_tokenizers
    if hf_tokenizer.chat_template is None:
        pytest.skip("No chat template")

    test_string = hf_tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    result, diff = check_tokenizer_output(
        bpe_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_bpe_tokenizers_multiple_strings(
    bpe_tokenizers_with_padding_options, test_string, do_add_special_tokens, use_max_padding, truncation
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": "max_length" if use_max_padding else True,
        "truncation": truncation,
    }
    result, diff = check_tokenizer_output(
        bpe_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_bpe_detokenizer(
    bpe_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        bpe_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_tokenizers(tiktoken_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    result, diff = check_tokenizer_output(
        tiktoken_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_chat",
    chat_messages,
)
def test_tiktoken_model_tokenizer_chat(tiktoken_tokenizers, test_chat, do_add_special_tokens):
    hf_tokenizer, ov_tokenizer = tiktoken_tokenizers
    if hf_tokenizer.chat_template is None:
        pytest.skip("No chat template")

    test_string = hf_tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    result, diff = check_tokenizer_output(
        tiktoken_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_tiktoken_tokenizers_multiple_strings(
    tiktoken_tokenizers_with_padding_options, test_string, do_add_special_tokens
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": True,
    }
    result, diff = check_tokenizer_output(
        tiktoken_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    assert result, diff


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_detokenizer(
    tiktoken_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        tiktoken_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


def test_streaming_detokenizer(sentencepiece_streaming_tokenizers):
    hf_tokenizer, _, ov_detokenizer = sentencepiece_streaming_tokenizers
    test_string = "this is a test string"
    tokenized_string = hf_tokenizer(test_string).input_ids
    hf_detokenized = hf_tokenizer.decode(tokenized_string)

    detokenized_stream = ""
    for token in tokenized_string:
        ov_output = ov_detokenizer(np.atleast_2d(token))["string_output"][0]
        detokenized_stream += ov_output

    assert detokenized_stream == hf_detokenized


def test_detokenizer_results_align_with_hf_on_multitoken_symbols_for_streaming():
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat", trust_remote_code=True)
    _, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    ov_detokenizer = core.compile_model(ov_detokenizer)

    test_string = "🤷‍♂️"  # tokenized into 5 tokens
    tokenized_string = hf_tokenizer(test_string).input_ids

    detokenized_stream = ""
    hf_detokenized_stream = ""
    for token in tokenized_string:
        ov_output = ov_detokenizer(np.atleast_2d(token))["string_output"][0]
        detokenized_stream += ov_output

        hf_output = hf_tokenizer.decode(token)
        hf_detokenized_stream += hf_output

    assert detokenized_stream == hf_detokenized_stream


def check_rt_info(hf_tokenizer, *models: Model) -> None:
    for model in models:
        assert model.has_rt_info(ORIGINAL_TOKENIZER_CLASS_NAME), ORIGINAL_TOKENIZER_CLASS_NAME
        assert model.get_rt_info(ORIGINAL_TOKENIZER_CLASS_NAME) == str(type(hf_tokenizer))

        for field_name, attributes in rt_info_to_hf_attribute_map.items():
            attribute = get_hf_tokenizer_attribute(hf_tokenizer, attributes)
            if attribute is None:
                assert not model.has_rt_info(field_name), field_name
            else:
                assert model.has_rt_info(field_name), field_name
                assert model.get_rt_info(field_name).value == attribute, (
                    field_name,
                    attributes,
                    model.get_rt_info(field_name).value,
                )


def test_rt_info_wordpiece(hf_wordpiece_tokenizers):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_wordpiece_tokenizers,
        with_detokenizer=True,
    )
    check_rt_info(hf_wordpiece_tokenizers, ov_tokenizer, ov_detokenizer)


def test_rt_info_bpe(hf_bpe_tokenizers):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_bpe_tokenizers,
        with_detokenizer=True,
    )
    check_rt_info(hf_bpe_tokenizers, ov_tokenizer, ov_detokenizer)


def test_rt_info_tiktoken(hf_tiktoken_tokenizers):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tiktoken_tokenizers,
        with_detokenizer=True,
    )
    check_rt_info(hf_tiktoken_tokenizers, ov_tokenizer, ov_detokenizer)


def test_rt_info_sentencepiece(hf_sentencepiece_tokenizers, is_sentencepiece_backend, is_fast_tokenizer):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_sentencepiece_tokenizers, with_detokenizer=True, use_sentencepiece_backend=is_sentencepiece_backend
    )
    check_rt_info(hf_sentencepiece_tokenizers, ov_tokenizer, ov_detokenizer)


models_to_check_rt_info = [
    # one model from each category
    "bert-base-uncased",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Xenova/gpt-4o",
    "Qwen/Qwen-14B-Chat",
]


@pytest.fixture(scope="session", params=models_to_check_rt_info)
def tokenizer_to_check_rt_info(request):
    return get_hf_tokenizer(request, trust_remote_code=True)


def test_rt_info_conversion_params(tokenizer_to_check_rt_info):
    conversion_params = TokenzierConversionParams(
        with_detokenizer=False,
        add_special_tokens=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=None,
        tokenizer_output_type=Type.i64,
        detokenizer_input_type=Type.i64,
        streaming_detokenizer=False,
        use_max_padding=False,
        truncation=False,
        handle_special_tokens_with_re=None,
        use_sentencepiece_backend=False,
        utf8_replace_mode=None,
        number_of_inputs=1,
    )

    ov_tokenizer = convert_tokenizer(tokenizer_to_check_rt_info, conversion_params)
    print(ov_tokenizer)
    if not conversion_params.with_detokenizer:
        ov_tokenizer = (ov_tokenizer,)

    for model in ov_tokenizer:
        for key in fields(conversion_params):
            val = getattr(conversion_params, key.name)
            if val is None:
                val = {}
            elif isinstance(val, (Type, int)) and not isinstance(val, bool):
                # bool is subcalss of int, hence there are 2 checks.
                # While bool values are stored as str, e.g. 'False'
                # type info and integers are stored as object.
                pass
            else:
                val = str(val)
            assert val == model.get_rt_info(key.name).value


cache_test_strings = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop.0987654321 - eng, but with d1gits: 123",
    "<s>[INST] <<SYS>> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. <</SYS>> You will act as a Christian, and fully summarize following text:\nSometimes it's nice to take a minute in the pew by yourself beforehand. You have this beautiful church probably almost all to yourself. Can you feel its energy resonating through you? Can you feel the majesty of the Lord's kingdom and how you're a part of it? Take a moment to kneel and pray with your head down and hands clasped together. Reflect on your faith and how you feel currently. Think about how you've been responding to God's call and how you've been living in the light of his love. When the priest is ready for you, of course. You'll probably see him there by his lonesome or someone else walk out just before you. Sit down either across from him or behind the screen -- it's totally up to you whether or not you prefer to remain anonymous. He won't treat you any differently either way. Make the sign of the cross upon his prompt, saying, \"Bless me, Father, for I have sinned. It has been 10 years since my last confession.\" This is your standard, traditional phrasing. However, if you just sit down and say hello, that's fine, too. The priest knows what he's doing. The Byzantine Rite is a bit different. The priest may sit to your side and put his epitrachelion on your head. He may then also do the Prayer of Absolution. But the idea remains the exact same -- just go wherever he takes you. Once you sit down and you've made the sign of the cross, just sit back and follow the priest's lead. He'll ask you how long it's been since your last confession (if you don't voluntarily offer that information), how you are feeling, maybe how your faith is going, and then ask you what sins you would like to talk about with him and God. It's just a casual conversation! Do not fret. There is absolutely zero pressure on your part. Again, as long as you come there with the intention of leaving with a clean heart, you're more than welcome in the church. There is no wrong way to go about confession! This part is intimidating, but think about it this way: the priest you're talking to has probably heard just about everything before. Whatever you have to say will not blow his mind. So when he asks, start rattling them off, from the most serious to the least. If he asks any questions, answer them, but do not feel the need to go into detail. A simple, \"I did so and so,\" will suffice. Your priest is going to be very understanding. If you don't remember the exact timeframe, that's fine. If you don't remember your motivation, that's fine. All your priest cares about is that you're being as honest as possible and that your heart is in the right place. He'll talk you through everything, possibly asking about your intentions, but mainly just letting you know that God loves you, sin and all. If he has any ideas to bring you closer to God, he may suggest them at this juncture. He's there to help, after all. He will then ask you to make an Act of Contrition. That goes like this: My God, I am sorry for my sins with all my heart.In choosing to do wrong and failing to do good,I have sinned against You whom I should loveabove all things. I firmly intend, with your help,to do penance, to sin no more, andto avoid whatever leads me to sin.Our Savior Jesus Christ suffered and died for us.In his name, my God, have mercy.If you are a Roman Catholic, your act of contrition will go like this: Oh my God, I am very sorry for having offended thee. But most of all, because they offend you, my God, who is all good and deserving of all my love. I firmly resolve with the help of thy grace, to sin no more, and to avoid the near occasion of sin. Amen. Don't worry! It won't be anything huge. Take the absolution to heart -- you now have a brand new, clean slate to work with. \"Penance\" is your expression of regret and repentance, showing God that you're truly sorry and that you wish for nothing more than to be forgiven. Thanks. [/INST]",
]


@pytest.mark.parametrize(
    "model_id",
    [
        "Xenova/gpt-4o",
    ],
)
@pytest.mark.parametrize("test_string", cache_test_strings)
def test_loading_from_cache(tmp_path, model_id, test_string):
    request = namedtuple("request", ["param"])(model_id)

    hf_tokenizer = get_hf_tokenizer(request, trust_remote_code=True)
    ov_tokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=False)

    save_model(ov_tokenizer, tmp_path / "openvino_tokenizer.xml")
    ov_tokenizer = Core().read_model(tmp_path / "openvino_tokenizer.xml")

    # Compile with cache dir, to check if after restoration still will work fine.
    compiled_tokenizer = Core().compile_model(ov_tokenizer, "CPU", {properties.cache_dir: str(tmp_path)})
    check_tokenizer_output((hf_tokenizer, compiled_tokenizer), test_string=test_string)

    # On the second run, it should be loaded from cache.
    # Check that output is still the same
    compiled_tokenizer = Core().compile_model(ov_tokenizer, "CPU", {properties.cache_dir: str(tmp_path)})
    check_tokenizer_output((hf_tokenizer, compiled_tokenizer), test_string=test_string)


models_with_pair_input = [
    "answerdotai/ModernBERT-base",
    "amberoad/bert-multilingual-passage-reranking-msmarco",
    "BAAI/bge-reranker-v2.5-gemma2-lightweight",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "koalajun/Gemma-2-9b-it-Ko-Crypto-Translate",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "cointegrated/rubert-tiny2",
    "google/mobilebert-uncased",
    "microsoft/deberta-base",
    "sentence-transformers/all-MiniLM-L6-v2",
    "rasa/LaBSE",
    "bert-base-multilingual-cased",
    # Rerankers with Unigram
    "BAAI/bge-reranker-v2-m3",
    # model with RobertaProcessing
    "FacebookAI/roberta-base",
    # model with BertProcessing
    "shay681/HeBERT_finetuned_Legal_Clauses",
]


@pytest.fixture(scope="session", params=[7, 100, None])
def max_length(request):
    return request.param


@pytest.fixture(scope="session", params=models_with_pair_input, ids=lambda checkpoint: checkpoint.split("/")[-1])
def ov_hf_tokenizer_pair_with_trunc(request, use_left_padding, max_length):
    hf_tokenizer = get_hf_tokenizer(request, left_padding=use_left_padding, trust_remote_code=True)
    ov_tokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=False, number_of_inputs=2, max_length=max_length)
    ov_tokenizer = Core().compile_model(ov_tokenizer, "CPU")
    return hf_tokenizer, ov_tokenizer


@pytest.mark.parametrize(
    "test_string",
    [
        [["hi"], ["sun in yellow"]],
        [["Eng... test, string?!" * 100], ["Multiline\nstring!\nWow!"]],
        [["Eng... test, string?!"], ["Multiline\nstring!\nWow!" * 100]],
        [["Eng... test, string?!" * 100], ["Multiline\nstring!\nWow!" * 100]],
        [["hi" * 20], ["buy" * 90]],
        [["What is the capital of Great Britain"] * 4, ["London is capital of Great Britain"]],
        [["What is the capital of Great Britain"], ["London is capital of Great Britain"] * 4],
    ],
)
def test_pair_input(ov_hf_tokenizer_pair_with_trunc, test_string):
    result, diff = check_tokenizer_output(ov_hf_tokenizer_pair_with_trunc, test_string=test_string)
    assert result, diff
