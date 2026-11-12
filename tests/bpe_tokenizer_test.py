"""Offline BPE parity regressions, including unreachable vocabulary entries (#809)."""

import json

import openvino as ov
import pytest
from openvino_tokenizers import convert_tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def check_parity(reference, texts, tmp_path, roundtrip=False):
    if reference.pre_tokenizer is None:
        reference.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    reference.add_special_tokens(["<special-test>"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=reference)
    model = convert_tokenizer(tokenizer)
    if roundtrip:
        model = model.clone()
        path = tmp_path / "tokenizer.xml"
        ov.save_model(model, path)
        core = ov.Core()
        from openvino_tokenizers import _ext_path

        core.add_extension(str(_ext_path))
        model = core.read_model(path)
    compiled = ov.compile_model(model, "CPU")
    # Exercise both the initial calculation and cached calls.
    for _ in range(2):
        for text in texts:
            expected = reference.encode(text, add_special_tokens=False).ids
            assert compiled([text])["input_ids"][0].tolist() == expected, repr(text)


@pytest.mark.parametrize("ignore_merges", [False, True, None])
@pytest.mark.parametrize("byte_level", [False, True])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_bpe_unreachable_vocab(ignore_merges, byte_level, roundtrip, tmp_path):
    reference = Tokenizer(
        models.BPE(
            vocab={"a": 0, "b": 1, "c": 2, "x": 3, "ab": 4, "abc": 5},
            merges=[("a", "b")],
            ignore_merges=bool(ignore_merges),
        )
    )
    if byte_level:
        reference.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    if ignore_merges is None:
        data = json.loads(reference.to_str())
        del data["model"]["ignore_merges"]
        reference = Tokenizer.from_str(json.dumps(data))
    check_parity(reference, ["abc", "abcx", "xabc", "ab", "abcabc", "", "a"], tmp_path, roundtrip)


@pytest.mark.parametrize("ignore_merges", [False, True])
def test_bpe_merge_rank(ignore_merges, tmp_path):
    reference = Tokenizer(
        models.BPE(
            vocab={s: i for i, s in enumerate(["a", "b", "c", "ab", "bc", "abc"])},
            merges=[("b", "c"), ("a", "b"), ("ab", "c")],
            ignore_merges=ignore_merges,
        )
    )
    check_parity(reference, ["abc", "abcabc", "ab", "bc"], tmp_path)


@pytest.mark.parametrize("fuse_unk", [False, True])
@pytest.mark.parametrize("byte_fallback", [False, True])
@pytest.mark.parametrize("complete_fallback", [False, True])
def test_bpe_unicode_and_unknowns(fuse_unk, byte_fallback, complete_fallback, tmp_path):
    vocab = ["<unk>", "é", "a", "éa", "<0xC3>"]
    if complete_fallback:
        vocab += ["<0xB6>"]
    reference = Tokenizer(
        models.BPE(
            vocab={s: i for i, s in enumerate(vocab)},
            merges=[],
            unk_token="<unk>",
            fuse_unk=fuse_unk,
            byte_fallback=byte_fallback,
        )
    )
    check_parity(reference, ["éa", "öö", "éööa", "zöa", "zöz", "😀😀"], tmp_path)


@pytest.mark.parametrize("ignore_merges", [False, True])
def test_bpe_end_suffix(ignore_merges, tmp_path):
    reference = Tokenizer(
        models.BPE(
            vocab={s: i for i, s in enumerate(["a", "b", "b</w>", "ab</w>", "ab", "a</w>"])},
            merges=[],
            end_of_word_suffix="</w>",
            ignore_merges=ignore_merges,
        )
    )
    check_parity(reference, ["ab", "a", "b", "abab"], tmp_path, True)


@pytest.mark.parametrize("byte_level", [False, True])
def test_bpe_added_tokens(byte_level, tmp_path):
    reference = Tokenizer(models.BPE(vocab={"a": 0, "b": 1}, merges=[]))
    if byte_level:
        reference.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    reference.add_special_tokens(["<special>"])
    reference.add_tokens(["abab"])
    check_parity(reference, ["<special>", "a<special>b", "abab", "ababab"], tmp_path, True)


@pytest.mark.parametrize("ignore_merges", [False, True])
def test_bpe_byte_alphabet(ignore_merges, tmp_path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {s: i for i, s in enumerate(alphabet)}
    vocab["Ã¶"] = len(vocab)  # A multi-byte character without a corresponding merge.
    reference = Tokenizer(models.BPE(vocab=vocab, merges=[], ignore_merges=ignore_merges))
    reference.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    reference.add_tokens(["特殊", "éé"])
    check_parity(reference, ["ö", "öx", " möglichen", "你好😀", "éé", "a特殊b"], tmp_path, True)


def test_bpe_overlapping_merges(tmp_path):
    import itertools

    reference = Tokenizer(
        models.BPE(
            vocab={s: i for i, s in enumerate(["a", "b", "aa", "ab", "ba", "aaa", "aab", "aba"])},
            merges=[("a", "a"), ("a", "b"), ("b", "a"), ("aa", "a"), ("aa", "b"), ("ab", "a")],
        )
    )
    texts = ["".join(chars) for n in range(1, 7) for chars in itertools.product("ab", repeat=n)]
    check_parity(reference, texts, tmp_path)


def test_bpe_tiktoken_bytes():
    from openvino_tokenizers.tokenizer_pipeline import (
        BPETokenizationStep,
        PaddingStep,
        RegexSplitStep,
        SpecialTokensSplit,
        TokenizerPipeline,
    )
    from tiktoken import Encoding

    ranks = {bytes([i]): i for i in range(256)}
    ranks.update({b"ab": 256, "ö".encode(): 257})
    encoding = Encoding("bpe-test", pat_str=r".+", mergeable_ranks=ranks, special_tokens={"<special>": 258})
    pipeline = TokenizerPipeline()
    pipeline.add_steps(
        [
            SpecialTokensSplit.from_string_iter(["<special>"]),
            RegexSplitStep(r".+", behaviour="contiguous"),
            BPETokenizationStep.from_tiktoken_encoding(encoding),
            PaddingStep(_token_id=0),
        ]
    )
    compiled = ov.compile_model(pipeline.get_tokenizer_ov_subgraph(), "CPU")
    for text in ["ab", "ö", "öx", "你好😀", "a<special>b"]:
        assert compiled([text])[0][0].tolist() == encoding.encode(text, allowed_special="all")
