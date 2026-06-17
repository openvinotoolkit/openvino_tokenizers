# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Tests for the ai.onnx.contrib ONNX Frontend conversion extensions registered
# by openvino_tokenizers:
#   * SentencepieceTokenizer (6-input and optional 7-input "fairseq" variant)
#   * SentencepieceDecoder
#   * VectorToString
#   * StringJoin
#   * StringSplit
#
# A tiny SentencePiece model is trained on the fly so the SentencePiece tests
# are self-contained and do not require any network access. The SentencePiece
# library itself is used as the reference implementation; the string ops are
# checked against hand-computed expected values.

import tempfile
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

# Importing the package patches openvino.Core to register the tokenizers
# extension (including the ONNX Frontend conversion extensions under test).
import openvino_tokenizers  # noqa: F401

onnx = pytest.importorskip("onnx")
spm = pytest.importorskip("sentencepiece")

from onnx import TensorProto, helper  # noqa: E402


_TRAIN_WORDS = (
    "the quick brown fox jumps over a lazy dog hello world openvino "
    "tokenizers sentencepiece model training corpus example text data "
    "machine learning inference engine neural network compute graph"
).split()


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory):
    import random

    tmp = tmp_path_factory.mktemp("spm")
    corpus = tmp / "corpus.txt"
    rng = random.Random(0)
    with corpus.open("w") as f:
        for _ in range(400):
            n = rng.randint(3, 10)
            f.write(" ".join(rng.choice(_TRAIN_WORDS) for _ in range(n)) + "\n")

    prefix = tmp / "spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=48,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
    )
    model_bytes = (prefix.with_suffix(".model")).read_bytes()
    processor = spm.SentencePieceProcessor(model_file=str(prefix.with_suffix(".model")))
    return model_bytes, processor


def _scalar_const(name, value, dtype):
    arr = np.asarray(value)
    return helper.make_node(
        "Constant",
        [],
        [name],
        value=helper.make_tensor(name, dtype, [1], arr.ravel().tolist()),
    )


def _build_tokenizer_model(model_bytes, *, add_bos, add_eos, reverse, fairseq=None):
    nodes = [
        _scalar_const("nbest", [0], TensorProto.INT64),
        _scalar_const("alpha", [0.0], TensorProto.FLOAT),
        _scalar_const("add_bos", [add_bos], TensorProto.BOOL),
        _scalar_const("add_eos", [add_eos], TensorProto.BOOL),
        _scalar_const("reverse", [reverse], TensorProto.BOOL),
    ]
    tok_inputs = ["text", "nbest", "alpha", "add_bos", "add_eos", "reverse"]
    if fairseq is not None:
        nodes.append(_scalar_const("fairseq", [fairseq], TensorProto.BOOL))
        tok_inputs.append("fairseq")
    nodes.append(
        helper.make_node(
            "SentencepieceTokenizer",
            tok_inputs,
            ["tokens", "row_splits"],
            domain="ai.onnx.contrib",
            model=model_bytes,
        )
    )
    graph = helper.make_graph(
        nodes,
        "tokenizer",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [None])],
        [
            helper.make_tensor_value_info("tokens", TensorProto.INT32, [None]),
            helper.make_tensor_value_info("row_splits", TensorProto.INT64, [None]),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.contrib", 1)],
    )


def _build_decoder_model(model_bytes, *, fairseq=False):
    nodes = [
        _scalar_const("fairseq", [fairseq], TensorProto.BOOL),
        helper.make_node(
            "SentencepieceDecoder",
            ["ids", "fairseq"],
            ["text"],
            domain="ai.onnx.contrib",
            model=model_bytes,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "decoder",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [None, None])],
        [helper.make_tensor_value_info("text", TensorProto.STRING, [None])],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.contrib", 1)],
    )


def _build_vector_to_string_model(map_text, unk):
    node = helper.make_node(
        "VectorToString",
        ["ids"],
        ["text"],
        domain="ai.onnx.contrib",
        map=map_text,
        unk=unk,
    )
    graph = helper.make_graph(
        [node],
        "vector_to_string",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [None])],
        [helper.make_tensor_value_info("text", TensorProto.STRING, [None])],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.contrib", 1)],
    )


def _build_string_join_model():
    node = helper.make_node(
        "StringJoin", ["inp", "sep", "axis"], ["joined"], domain="ai.onnx.contrib"
    )
    graph = helper.make_graph(
        [node],
        "string_join",
        [
            helper.make_tensor_value_info("inp", TensorProto.STRING, [None]),
            helper.make_tensor_value_info("sep", TensorProto.STRING, []),
            helper.make_tensor_value_info("axis", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("joined", TensorProto.STRING, [None])],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.contrib", 1)],
    )


def _build_string_split_model():
    node = helper.make_node(
        "StringSplit",
        ["inp", "delim", "skip"],
        ["indices", "values", "shape"],
        domain="ai.onnx.contrib",
    )
    graph = helper.make_graph(
        [node],
        "string_split",
        [
            helper.make_tensor_value_info("inp", TensorProto.STRING, [None]),
            helper.make_tensor_value_info("delim", TensorProto.STRING, []),
            helper.make_tensor_value_info("skip", TensorProto.BOOL, []),
        ],
        [
            helper.make_tensor_value_info("indices", TensorProto.INT64, [None, None]),
            helper.make_tensor_value_info("values", TensorProto.STRING, [None]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [None]),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.contrib", 1)],
    )


def _save(model, tmp_path, name):
    path = tmp_path / name
    onnx.save(model, str(path))
    return str(path)


def _run(onnx_path, feed):
    core = ov.Core()
    compiled = core.compile_model(core.read_model(onnx_path), "CPU")
    request = compiled.create_infer_request()
    for port in compiled.inputs:
        request.set_tensor(port, feed[port.get_any_name()])
    request.infer()
    return [request.get_output_tensor(i) for i in range(len(compiled.outputs))]


TEST_STRINGS = ["Hello world.", "openvino tokenizers", "the quick brown fox"]


@pytest.mark.parametrize("text", TEST_STRINGS)
@pytest.mark.parametrize(
    "add_bos, add_eos, reverse",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
    ],
)
def test_sentencepiece_tokenizer(spm_model, tmp_path, text, add_bos, add_eos, reverse):
    model_bytes, processor = spm_model
    onnx_path = _save(
        _build_tokenizer_model(model_bytes, add_bos=add_bos, add_eos=add_eos, reverse=reverse),
        tmp_path,
        "tok.onnx",
    )

    outputs = _run(onnx_path, {"text": ov.Tensor(np.array([text], dtype=str))})
    tokens = outputs[0].data.tolist()

    expected = processor.encode(text, add_bos=add_bos, add_eos=add_eos)
    if reverse:
        expected = expected[::-1]

    assert tokens == expected
    assert outputs[1].data.tolist() == [0, len(expected)]


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_sentencepiece_tokenizer_fairseq_input(spm_model, tmp_path, text):
    # The optional 7th input (fairseq) is accepted when set to False and must
    # produce the same result as the 6-input form.
    model_bytes, processor = spm_model
    onnx_path = _save(
        _build_tokenizer_model(
            model_bytes, add_bos=True, add_eos=True, reverse=False, fairseq=False
        ),
        tmp_path,
        "tok7.onnx",
    )

    outputs = _run(onnx_path, {"text": ov.Tensor(np.array([text], dtype=str))})
    assert outputs[0].data.tolist() == processor.encode(text, add_bos=True, add_eos=True)


def test_sentencepiece_tokenizer_fairseq_true_unsupported(spm_model, tmp_path):
    # fairseq-mode id remapping is not implemented by the underlying op, so
    # conversion must fail fast rather than silently produce wrong results.
    model_bytes, _ = spm_model
    onnx_path = _save(
        _build_tokenizer_model(
            model_bytes, add_bos=True, add_eos=True, reverse=False, fairseq=True
        ),
        tmp_path,
        "tok_fairseq.onnx",
    )
    with pytest.raises(Exception, match="fairseq"):
        ov.Core().read_model(onnx_path)


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_sentencepiece_decoder(spm_model, tmp_path, text):
    model_bytes, processor = spm_model
    ids = processor.encode(text, add_bos=True, add_eos=True)
    onnx_path = _save(_build_decoder_model(model_bytes), tmp_path, "dec.onnx")

    outputs = _run(onnx_path, {"ids": ov.Tensor(np.array([ids], dtype=np.int64))})
    decoded = outputs[0].str_data[0]

    assert decoded == processor.decode(ids)


def test_sentencepiece_decoder_fairseq_true_unsupported(spm_model, tmp_path):
    model_bytes, _ = spm_model
    onnx_path = _save(_build_decoder_model(model_bytes, fairseq=True), tmp_path, "dec_fairseq.onnx")
    with pytest.raises(Exception, match="fairseq"):
        ov.Core().read_model(onnx_path)


def test_vector_to_string(tmp_path):
    # id -> string lookup; out-of-range ids map to the `unk` token.
    vocab = ["a", "b", "c", "hello", "world"]
    map_text = "\n".join(f"{tok}\t{idx}" for idx, tok in enumerate(vocab))
    onnx_path = _save(_build_vector_to_string_model(map_text, "<unk>"), tmp_path, "v2s.onnx")

    ids = [3, 4, 0, 99, -1]
    outputs = _run(onnx_path, {"ids": ov.Tensor(np.array(ids, dtype=np.int64))})

    expected = [vocab[i] if 0 <= i < len(vocab) else "<unk>" for i in ids]
    assert list(outputs[0].str_data) == expected


def test_string_join(tmp_path):
    onnx_path = _save(_build_string_join_model(), tmp_path, "join.onnx")
    parts = ["hello", "world", "foo"]
    outputs = _run(
        onnx_path,
        {
            "inp": ov.Tensor(np.array(parts, dtype=str)),
            "sep": ov.Tensor(np.array(" ", dtype=str)),
            "axis": ov.Tensor(np.array(0, dtype=np.int64)),
        },
    )
    assert outputs[0].str_data.ravel().tolist() == [" ".join(parts)]


def test_string_split(tmp_path):
    onnx_path = _save(_build_string_split_model(), tmp_path, "split.onnx")
    strings = ["a b c", "d e"]
    outputs = _run(
        onnx_path,
        {
            "inp": ov.Tensor(np.array(strings, dtype=str)),
            "delim": ov.Tensor(np.array(" ", dtype=str)),
            "skip": ov.Tensor(np.array(True)),
        },
    )
    indices, values, dense_shape = outputs

    expected_values = []
    expected_indices = []
    max_cols = 0
    for row, s in enumerate(strings):
        cols = s.split(" ")
        max_cols = max(max_cols, len(cols))
        for col, token in enumerate(cols):
            expected_indices.append([row, col])
            expected_values.append(token)

    assert list(values.str_data) == expected_values
    assert indices.data.tolist() == expected_indices
    assert dense_shape.data.tolist() == [len(strings), max_cols]


def test_string_split_skip_empty_preserves_original_positions(tmp_path):
    # When skip_empty=True and consecutive delimiters produce empty tokens,
    # the sparse COO indices must reflect the original (pre-skip) slot positions,
    # not compressed filtered positions. dense_shape last dim must also be the
    # max original token count (3 here), not the filtered count (2).
    onnx_path = _save(_build_string_split_model(), tmp_path, "split_skip.onnx")
    # "a  b" split on " " → ["a", "", "b"]; skip_empty drops ""; original positions 0, 2
    strings = ["a  b", "x"]
    outputs = _run(
        onnx_path,
        {
            "inp": ov.Tensor(np.array(strings, dtype=str)),
            "delim": ov.Tensor(np.array(" ", dtype=str)),
            "skip": ov.Tensor(np.array(True)),
        },
    )
    indices, values, dense_shape = outputs

    assert list(values.str_data) == ["a", "b", "x"]
    assert indices.data.tolist() == [[0, 0], [0, 2], [1, 0]]
    assert dense_shape.data.tolist() == [2, 3]
