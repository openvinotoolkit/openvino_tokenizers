import itertools

import pytest
from openvino_tokenizers.tokenizer_pipeline import RegexSplitStep
from tokenizers.pre_tokenizers import Split

from tests.layer_tests import create_splitting_model


@pytest.mark.parametrize(
    "mode,hf_mode", [("mergedwithnext", "merged_with_next"), ("mergedwithprevious", "merged_with_previous")]
)
@pytest.mark.parametrize("invert", [False, True])
def test_adjacent_split_matches_hugging_face(mode, hf_mode, invert):
    model = create_splitting_model(RegexSplitStep("-", behaviour=mode, invert=invert))
    reference = Split("-", hf_mode, invert=invert)
    for length in range(1, 7):
        for chars in itertools.product("a-", repeat=length):
            text = "".join(chars)
            expected = [piece for piece, _ in reference.pre_tokenize_str(text)]
            actual = model([text])[0].tolist()
            assert actual == expected, (text, actual, expected)
            assert "".join(actual) == text


@pytest.mark.parametrize(
    "mode,hf_mode", [("mergedwithnext", "merged_with_next"), ("mergedwithprevious", "merged_with_previous")]
)
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize(
    "pattern,texts",
    [
        ("▁", ["▁▁a", "a▁▁", "▁▁", "a▁▁b▁▁c", "unchanged"]),
        ("::", ["::::a", "a::::b", "a::::", "::::"]),
        ("-+", ["--a---b", "a---", "---", "a-b"]),
    ],
)
def test_split_boundaries(mode, hf_mode, invert, pattern, texts):
    from tokenizers import Regex

    model = create_splitting_model(RegexSplitStep(pattern, behaviour=mode, invert=invert))
    reference = Split(Regex(pattern), hf_mode, invert=invert)
    for text in texts:
        expected = [piece for piece, _ in reference.pre_tokenize_str(text)]
        for _ in range(2):
            actual = model([text])[0].tolist()
            assert actual == expected
            assert "".join(actual).encode() == text.encode()


def test_skip_flags_and_batch():
    import openvino as ov
    from openvino import Model, PartialShape, Type, op
    from openvino_tokenizers import _get_factory
    from openvino_tokenizers.tokenizer_pipeline import SpecialToken, SpecialTokensSplit, TokenizerPipeline

    node = op.Parameter(Type.string, PartialShape(["?"]))
    parts = _get_factory().create("StringTensorUnpack", node.outputs()).outputs()
    parts = TokenizerPipeline.add_ragged_dimension(parts)
    parts = SpecialTokensSplit([SpecialToken("<special>")]).get_ov_subgraph(parts)
    parts = RegexSplitStep("-", behaviour="mergedwithnext").get_ov_subgraph(parts)
    packed = _get_factory().create("StringTensorPack", parts[2:5]).outputs()
    model = ov.Core().compile_model(Model(packed + parts[:4] + parts[5:], [node]))
    result = list(model(["é--b<special>c--d", "--e"]).values())
    assert result[0].tolist() == ["é", "-", "-b", "<special>", "c", "-", "-d", "-", "-e"]
    assert result[-1].tolist() == [False, False, False, True, False, False, False, False, False]
    assert result[1].tolist() == [0, 7]
    assert result[2].tolist() == [7, 9]
    begins, ends = result[3], result[4]
    source = "é--b<special>c--d--e".encode()
    assert [source[a:b].decode() for a, b in zip(begins, ends)] == result[0].tolist()


@pytest.mark.parametrize("kind", ["metaspace", "next", "previous"])
def test_toy_tokenizer_parity(kind):
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "▁": 1, "a": 2, "b": 3}
    if kind == "previous":
        vocab.update({"a▁": 4, "b▁": 5})
        merges = [("a", "▁"), ("b", "▁")]
        normalizer = normalizers.Replace(" ", "▁")
        pre = pre_tokenizers.Split("▁", behavior="merged_with_previous")
    else:
        vocab.update({"▁a": 4, "▁b": 5})
        merges = [("▁", "a"), ("▁", "b")]
        normalizer = (
            normalizers.Sequence([])
            if kind == "metaspace"
            else normalizers.Sequence([normalizers.Replace(" ", "▁"), normalizers.Prepend("▁")])
        )
        pre = (
            pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
            if kind == "metaspace"
            else pre_tokenizers.Split("▁", behavior="merged_with_next")
        )
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    compiled = ov.Core().compile_model(convert_tokenizer(hf), "CPU")
    for text in ["a b", "a  b", "a   b", "  a", "a  ", "a\n    b"]:
        expected = hf(text, add_special_tokens=False)["input_ids"]
        actual = compiled([text])["input_ids"][0].tolist()
        assert actual == expected, (text, actual, expected)


@pytest.mark.parametrize("mode", ["mergedwithnext", "mergedwithprevious"])
def test_roundtrip_model(tmp_path, mode):
    import openvino as ov
    from openvino import Model, PartialShape, Type, op
    from openvino_tokenizers import _ext_path, _get_opset_factory
    from openvino_tokenizers.tokenizer_pipeline import TokenizerPipeline

    node = op.Parameter(Type.string, PartialShape(["?"]))
    parts = _get_opset_factory("opset15").create("StringTensorUnpack", node.outputs()).outputs()
    parts = TokenizerPipeline.add_ragged_dimension(parts)
    parts = RegexSplitStep("-", behaviour=mode).get_ov_subgraph(parts)
    output = _get_opset_factory("opset15").create("StringTensorPack", parts[2:5]).outputs()
    model = Model(output, [node])
    path = tmp_path / "split.xml"
    ov.save_model(model, path)
    core = ov.Core()
    core.add_extension(str(_ext_path))
    compiled = core.compile_model(core.read_model(path), "CPU")
    expected = [
        x
        for x, _ in Split(
            "-", "merged_with_next" if mode == "mergedwithnext" else "merged_with_previous"
        ).pre_tokenize_str("a--b")
    ]
    assert compiled(["a--b"])[0].tolist() == expected


@pytest.mark.parametrize("mode", ["mergedwithnext", "mergedwithprevious"])
def test_empty_input_preserves_existing_behavior(mode):
    model = create_splitting_model(RegexSplitStep("-", behaviour=mode))
    assert model([""])[0].tolist() == [""]


@pytest.mark.parametrize("mode,expected", [("mergedwithnext", ["", "a"]), ("mergedwithprevious", ["a"])])
def test_empty_piece_in_mixed_batch_preserves_existing_behavior(mode, expected):
    model = create_splitting_model(RegexSplitStep("-", behaviour=mode))
    assert model(["", "a"])[0].tolist() == expected
