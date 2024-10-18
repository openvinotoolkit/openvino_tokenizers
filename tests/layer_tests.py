import json
import tempfile
from pathlib import Path
from typing import Union

import openvino as ov
import pytest
from openvino import Model, PartialShape, Type
from openvino.runtime import op
from openvino_tokenizers import _get_factory
from openvino_tokenizers.constants import UTF8ReplaceMode
from openvino_tokenizers.tokenizer_pipeline import CharsmapStep, DecodingStep, NormalizationStep, UTF8ValidateStep

from tests.utils import get_hf_tokenizer


core = ov.Core()


utf8_validate_strings = [
    # Valid sequences.
    b"Eng... test, string?!",
    b"Eng... test, string?!",
    b"\xe2\x82\xac",  # Euro sign €ß
    "Проверка, как работает кириллица Љ љ Ђ ђ".encode(),
    "測試字符串".encode(),
    "Tester, la chaîne...".encode(),
    "سلسلة الاختبار".encode(),
    "מחרוזת בדיקה".encode(),
    "Сынақ жолы á".encode(),
    "😁😁".encode(),
    "🤣🤣🤣😁😁😁😁".encode(),
    "🫠".encode(),
    "介绍下清华大学".encode(),
    "折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。".encode(),
    # Invalid sequences.
    b"\x81First byte is invalid utf8",
    b"\x80\x80\x80",  # Bytes 0x80 are valid as continuation bytes, but not as a start byte
    bytes([0b11000000, 0b11000000, 0b11000000]),
    bytes(
        [0b11110000, 0b10010011, 0b10000001, 0b11101000, 0b11110000, 0b10010011, 0b10000001, 0b10101000]
    ),  # 4th byte is invalid continuation
    bytes(
        [0b11110000, 0b10011111, 0b10011000, 0b11000001, 0b11110000, 0b10011111, 0b10011000, 0b10000001]
    ),  # 4th byte is invalid continuation
    b"\xc0\x80",  # 2 bytes sequence but codepoint is less than 0x80
    b"\xe0\x81\x81",  # 3 bytes sequence but codepoint is less than 0x800
    b"\xf0\x80\x80\x80",  # 4 bytes sequence but codepoint is less than 0x1000
    b"\xe2\x28\xa1",  # \x28 is not a valid continuation byte
    b"the following block is invalid \xe2\x28\xa1 but this text is valid",  # \x28 is not a valid continuation byte
    b"A\xc3\x28B",  # 'A' and 'B' are valid \x28 is invalid
    b"\xe2\x82",  # 3 byte symbol but is incomplete
    b"A\xc3\xa9\xe2\x82\xac\xf0\x90\x8d\x88",  # Mix of ASCII, 2-byte, 3-byte, and 4-byte characters
]


def crate_normalization_model(layer: Union[NormalizationStep, DecodingStep]) -> ov.CompiledModel:
    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")

    output = _get_factory().create("StringTensorUnpack", input_node.outputs()).outputs()
    output = layer.get_ov_subgraph(output)
    output = _get_factory().create("StringTensorPack", output).outputs()
    normalizer = Model(output, [input_node], "normalizer")

    return core.compile_model(normalizer)


@pytest.mark.parametrize("test_string", utf8_validate_strings)
@pytest.mark.parametrize("replace_mode", ["ignore", "replace"])
def test_utf8_validate(test_string, replace_mode):
    utf_validation_node = UTF8ValidateStep(
        UTF8ReplaceMode.REPLACE if replace_mode == "replace" else UTF8ReplaceMode.IGNORE
    )
    compiled_model = crate_normalization_model(utf_validation_node)
    res_ov = compiled_model([test_string])[0]
    res_py = test_string.decode(errors=replace_mode)
    assert res_ov == res_py


tokenizers_with_charsmap = ["google/flan-t5-xxl"]


charsmap_test_strings = ["Henry \u2163  ①②③", ""]


@pytest.fixture(scope="session", params=tokenizers_with_charsmap, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_charsmap_tokenizer(request):
    hf_tokenizer = get_hf_tokenizer(request, fast_tokenizer=True, trust_remote_code=True)
    if not hf_tokenizer.is_fast:
        pytest.skip("Fast tokenizer should use Rust backend.")

    return hf_tokenizer


@pytest.fixture(scope="session")
def precompiled_charsmap_json(request, hf_charsmap_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_charsmap_tokenizer.save_pretrained(tmpdir)

        tmpdir = Path(tmpdir)
        with open(tmpdir / "tokenizer.json") as tok_json:
            tj = json.load(tok_json)
            return tj["normalizer"]["normalizers"][0]


@pytest.mark.parametrize("test_string", charsmap_test_strings)
def test_charsmap_normalizartion(test_string, hf_charsmap_tokenizer, precompiled_charsmap_json):
    charsmap_normalization_node = CharsmapStep.from_hf_step_json(precompiled_charsmap_json)
    compiled_model = crate_normalization_model(charsmap_normalization_node)
    res_ov = compiled_model([test_string])[0][0]
    res_hf = hf_charsmap_tokenizer.backend_tokenizer.normalizer.normalize_str(test_string)
    assert res_ov == res_hf
