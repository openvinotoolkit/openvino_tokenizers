import openvino as ov
import pytest
from openvino_tokenizers import _get_factory


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


def get_utf8_validate_subgraph(replace_mode) -> ov.CompiledModel:
    from openvino.runtime import op

    replace_mode = False if replace_mode == "ignore" else True
    input_node = op.Parameter(ov.Type.string, ov.PartialShape(["?"]))
    input_node.set_friendly_name("string_input")
    unpacked_ = _get_factory().create("StringTensorUnpack", input_node.outputs()).outputs()
    validated_ = _get_factory().create("UTF8Validate", unpacked_, {"replace_mode": replace_mode}).outputs()
    packed_ = _get_factory().create("StringTensorPack", validated_).outputs()

    ov_model = ov.Model(packed_, [input_node], "test_net")
    validator = core.compile_model(ov_model)
    return validator


@pytest.mark.parametrize("test_string", utf8_validate_strings)
@pytest.mark.parametrize("replace_mode", ["ignore", "replace"])
def test_utf8_validate(test_string, replace_mode):
    compiled_model = get_utf8_validate_subgraph(replace_mode)
    res_ov = compiled_model([test_string])[0]
    res_py = test_string.decode(errors=replace_mode)
    assert res_ov == res_py
