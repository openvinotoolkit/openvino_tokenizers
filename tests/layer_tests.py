import pytest
import openvino as ov
core = ov.Core()

utf8_validate_strings = [
    # Valid sequences.
    b"Eng... test, string?!",
    b"Eng... test, string?!", 
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
    bytes([0b11000000, 0b11000000])
    b'A\xC3\x28B'  # 'A' and 'B' are valid, \xC3\x28 is invalid
    # TODO: Add more invalid sequences as well.
]


def get_utf8_validate_subgraph(replace_mode) -> ov.CompiledModel:
    from openvino_tokenizers import _get_factory
    from openvino.runtime import op

    replace_mode = False if replace_mode == "ignore" else True
    input_node = ov.op.Parameter(ov.Type.string, ov.PartialShape(["?"]))
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
    res_ov = compiled_model([test_string])
    res_py = test_string.decode(errors=replace_mode)
    assert res_ov[0] == res_py
