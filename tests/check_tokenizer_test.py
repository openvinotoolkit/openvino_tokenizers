from openvino_tokenizers.cli_tools.check_tokenizer import _genai_encode_params


def test_genai_encode_params_match_checker_options():
    assert _genai_encode_params(True, False, None, None, False) == {
        "add_special_tokens": True,
        "pad_to_max_length": False,
        "truncation": False,
    }
    assert _genai_encode_params(False, True, 32, "left", True) == {
        "add_special_tokens": False,
        "pad_to_max_length": True,
        "max_length": 32,
        "padding_side": "left",
        "truncation": True,
    }
