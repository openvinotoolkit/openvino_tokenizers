import json
import re
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Union

import numpy as np
import openvino as ov
import pytest
import requests
from openvino import Model, PartialShape, Type, op
from openvino_tokenizers import _get_factory, _get_opset_factory
from openvino_tokenizers.constants import UTF8ReplaceMode
from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser
from openvino_tokenizers.tokenizer_pipeline import (
    CaseFoldStep,
    CharsmapStep,
    DecodingStep,
    NormalizationStep,
    NormalizeUnicode,
    PreTokenizatinStep,
    RegexNormalizationStep,
    RegexSplitStep,
    SpecialToken,
    SpecialTokensSplit,
    TokenizerPipeline,
    UTF8ValidateStep,
)
from openvino_tokenizers.utils import TokenzierConversionParams

from tests.utils import get_hf_tokenizer


core = ov.Core()
UNICODE_TEST_FILE_URL = "https://www.unicode.org/Public/UCD/latest/ucd/NormalizationTest.txt"


class NormalizationTestLine(NamedTuple):
    source: str
    nfc: str
    nfd: str
    nfkc: str
    nfkd: str
    comment: str


def parse_normalization_test_line(line):
    parts, comment = line.split("#", 1)
    parts = [part.strip() for part in parts.split(";")]

    # Convert the hexadecimal Unicode code points to characters
    def hex_to_char(hex_str):
        return "".join(chr(int(code, 16)) for code in hex_str.split())

    # Parse the components
    source = hex_to_char(parts[0])
    nfc = hex_to_char(parts[1])
    nfd = hex_to_char(parts[2])
    nfkc = hex_to_char(parts[3])
    nfkd = hex_to_char(parts[4])

    return NormalizationTestLine(source, nfc, nfd, nfkc, nfkd, comment)


@pytest.fixture(scope="session")
def icu_test_data(request):
    return requests.get(UNICODE_TEST_FILE_URL).text


@pytest.fixture(scope="session")
def unicode_normalization_test_data(request, icu_test_data):
    # check https://www.unicode.org/Public/UCD/latest/ucd/NormalizationTest.txt for details
    return [
        parse_normalization_test_line(line)
        for line in icu_test_data.split("\n")
        if line and not line.startswith("#") and not line.startswith("@")
    ]


############################################
########## Test Normalizer Step ############
############################################

utf8_validate_strings = [
    # Valid sequences.
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


def create_normalization_model(layer: Union[NormalizationStep, DecodingStep]) -> ov.CompiledModel:
    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")

    output = _get_opset_factory("opset15").create("StringTensorUnpack", input_node.outputs()).outputs()
    output = layer.get_ov_subgraph(output)
    output = _get_opset_factory("opset15").create("StringTensorPack", output).outputs()
    normalizer = Model(output, [input_node], "normalizer")

    return core.compile_model(normalizer)


@pytest.mark.parametrize("test_string", utf8_validate_strings)
@pytest.mark.parametrize("replace_mode", ["ignore", "replace"])
def test_utf8_validate(test_string, replace_mode):
    utf_validation_node = UTF8ValidateStep(UTF8ReplaceMode(replace_mode))
    compiled_model = create_normalization_model(utf_validation_node)
    res_ov = compiled_model([test_string])[0]
    res_py = test_string.decode(errors=replace_mode)
    assert res_ov == res_py


tokenizers_with_charsmap = ["google/flan-t5-xxl"]


charsmap_test_strings = [
    "Henry \u2163  ①②③",
    "",
    pytest.param(" \t\n", marks=pytest.mark.xfail(reason="Whitespace is deleted by OV tokenizer, need a fix")),
]


@pytest.fixture(scope="session", params=tokenizers_with_charsmap, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_charsmap_sentencepiece_tokenizer(request):
    hf_tokenizer = get_hf_tokenizer(request, fast_tokenizer=True, trust_remote_code=True)
    if not hf_tokenizer.is_fast:
        pytest.skip("Fast tokenizer should use Rust backend.")

    return hf_tokenizer


@pytest.fixture(scope="session")
def unigram_model_json(request, hf_charsmap_sentencepiece_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_charsmap_sentencepiece_tokenizer.save_pretrained(tmpdir)

        tmpdir = Path(tmpdir)
        with open(tmpdir / "tokenizer.json") as tok_json:
            return json.load(tok_json)


@pytest.fixture(scope="session")
def precompiled_charsmap_json(request, unigram_model_json):
    return unigram_model_json["normalizer"]["normalizers"][0]


@pytest.mark.parametrize("test_string", charsmap_test_strings)
def test_charsmap_normalizartion(test_string, hf_charsmap_sentencepiece_tokenizer, precompiled_charsmap_json):
    charsmap_normalization_node = CharsmapStep.from_hf_step_json(precompiled_charsmap_json)
    compiled_model = create_normalization_model(charsmap_normalization_node)
    res_ov = compiled_model([test_string])[0][0]
    res_hf = hf_charsmap_sentencepiece_tokenizer.backend_tokenizer.normalizer.normalize_str(test_string)
    assert res_ov == res_hf


@pytest.mark.parametrize(
    "test_parameters",
    [
        # results for sentencepiece charsmap:
        ("NFC", 17325),  # failed examples: 2640
        ("NFD", 17736),  # failed examples: 2229
        ("NFKC", 17224),  # failed examples: 2741
        ("NFKD", 17619),  # failed examples: 2346
        # results for icu70:
        # ("NFC", 19875),  # failed examples: 90
        # ("NFD", 19851),  # failed examples: 114
        # ("NFKC", 19777),  # failed examples: 188
        # ("NFKD", 19753),  # failed examples: 212
        # results for huggingface tokenizers:
        # ("NFC", 19247),  # failed examples: 718
        # ("NFD", 19220),  # failed examples: 745
        # ("NFKC", 19077),  # failed examples: 888
        # ("NFKD", 19050),  # failed examples: 915
    ],
)
def test_unicode_normalization_model(test_parameters, unicode_normalization_test_data):
    normalization_type, positive_threshold = test_parameters
    normalizer_layer = NormalizeUnicode(normalization_type)
    compiled_model = create_normalization_model(normalizer_layer)
    positive, negative, no_transformation = 0, 0, 0
    for test_input in unicode_normalization_test_data:
        res_ov = compiled_model([test_input.source])[0][0].encode()
        expected = getattr(test_input, normalization_type.lower()).encode()
        positive += res_ov == expected
        negative += res_ov != expected
        no_transformation += test_input.source.encode() == expected

    assert positive == positive_threshold, (
        f"{normalization_type}\n"
        f"Positive: {positive}, expected: {positive_threshold}\n"
        f"Negative: {negative}, expected: {len(unicode_normalization_test_data) - positive_threshold}\n"
        f"No transformation: {no_transformation}, positive delta: {positive - no_transformation}"
    )


@pytest.mark.parametrize(
    "test_string, expected, is_uft8",
    [
        ("a", "a", True),
        ("a", "a", False),
        ("A", "a", True),
        ("A", "a", False),
        ("Ю", "ю", True),
        ("Ю", "Ю", False),
        ("Σ", "σ", True),
        ("Σ", "Σ", False),
        ("Hello World!", "hello world!", True),
        ("Hello World!", "hello world!", False),
    ],
)
def test_casefold_normalization(test_string, expected, is_uft8):
    casefold = CaseFoldStep("utf-8" if is_uft8 else "")
    compiled_model = create_normalization_model(casefold)
    res_ov = compiled_model([test_string])[0]
    assert res_ov == expected


@pytest.mark.parametrize(
    "test_string, expected, layer",
    [
        ("Hello world!", " Hello world!", RegexNormalizationStep.add_prefix_whitespace_regex()),
        (" Hello world!", " Hello world!", RegexNormalizationStep.add_prefix_whitespace_regex()),
        ("\tHello world!", "\tHello world!", RegexNormalizationStep.add_prefix_whitespace_regex()),
        ("Hello world!", " Hello world!", RegexNormalizationStep.add_prefix_whitespace_to_not_whitespace_regex()),
        (" Hello world!", " Hello world!", RegexNormalizationStep.add_prefix_whitespace_to_not_whitespace_regex()),
        ("\tHello world!", " \tHello world!", RegexNormalizationStep.add_prefix_whitespace_to_not_whitespace_regex()),
        ("\tHello", "▁\tHello", RegexNormalizationStep.prepend_regex("▁")),
        (  # test backward compatibility with old regex
            " ' declare",
            "'declare",
            RegexNormalizationStep(
                regex_search_pattern=r" ([\\.\\?\\!,])| ('[ms])| (') | ('[rv]e)| (n't)",
                replace_term=r"\1",
            ),
        ),
        ("", "", RegexNormalizationStep.prepend_regex("▁")),
        ("\n", "▁\n", RegexNormalizationStep.prepend_regex("▁")),
        ("n", "▁n", RegexNormalizationStep.prepend_regex("▁")),
        (" ", "▁ ", RegexNormalizationStep.prepend_regex("▁")),
        (  # test backward compatibility with old regex
            "\n",
            "▁\n",
            RegexNormalizationStep(
                regex_search_pattern=r"(^)(.)",
                replace_term=r"▁\2",
            ),
        ),
        (  # test backward compatibility with old regex
            "\n",
            "▁\n",
            RegexNormalizationStep(
                regex_search_pattern=r"(^)(.+)",
                replace_term=r"▁$2",
            ),
        ),
    ],
)
def test_regex_normalization(test_string, expected, layer):
    compiled_model = create_normalization_model(layer)
    res_ov = compiled_model([test_string])[0]
    assert res_ov[0] == expected


############################################
######## Test PreTokenizatin Step ##########
############################################


def create_splitting_model(layer: PreTokenizatinStep) -> ov.CompiledModel:
    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")

    output = _get_opset_factory("opset15").create("StringTensorUnpack", input_node.outputs()).outputs()
    output = TokenizerPipeline.add_ragged_dimension(output)
    output = layer.get_ov_subgraph(output)
    output = _get_opset_factory("opset15").create("StringTensorPack", output[2:5]).outputs()
    splitter = Model(output, [input_node], "splitter")

    return core.compile_model(splitter)


clip_regex_pattern = (
    r"<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
)
re_clip_splitter = re.compile(clip_regex_pattern)
clip_splitter = RegexSplitStep(clip_regex_pattern, invert=True)

text2image_prompts = [
    "Cinematic, a vibrant Mid-century modern dining area, colorful chairs and a sideboard, ultra realistic, many detail",
    "colibri flying near a flower, side view, forest background, natural light, photorealistic, 4k",
    "Illustration of an astronaut sitting in outer space, moon behind him",
    "A vintage illustration of a retro computer, vaporwave aesthetic, light pink and light blue",
    "A view from beautiful alien planet, very beautiful, surealism, retro astronaut on the first plane, 8k photo",
    "red car in snowy forest, epic vista, beautiful landscape, 4k, 8k",
    "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
    "cute cat 4k, high-res, masterpiece, best quality, soft lighting, dynamic angle",
    "A cat holding a sign that says hello OpenVINO",
    "A small cactus with a happy face in the Sahara desert.",
]


@pytest.mark.parametrize(
    "test_string, expected, layer",
    [
        ("Hello world!", ("Hello", "world", "!"), RegexSplitStep.whitespace_splitter()),
        ("Hello     world!", ("Hello", "world!"), RegexSplitStep.bert_whitespace_splitter()),
        ("", ("",), RegexSplitStep.whitespace_splitter()),
        *[(prompt, tuple(re_clip_splitter.findall(prompt)), clip_splitter) for prompt in text2image_prompts],
        (
            "▁one▁two▁three▁",
            ("▁one", "▁two", "▁three", "▁"),
            RegexSplitStep(split_pattern="▁", behaviour="mergedwithnext"),
        ),
        ("▁", ("▁",), RegexSplitStep(split_pattern="▁", behaviour="mergedwithnext")),
        ("No split pattern", ("No split pattern",), RegexSplitStep(split_pattern="▁", behaviour="mergedwithnext")),
        (
            "▁one▁two▁three▁",
            ("▁", "one▁", "two▁", "three▁"),
            RegexSplitStep(split_pattern="▁", behaviour="mergedwithprevious"),
        ),
        ("▁", ("▁",), RegexSplitStep(split_pattern="▁", behaviour="mergedwithprevious")),
        ("No split pattern", ("No split pattern",), RegexSplitStep(split_pattern="▁", behaviour="mergedwithprevious")),
        ("split", tuple("split"), RegexSplitStep.split_by_chars()),
        ("split by chars", tuple("split by chars"), RegexSplitStep.split_by_chars()),
    ],
)
def test_regex_split(test_string, expected, layer):
    compiled_model = create_splitting_model(layer)
    res_ov = compiled_model([test_string])[0]
    assert (res_ov == expected).all()


def create_special_tokens_split(special_tokens: List[SpecialToken]) -> ov.CompiledModel:
    layer = SpecialTokensSplit(special_tokens)

    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    output = _get_factory().create("StringTensorUnpack", input_node.outputs()).outputs()
    output = TokenizerPipeline.add_ragged_dimension(output)
    output = layer.get_ov_subgraph(output)
    output_string = _get_factory().create("StringTensorPack", output[2:5]).outputs()

    splitter = Model(output_string + output[-1:], [input_node], "splitter")
    return core.compile_model(splitter)


@pytest.mark.parametrize(
    "special_tokens, text, expected, expected_skips",
    [
        (
            [
                SpecialToken("<｜begin▁of▁sentence｜>"),
            ],
            "<｜begin▁of▁sentence｜> the user's <</SYS>>",
            ("<｜begin▁of▁sentence｜>", " the user's <</SYS>>"),
            [1, 0],
        ),
        (
            [SpecialToken("<｜begin▁of▁sentence｜>", strip_right=True)],
            "<｜begin▁of▁sentence｜>   the user's <</SYS>>",
            ("<｜begin▁of▁sentence｜>   ", "the user's <</SYS>>"),
            [1, 0],
        ),
        (
            [SpecialToken("<|eot_id|>", strip_left=True)],
            "    the user's <</SYS>>    <|eot_id|>",
            ("    the user's <</SYS>>", "    <|eot_id|>"),
            [0, 1],
        ),
        ([SpecialToken("    ")], "    def", ("    ", "def"), [1, 0]),
        ([SpecialToken("    ")], "    def  ", ("    ", "def  "), [1, 0]),
        ([SpecialToken("    ")], "    def    ", ("    ", "def", "    "), [1, 0, 1]),
    ],
)
def test_special_tokens_split(special_tokens, text, expected, expected_skips):
    compiled_model = create_special_tokens_split(special_tokens)
    res, skips = compiled_model([text]).values()
    assert (res == expected).all()
    assert (skips == expected_skips).all()


###############################################
########## Test Tokenization Model ############
###############################################


model_test_strings = [
    "test",
    "Hello world!",
]


@pytest.mark.parametrize("test_string", model_test_strings)
def test_unigram_model(test_string, hf_charsmap_sentencepiece_tokenizer):
    pipeline = TransformersTokenizerPipelineParser(
        hf_charsmap_sentencepiece_tokenizer, TokenzierConversionParams()
    ).parse()
    pipeline.steps = pipeline.steps[:7]
    unigram_model = pipeline.get_tokenizer_ov_subgraph()
    compiled_model = core.compile_model(unigram_model)

    res_ov = compiled_model([test_string])[compiled_model.output(2)]

    test_string = hf_charsmap_sentencepiece_tokenizer.backend_tokenizer.normalizer.normalize_str(test_string)
    test_string = hf_charsmap_sentencepiece_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(test_string)
    res_hf = [
        token.id
        for string in test_string
        for token in hf_charsmap_sentencepiece_tokenizer.backend_tokenizer.model.tokenize(string[0])
    ]
    assert (res_ov == res_hf).all()


################################################
########## Test PostTokenizatin Step ###########
################################################


@pytest.mark.parametrize(
    "input_values, expected",
    [
        (
            {
                "begins": [0, 3],
                "ends": [3, 8],
                "data": [10, 20, 100, 30, 40, 50, 200, 300],
                "padding_size": 10,
                "value": 42,
                "padding_side": "right",
            },
            [
                [10, 20, 100, 42, 42, 42, 42, 42, 42, 42],
                [30, 40, 50, 200, 300, 42, 42, 42, 42, 42],
            ],
        ),
        (
            {
                "begins": [0, 3],
                "ends": [3, 8],
                "data": [10, 20, 100, 30, 40, 50, 200, 300],
                "padding_size": 10,
                "value": 42,
                "padding_side": "left",
            },
            [
                [42, 42, 42, 42, 42, 42, 42, 10, 20, 100],
                [42, 42, 42, 42, 42, 30, 40, 50, 200, 300],
            ],
        ),
        (
            {
                "begins": [0, 3],
                "ends": [3, 8],
                "data": [10, 20, 100, 30, 40, 50, 200, 300],
                "padding_size": 2,
                "value": 42,
                "padding_side": "right",
            },
            [
                [10, 20],
                [30, 40],
            ],
        ),
    ],
)
def test_ragged_to_dense(input_values, expected):
    numeric_input_names = "begins", "ends", "data", "padding_size", "value"
    np_input_values = [np.array(input_values[key], dtype=np.int32) for key in numeric_input_names]

    # Parameter for all inputs except value
    input_params = [op.Parameter(Type.i32, PartialShape(["?"])) for _ in range(len(numeric_input_names) - 1)]
    # Parameter for value
    input_params = [*input_params, op.Parameter(Type.i32, PartialShape([]))]

    assert input_values["padding_side"] in ["right", "left"]
    pad_right = True if input_values["padding_side"] == "right" else False
    ragged_to_dense = _get_factory().create("RaggedToDense", input_params, {"pad_right": pad_right}).outputs()

    ragged_to_dense_model = Model(ragged_to_dense, input_params, "ragged_to_dense")
    compiled_model = core.compile_model(ragged_to_dense_model)

    res = compiled_model(np_input_values)
    assert np.all(res[0] == np.array(expected, dtype=np.int32))


@pytest.mark.parametrize(
    "input_values, expected",
    [
        (
            [
                {"begins": [0, 2], "ends": [2, 5], "data": [10, 20, 30, 40, 50]},  # [10, 20], [30, 40, 50]
                {"begins": [0, 1], "ends": [1, 3], "data": [100, 200, 300]},  # [100], [200, 300]
            ],
            {
                "begins": [0, 3],
                "ends": [3, 8],
                "data": [10, 20, 100, 30, 40, 50, 200, 300],
            },  # [[10, 20, 100], [30, 40, 50, 200, 300]]
        ),
        (
            [
                {"begins": [0, 2], "ends": [2, 5], "data": [10, 20, 30, 40, 50]},  # [10, 20], [30, 40, 50]
                {"begins": [0, 1], "ends": [1, 3], "data": [100, 200, 300]},  # [100], [200, 300]
                {"begins": [0, 2], "ends": [2, 3], "data": [1000, 2000, 3000]},  # [1000, 2000], [3000]
            ],
            {
                "begins": [0, 5],
                "ends": [5, 11],  # [[10, 20, 100, 1000, 2000], [30, 40, 50, 200, 300, 3000]]
                "data": [10, 20, 100, 1000, 2000, 30, 40, 50, 200, 300, 3000],
            },
        ),
    ],
)
def test_combine_segments(input_values, expected):
    numeric_input_names = "begins", "ends", "data"

    np_input_values = []
    for value in input_values:
        np_input_values.extend([np.array(value[k], dtype=np.int32) for k in numeric_input_names])
    np_input_values.append(np.arange(len(input_values), dtype=np.int32))

    input_params = [op.Parameter(Type.i32, PartialShape(["?"])) for _ in range(len(np_input_values))]
    combine_segments = _get_factory().create("CombineSegments", input_params).outputs()
    combine_segments_model = Model(combine_segments, input_params, "combine_segments")

    compiled_model = core.compile_model(combine_segments_model)
    res = compiled_model(np_input_values)
    for (_, val), (_, expect_val) in zip(res.items(), expected.items()):
        assert np.all(val == np.array(expect_val, dtype=np.int32))
