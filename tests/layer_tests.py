import json
import re
import tempfile
from pathlib import Path
from typing import Union, List
import numpy as np
import openvino as ov
import pytest
from openvino import Model, PartialShape, Type
from openvino import op
from openvino_tokenizers import _get_factory
from openvino_tokenizers.constants import UTF8ReplaceMode
from openvino_tokenizers.tokenizer_pipeline import (
    CharsmapStep,
    DecodingStep,
    NormalizationStep,
    PreTokenizatinStep,
    RegexNormalizationStep,
    RegexSplitStep,
    TokenizerPipeline,
    UTF8ValidateStep,
)

from tests.utils import get_hf_tokenizer


core = ov.Core()

############################################
########## Test Normalizer Step ############
############################################

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


def create_normalization_model(layer: Union[NormalizationStep, DecodingStep]) -> ov.CompiledModel:
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
    utf_validation_node = UTF8ValidateStep(UTF8ReplaceMode(replace_mode))
    compiled_model = create_normalization_model(utf_validation_node)
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
    compiled_model = create_normalization_model(charsmap_normalization_node)
    res_ov = compiled_model([test_string])[0][0]
    res_hf = hf_charsmap_tokenizer.backend_tokenizer.normalizer.normalize_str(test_string)
    assert res_ov == res_hf


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
            )
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
            )
        ),
        (  # test backward compatibility with old regex
            "\n",
            "▁\n",
            RegexNormalizationStep(
                regex_search_pattern=r"(^)(.+)",
                replace_term=r"▁$2",
            )
        ),
    ]
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

    output = _get_factory().create("StringTensorUnpack", input_node.outputs()).outputs()
    output = TokenizerPipeline.add_ragged_dimension(output)
    output = layer.get_ov_subgraph(output)
    output = _get_factory().create("StringTensorPack", output[2:5]).outputs()
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


################################################
######## Test RaggedToDense Operation ##########
################################################

def infer_ragged_to_dense(test_inputs: List[List[int]], padding: str) -> List[np.array]:
    begins = op.Parameter(Type.i32, PartialShape(["?"]))
    ends = op.Parameter(Type.i32, PartialShape(["?"]))
    data = op.Parameter(Type.i32, PartialShape(["?"]))
    vaulue = op.Parameter(Type.i32, PartialShape(["?"]))
    padding_size = op.Parameter(Type.i32, PartialShape([]))

    input_params = [begins, ends, data, vaulue, padding_size]
    assert padding in ["right", "left"]
    pad_right = True if padding == "right" else False
    combine_segments = _get_factory().create("RaggedToDense", input_params, {"pad_right": pad_right}).outputs()

    ragged_to_dense_model = Model(combine_segments, input_params, "ragged_to_dense")
    # ov.save_model(ragged_to_dense_model, 'ragged_to_dense.xml')

    compiled_model = core.compile_model(ragged_to_dense_model)
    return compiled_model([np.array(inp, dtype=np.int32) for inp in test_inputs])


@pytest.mark.parametrize(
    "test_inputs, padding, expected",
    [   # beging, ends, data, padding_size, value
        (([0, 3], [3, 8] , [10, 20, 100, 30, 40, 50, 200, 300], 10, 42), 
         "right",
         [[10, 20, 100, 42, 42,  42,  42, 42, 42, 42],
          [30, 40, 50,  200, 300, 42, 42, 42, 42, 42]]),
        (([0, 3], [3, 8] , [10, 20, 100, 30, 40, 50, 200, 300], 10, 42), 
         "left",
         [[42, 42,  42,  42, 42, 42, 42, 10, 20, 100],
          [42, 42, 42, 42, 42, 30, 40, 50,  200, 300]]),
        # RaggedToDense can be used for truncation as well
        (([0, 3], [3, 8] , [10, 20, 100, 30, 40, 50, 200, 300], 2, 42), 
         "right",
         [[10, 20],
          [30, 40]]),
    ]
)
def test_ragged_to_dense(test_inputs, padding, expected):
    res = infer_ragged_to_dense(test_inputs, padding)
    assert np.all(res[0] == np.array(expected, dtype=np.int32))


##################################################
######## Test CombinedSegments Operation #########
##################################################

def infer_combine_segments(test_inputs: List[List[int]]) -> ov.CompiledModel:
    num_of_segments = int(len(test_inputs) // 3)
    
    input_tensors = [np.array(x, dtype=np.int32) for x in test_inputs]
    input_tensors.append(np.arange(num_of_segments, dtype=np.int32))
    input_params = [op.Parameter(Type.i32, PartialShape(["?"])) for _ in range(len(input_tensors))]

    combine_segments = _get_factory().create("CombineSegments", input_params).outputs()
    combine_segments_model = Model(combine_segments, input_params, "combine_segments")

    compiled_model = core.compile_model(combine_segments_model)
    return compiled_model(input_tensors)


@pytest.mark.parametrize(
    "test_inputs, expected",
    [
        (([0, 2], [2, 5], [10, 20, 30, 40, 50], # [10, 20], [30, 40, 50]
          [0, 1], [1, 3], [100, 200, 300]),     # [100], [200, 300]
         ([0, 3], [3, 8] , [10, 20, 100, 30, 40, 50, 200, 300]),  # [[10, 20, 100], [30, 40, 50, 200, 300]]
        ),
        (([0, 2], [2, 5], [10, 20, 30, 40, 50], # [10, 20], [30, 40, 50]
          [0, 1], [1, 3], [100, 200, 300],     # [100], [200, 300]
          [0, 2], [2, 3], [1000, 2000, 3000]),  # [1000, 2000], [3000]
         ([0, 5], [5, 11] , [10, 20, 100, 1000, 2000, 30, 40, 50, 200, 300, 3000]),  # [[10, 20, 100, 1000, 2000], [30, 40, 50, 200, 300, 3000]]
        )
    ]
)
def test_combine_segments(test_inputs, expected):
    res = infer_combine_segments(test_inputs)
    for (k, value), expect_val in zip(res.items(), expected):
        assert np.all(value == np.array(expect_val, dtype=np.int32))
