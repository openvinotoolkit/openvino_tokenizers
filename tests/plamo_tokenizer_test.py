from types import SimpleNamespace

import numpy as np
import openvino as ov
import pytest
from openvino_tokenizers.plamo_parser import convert_plamo_tokenizer
from openvino_tokenizers.utils import TokenzierConversionParams


@pytest.fixture
def plamo_config():
    vocab = [
        ("<|plamo:unk|>", 0, "UNKNOWN"),
        ("<|plamo:bos|>", 0, "CONTROL"),
        ("<|plamo:pad|>", 0, "CONTROL"),
        ("。", -4.9746, "NORMAL"),
        ("。。。", -9.7236, "NORMAL"),
    ]
    vocab.extend((f"<0x{byte:02X}>", 0, "BYTE") for byte in range(256))
    return SimpleNamespace(
        data=vocab,
        added_tokens_decoder={
            i: SimpleNamespace(content=piece, lstrip=False, rstrip=False) for i, (piece, _, _) in enumerate(vocab[:3])
        },
        all_special_ids=[0, 1, 2],
        add_bos_token=True,
        bos_token_id=1,
        add_eos_token=False,
        pad_token_id=2,
        padding_side="left",
        truncation_side="right",
        model_max_length=128,
        break_around_repeated_chars_threshold=4,
        break_around_consecutive_spaces_threshold=2,
        clean_up_tokenization_spaces=False,
    )


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_plamo_ties_and_byte_fallback_after_ir_reload(tmp_path, plamo_config, skip_special_tokens):
    byte_offset = len(plamo_config.data) - 256
    tokenizer, detokenizer = convert_plamo_tokenizer(
        plamo_config,
        TokenzierConversionParams(
            with_detokenizer=True, add_special_tokens=False, skip_special_tokens=skip_special_tokens
        ),
    )
    tokenizer_path = tmp_path / "openvino_tokenizer.xml"
    detokenizer_path = tmp_path / "openvino_detokenizer.xml"
    ov.save_model(tokenizer, tokenizer_path)
    ov.save_model(detokenizer, detokenizer_path)
    core = ov.Core()
    encode = core.compile_model(core.read_model(tokenizer_path), "CPU")
    decode = core.compile_model(core.read_model(detokenizer_path), "CPU")

    samples = ["。。。。", "。。。。。", "🫠"]
    result = encode(samples)
    ids = result["input_ids"]
    mask = result["attention_mask"].astype(bool)
    assert [row[keep].tolist() for row, keep in zip(ids, mask)] == [
        [4, 3],
        [4, 3, 3],
        [byte_offset + byte for byte in "🫠".encode("utf-8")],
    ]
    assert [decode(row[keep][None, :])["string_output"][0] for row, keep in zip(ids, mask)] == samples
    # Feed the decoder known IDs directly, including tags and a malformed byte sequence.
    decoder_ids = np.array([[1, byte_offset + 0xE0, byte_offset + 0x80, 2]], dtype=np.int64)
    expected = "��" if skip_special_tokens else "<|plamo:bos|>��<|plamo:pad|>"
    assert decode(decoder_ids)["string_output"].tolist() == [expected]
    empty_batch = encode(["", ""])
    assert empty_batch["input_ids"].shape == (2, 0)
    assert empty_batch["attention_mask"].shape == (2, 0)


@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("model_max_length", [10**30, 262144])
def test_plamo_truncation_limit_fits_genai_state_after_ir_reload(
    tmp_path, plamo_config, add_special_tokens, model_max_length
):
    plamo_config.model_max_length = model_max_length
    model = convert_plamo_tokenizer(plamo_config, TokenzierConversionParams(add_special_tokens=add_special_tokens))
    path = tmp_path / "openvino_tokenizer.xml"
    ov.save_model(model, path)
    core = ov.Core()
    reloaded = core.read_model(path)
    combine = next(node for node in reloaded.get_ops() if node.get_type_name() == "CombineSegments")
    truncate = next(node for node in reloaded.get_ops() if node.get_type_name() == "Truncate")
    # GenAI adds the number of serialized special-token segments to the configured limit.
    # The BOS segment is still present when automatic insertion starts disabled.
    special_token_slots = (combine.get_input_size() - 1) // 3 - 1
    assert special_token_slots == 1
    configured_limit = int(truncate.input_value(3).get_node().input_value(1).get_node().get_data().item())
    assert configured_limit + special_token_slots <= np.iinfo(np.int32).max
    assert configured_limit == min(
        model_max_length - int(add_special_tokens), np.iinfo(np.int32).max - special_token_slots
    )

    encoded = core.compile_model(reloaded, "CPU")(["。"])["input_ids"].tolist()
    assert encoded == ([[1, 3]] if add_special_tokens else [[3]])
