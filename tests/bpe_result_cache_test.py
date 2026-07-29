import subprocess
import sys
import textwrap


def run_subprocess(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


COLLISION_SCRIPT = textwrap.dedent(
    """
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    keys = ["k0", "k8", "k13", "k22"]

    def cache_hash(text):
        value = 14695981039346656037
        for byte in text.encode():
            value ^= byte
            value = (value * 1099511628211) & ((1 << 64) - 1)
        return value

    assert len({cache_hash(key) & 7 for key in keys}) == 1
    vocab = {"[UNK]": 0, **{key: idx + 100 for idx, key in enumerate(keys)}}
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model = convert_tokenizer(tokenizer, with_detokenizer=False, add_special_tokens=False)
    bpe_node = next(node for node in model.get_ordered_ops() if node.get_type_name() == "BPETokenizer")
    bpe_node.set_attribute("cache_capacity", 4)
    compiled = ov.Core().compile_model(model, "CPU")

    expected = [[vocab[key]] for key in keys]
    first = [compiled([key])["input_ids"].tolist()[0] for key in keys]
    second = [compiled([key])["input_ids"].tolist()[0] for key in keys]
    assert first == expected
    assert second == expected
    """
)


def test_bpe_cache_collision_chain():
    run_subprocess(COLLISION_SCRIPT)


EDGE_CASE_SCRIPT = textwrap.dedent(
    r"""
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "a": 1, "b": 2}
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model = convert_tokenizer(tokenizer, with_detokenizer=False, add_special_tokens=False)
    bpe_node = next(node for node in model.get_ordered_ops() if node.get_type_name() == "BPETokenizer")
    bpe_node.set_attribute("cache_capacity", 4)
    compiled = ov.Core().compile_model(model, "CPU")

    long_key = "ab" * 128
    long_value = [1, 2] * 128
    cases = [
        (long_key, long_value),
        ("a", [1]),
    ]
    for text, expected in cases:
        assert compiled([text])["input_ids"].tolist()[0] == expected
        assert compiled([text])["input_ids"].tolist()[0] == expected

    # Empty strings are removed before BPETokenizer by the production graph.
    assert compiled([""])["input_ids"].shape[-1] == 0
    """
)


def test_bpe_cache_empty_and_long_values():
    run_subprocess(EDGE_CASE_SCRIPT)


CONCURRENT_SCRIPT = textwrap.dedent(
    """
    from concurrent.futures import ThreadPoolExecutor

    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "a": 1, "b": 2}
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model = convert_tokenizer(tokenizer, with_detokenizer=False, add_special_tokens=False)
    bpe_node = next(node for node in model.get_ordered_ops() if node.get_type_name() == "BPETokenizer")
    bpe_node.set_attribute("cache_capacity", 2)
    compiled = ov.Core().compile_model(model, "CPU")
    expected = [1, 2] * 32
    text = " ".join(["ab"] * 32)

    def infer_many(_):
        request = compiled.create_infer_request()
        for _ in range(50):
            result = request.infer([text])
            assert result[compiled.output("input_ids")].tolist()[0] == expected

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(infer_many, range(8)))
    """
)


def test_bpe_cache_concurrent_inference():
    run_subprocess(CONCURRENT_SCRIPT)
