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
        # 8 workers x 125 inferences = 1,000 concurrent inferences.
        for _ in range(125):
            result = request.infer([text])
            assert result[compiled.output("input_ids")].tolist()[0] == expected

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(infer_many, range(8)))
    """
)


def test_bpe_cache_concurrent_inference():
    run_subprocess(CONCURRENT_SCRIPT)


# The cache stores results of at most four token ids inline in the slot and spills
# longer ones to a side arena, and keys of at most 15 bytes inside the std::string
# small-string buffer. These scripts pin both boundaries and the sentinel/NUL rules
# through the production graph.
SLOT_BOUNDARY_SCRIPT = textwrap.dedent(
    r"""
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    # One token per character, so a key of N characters encodes to N tokens and the
    # key-length and value-length boundaries can be driven independently.
    letters = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"[UNK]": 0, **{ch: idx + 1 for idx, ch in enumerate(letters)}}
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model = convert_tokenizer(tokenizer, with_detokenizer=False, add_special_tokens=False)
    bpe_node = next(node for node in model.get_ordered_ops() if node.get_type_name() == "BPETokenizer")
    bpe_node.set_attribute("cache_capacity", 4096)
    compiled = ov.Core().compile_model(model, "CPU")

    def encode(text):
        return compiled([text])["input_ids"].tolist()[0]

    ids = {ch: vocab[ch] for ch in letters}

    # Key lengths straddling the 15-byte small-string boundary, and value lengths
    # straddling the 4-token inline boundary. Each is checked twice: the first pass
    # inserts (miss), the second reads back (hit).
    for length in (1, 3, 4, 5, 6, 14, 15, 16, 17, 31, 32, 33, 64):
        text = "".join(letters[i % 26] for i in range(length))
        expected = [ids[ch] for ch in text]
        assert encode(text) == expected, (length, "miss")
        assert encode(text) == expected, (length, "hit")
        # A third read, after other keys have been inserted and the spill arena has
        # grown, catches a stale pointer into reallocated storage.
        assert encode(text) == expected, (length, "rehit")

    # Repeated hits on a spilled (>4 token) value, interleaved with fresh inserts
    # that keep extending the arena.
    spilled = "".join(letters[i % 26] for i in range(200))
    spilled_expected = [ids[ch] for ch in spilled]
    for round_idx in range(20):
        filler = "".join(letters[(i + round_idx) % 26] for i in range(50 + round_idx))
        assert encode(filler) == [ids[ch] for ch in filler]
        assert encode(spilled) == spilled_expected, round_idx
    """
)


def test_bpe_cache_inline_and_spill_boundaries():
    run_subprocess(SLOT_BOUNDARY_SCRIPT)


NUL_KEY_SCRIPT = textwrap.dedent(
    r"""
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    # Keys containing an embedded NUL, including keys that are equal up to and
    # including a NUL and differ only after it. A cache that compared keys as
    # C strings, or hashed only up to the NUL, would confuse these. The last two
    # pairs put the distinguishing bytes on either side of the 15-byte
    # small-string boundary, so the NUL and the boundary are exercised together.
    #
    # Keys with a *trailing* or lone NUL are deliberately absent: the production
    # graph resolves those before BPE ("a\x00" -> [unk], "\x00" -> empty),
    # independently of caching, so they would not test the cache.
    pairs = [
        ("a\x00b", 1),
        ("a\x00c", 2),
        ("\x00a", 4),
        ("a\x00b\x00d", 6),
        ("a\x00b\x00e", 7),
        ("a\x00b" + "X" * 14, 8),
        ("a\x00b" + "Y" * 15, 9),
    ]
    vocab = {"[UNK]": 0, **{key: token for key, token in pairs}}
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model = convert_tokenizer(tokenizer, with_detokenizer=False, add_special_tokens=False)
    bpe_node = next(node for node in model.get_ordered_ops() if node.get_type_name() == "BPETokenizer")
    bpe_node.set_attribute("cache_capacity", 64)
    compiled = ov.Core().compile_model(model, "CPU")

    def encode(text):
        return compiled([text])["input_ids"].tolist()[0]

    # Two passes so the second reads every key back out of the cache.
    for _ in range(2):
        for key, token in pairs:
            assert encode(key) == [token], (key.encode(), token)
    """
)


def test_bpe_cache_embedded_nul_keys():
    run_subprocess(NUL_KEY_SCRIPT)


SATURATION_SCRIPT = textwrap.dedent(
    r"""
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    # Saturation must stay gated on entry count, not bytes: a denser slot may not
    # silently raise the number of cached entries. With capacity 3, keys beyond the
    # third are never cached but must still tokenize correctly, every time.
    letters = "abcdefghij"
    vocab = {"[UNK]": 0, **{ch: idx + 1 for idx, ch in enumerate(letters)}}
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model = convert_tokenizer(tokenizer, with_detokenizer=False, add_special_tokens=False)
    bpe_node = next(node for node in model.get_ordered_ops() if node.get_type_name() == "BPETokenizer")
    bpe_node.set_attribute("cache_capacity", 3)
    compiled = ov.Core().compile_model(model, "CPU")

    ids = {ch: vocab[ch] for ch in letters}
    # Mix of inline-sized and spill-sized results, more distinct keys than capacity.
    words = ["a", "ab", "abc", "abcd", "abcde", "abcdefghij", "bcd", "cde", "j"]
    for _ in range(3):
        for word in words:
            assert compiled([word])["input_ids"].tolist()[0] == [ids[ch] for ch in word], word
    # All of them at once, in one row.
    text = " ".join(words)
    expected = [ids[ch] for word in words for ch in word]
    assert compiled([text])["input_ids"].tolist()[0] == expected
    """
)


def test_bpe_cache_saturation_unchanged():
    run_subprocess(SATURATION_SCRIPT)
