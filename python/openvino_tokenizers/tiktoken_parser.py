from typing import Dict, List, Optional, Tuple

from tiktoken import Encoding

from .utils import bytes_to_unicode


#  https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee
def token_bytes_to_string(b: bytes) -> str:
    byte_encoder = bytes_to_unicode()
    return "".join(byte_encoder[ord(char)] for char in b.decode("latin-1"))


def bpe(mergeable_ranks: Dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> List[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        if min_idx is None:
            raise ValueError(f"Tiktoken conversion error: cannot determine bpe for token {token}.")
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def generate_vocab_and_merges(encoding: Encoding) -> Tuple[Dict[bytes, int], List[Tuple[bytes, bytes]], Dict[int, bytes]]:
    mergeable_ranks = encoding._mergeable_ranks

    vocab = {}
    merges = []
    added_tokens = {}

    for token, rank in mergeable_ranks.items():
        vocab[token] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))

        #  if special tokens added to the tokenizer and the bpe split might produce more than 2 tokens
        #  if there are "\t" in the vocab and special token "\t\t\t" was added before "\t\t" it will
        #  be tokenized into 3 tokens: bpe("\t\t\t") -> ["\t", "\t", "\t"] which is cannot be included
        #  in merges
        if len(merged) == 2:
            merges.append(merged)
        else:
            try:
                added_tokens[rank] = token.decode("latin-1")
            except UnicodeDecodeError:
                added_tokens[rank] = token

    # Also add special tokens
    vocab.update({string.encode(): idx for string, idx in encoding._special_tokens.items()})

    return vocab, merges, added_tokens
