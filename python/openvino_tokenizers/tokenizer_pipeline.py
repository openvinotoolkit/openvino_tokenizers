# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import weakref
from copy import copy
from dataclasses import dataclass, field
from functools import singledispatchmethod
from itertools import chain, islice, takewhile, groupby
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from openvino.runtime import Model, Output, PartialShape, Type, op
from openvino.runtime import opset12 as opset
from openvino.runtime.exceptions import OVTypeError, UserInputError
from openvino.runtime.utils.types import as_node, make_constant_node

from . import _get_factory
from .constants import (
    ATTENTION_MASK_INPUT_NAME,
    DETOKENIZER_NAME,
    MIN_CACHE_CAPACITY,
    STRING_OUTPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TOKEN_TYPE_IDS_INPUT_NAME,
    TOKENIZER_NAME,
    VOCAB_SIZE_CACHE_PROPORTION,
    UTF8ReplaceMode
)
from .str_pack import pack_string, pack_strings
from .utils import apply_bytes_to_unicode, generate_tokens_with_space_symbols, has_incompatible_re2_op


logger = logging.getLogger(__name__)


@dataclass
class BasePipelineStep:
    _pipeline: Optional[weakref.ReferenceType["TokenizerPipeline"]] = field(default=None, init=False, repr=False)

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self.get_config().items())
        return f"{self.__class__.__name__}({params_string})"

    def get_config(self) -> Dict[str, Any]:
        config = {key: value for key, value in vars(self).items() if not key.startswith("_")}
        properties = {
            key: getattr(self, key)
            for key in dir(type(self))
            if not key.startswith("_") and isinstance(getattr(type(self), key), property)
        }
        config.update(properties)
        return config

    def get_pipeline(self) -> Optional["TokenizerPipeline"]:
        return self._pipeline() if self._pipeline is not None else None

    def set_pipeline(self, pipeline: "TokenizerPipeline") -> None:
        self._pipeline = weakref.ref(pipeline)

    def get_ov_subgraph(self, *input_nodes: List[Output]) -> List[Output]:
        raise NotImplementedError

    @staticmethod
    def create_string_constant_node(value: Union[str, Iterable[str]]) -> op.Constant:
        if isinstance(value, str):
            # string scalar
            ps = pack_string(value)
            return op.Constant(ps)
        else:
            # support only 1D strings for now
            ps = pack_strings(value)
            return _get_factory().create("StringTensorUnpack", op.Constant(ps).outputs())

    def finalize(self) -> None:
        """Called after the entire pipeline has been built"""
        return


@dataclass
class NormalizationStep(BasePipelineStep):
    pass


@dataclass
class NormalizeUnicode(NormalizationStep):
    normalization_form: str = "NFD"

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return (
            _get_factory()
            .create(
                "NormalizeUnicode",
                input_nodes,
                {"normalization_form": self.normalization_form},
            )
            .outputs()
        )


@dataclass
class CaseFoldStep(NormalizationStep):
    #  attribute from tf.StringLower operation
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.encoding not in ["", "utf-8"]:
            raise ValueError(
                f"[ CaseFoldStep ] `encoding` attribute must be one of ['', 'utf-8'], got {self.encoding!r}."
            )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return _get_factory().create("CaseFold", input_nodes, {"encoding": self.encoding}).outputs()


@dataclass
class RegexNormalizationStep(NormalizationStep):
    regex_search_pattern: str
    replace_term: str
    global_replace: bool = True

    def __post_init__(self):
        self.vet_search_pattern()

    def vet_search_pattern(self) -> None:
        if has_incompatible_re2_op(self.regex_search_pattern):
            logger.warning(
                "RegexNormalization pattern is not supported, operation output might differ from the original tokenizer."
            )

    @classmethod
    def strip_accents_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"\p{Mn}", replace_term="")

    @classmethod
    def add_prefix_whitespace_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"^(\S)", replace_term=r" \1")

    @classmethod
    def add_prefix_whitespace_to_not_whitespace_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"^([^ ])", replace_term=r" \1")

    @classmethod
    def replace_spaces_metaspace(cls, replace_term=r"▁") -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r" ", replace_term=replace_term)

    @classmethod
    def prepend_regex(cls, string: str) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"(^)(.+)", replace_term=rf"{string}\2")

    @classmethod
    def prepend_with_check_regex(cls, string: str, check_string: str) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=rf"(^)([^{check_string}])", replace_term=rf"{string}\2")

    @classmethod
    def del_control_chars_regex(cls) -> "RegexNormalizationStep":
        return cls(
            regex_search_pattern=r"([\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F])",  # exclude \n\t\r
            replace_term="",
        )

    @classmethod
    def clean_up_tokenization_spaces(cls) -> "RegexNormalizationStep":
        return cls(
            regex_search_pattern=r" ([\.\?\!\,])| ('[ms])| (') | ('[rv]e)",
            replace_term="\1",
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(
            (
                self.create_string_constant_node(self.regex_search_pattern),
                self.create_string_constant_node(self.replace_term),
            )
        )
        return (
            _get_factory().create("RegexNormalization", input_nodes, {"global_replace": self.global_replace}).outputs()
        )


@dataclass
class NMTNormalizationStep(NormalizationStep):
    """Normaization based on NMT task.

    https://github.com/huggingface/tokenizers/blob/28cd3dce2a75d106572392194ff2564574c33235/tokenizers/src/normalizers/unicode.rs#L44
    """


@dataclass
class StripStringStep(NormalizationStep):
    left: bool
    right: bool


@dataclass
class PreTokenizatinStep(BasePipelineStep):
    pass


@dataclass
class RegexSplitStep(PreTokenizatinStep):
    split_pattern: str
    invert: bool = False
    behaviour: str = "remove"
    max_splits: int = -1
    skip_tokens: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.max_splits < -1:
            raise ValueError(
                "RegexSplitStep max_splits attribute must be greater then `0` or equal to `-1`, "
                f"got `{self.max_splits}`"
            )

    @classmethod
    def bert_whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(split_pattern=r"\s+", invert=False)

    @classmethod
    def bert_keep_delimeters_splitter(cls) -> "RegexSplitStep":
        """Generates a step with a standard BERT regex.

        The source:
        https://github.com/tensorflow/text/blob/4a098cd852c0b7ebee621e2d211c7f202dd679c2/tensorflow_text/python/ops/bert_tokenizer.py#L39
        """
        return cls(
            "|".join(
                [
                    r"|".join(
                        [
                            r"[!-/]",
                            r"[:-@]",
                            r"[\[-`]",
                            r"[{-~]",
                            r"[\p{P}]",
                        ],
                    ),
                    r"|".join(
                        [
                            r"[\x{4E00}-\x{9FFF}]",
                            r"[\x{3400}-\x{4DBF}]",
                            r"[\x{20000}-\x{2A6DF}]",
                            r"[\x{2A700}-\x{2B73F}]",
                            r"[\x{2B740}-\x{2B81F}]",
                            r"[\x{2B820}-\x{2CEAF}]",
                            r"[\x{F900}-\x{FAFF}]",
                            r"[\x{2F800}-\x{2FA1F}]",
                        ],
                    ),
                ],
            ),
            invert=False,
            behaviour="isolate",
        )

    @classmethod
    def bert_splitter(cls) -> List["RegexSplitStep"]:
        return [cls.bert_whitespace_splitter(), cls.bert_keep_delimeters_splitter()]

    @classmethod
    def whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(r"\w+|[^\w\s]+", invert=True)

    @classmethod
    def metaspace_splitter(cls, metaspace=r"▁") -> "RegexSplitStep":
        return cls(metaspace, invert=False, behaviour="merge_with_next")

    @classmethod
    def byte_level_splitter(cls) -> "RegexSplitStep":
        return cls(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+",
            invert=False,
            behaviour="isolate",
        )

    @classmethod
    def digits_splitter(cls, behaviour="isolate") -> "RegexSplitStep":
        return cls(
            r"\p{Nd}|\p{Nl}|\p{No}",
            invert=False,
            behaviour=behaviour,
        )

    @classmethod
    def punctuation_splitter(cls, behaviour="isolate") -> "RegexSplitStep":
        return cls(
            r"\p{P}",
            invert=False,
            behaviour=behaviour,
        )

    @classmethod
    def special_tokens_splitter(cls, special_tokens: List[str]) -> "RegexSplitStep":
        def quote_meta(unquoted: Union[str, bytes]) -> str:
            if isinstance(unquoted, bytes):
                unquoted = unquoted.decode()
            symbols = []
            for char in unquoted:
                if not char.isalnum() and char != "_":
                    symbols.append("\\")
                symbols.append(char)
            return "".join(symbols)

        return cls(split_pattern="|".join(map(quote_meta, special_tokens)), invert=False, behaviour="isolate")

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        if self.skip_tokens:
            skip_tokens_outputs = self.create_string_constant_node(self.skip_tokens).outputs()
        else:
            skip_tokens_outputs = []

        input_nodes.extend(
            (
                *self.create_string_constant_node(self.split_pattern).outputs(),
                *skip_tokens_outputs,
            )
        )
        return (
            _get_factory()
            .create(
                "RegexSplit",
                input_nodes,
                {
                    "behaviour": self.behaviour.lower(),
                    "invert": self.invert,
                    "max_splits": self.max_splits,
                },
            )
            .outputs()
        )


@dataclass
class WhitespaceSplitStep(PreTokenizatinStep):
    """Works like python `str.split`."""

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return RegexSplitStep.whitespace_splitter().get_ov_subgraph(input_nodes).outputs()


@dataclass
class BytesToCharsStep(PreTokenizatinStep):
    """Maps chars to other chars for Byte-level BPE Tokenizer"""

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return (
            _get_factory()
            .create(
                "BytesToChars",
                input_nodes,
            )
            .outputs()
        )


@dataclass
class TokenizationModelStep(BasePipelineStep):
    pass


@dataclass
class VocabEncoderStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    vocab_values: Optional[List[int]] = None
    default_value: int = -1

    def __post_init__(self) -> None:
        if self.vocab_values is None:
            self.vocab_values = list(range(len(self.vocab)))

    def get_vocab_node_outputs(self) -> Optional[List[Output]]:
        return self.get_pipeline().vocab_node_outputs

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.vocab).outputs(),
                make_constant_node(np.array(self.vocab_values, dtype=np.int32), Type.i32),
                make_constant_node(self.default_value, Type.i32),  # default_value
            )
        )
        return _get_factory().create("VocabEncoder", input_nodes).outputs()


@dataclass
class TrieTokenizerStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    indices: List[int] = field(repr=False)

    def __post_init__(self):
        if len(self.vocab) != len(self.indices):
            raise UserInputError("Vocab and Indices must be the same length.")

        self.vocab, self.indices = self.fill_vocab(self.vocab, self.indices)

    @staticmethod
    def fill_vocab(vocab: List[str], indices: List[int]) -> Tuple[List[str], List[int]]:
        max_idx = max(indices)
        new_indices = list(range(max_idx + 1))

        idx_to_token = dict(zip(indices, vocab))
        new_vocab = []
        for idx in new_indices:
            new_vocab.append(idx_to_token.get(idx, ""))

        return new_vocab, new_indices

    @classmethod
    def from_rwkv_vocab(cls, vocab_file_strings: Iterable[str]) -> TrieTokenizerStep:
        vocab = []
        indices = []
        for line in vocab_file_strings:
            idx = int(line.split(" ")[0])
            x = eval(line.split(" ", 1)[1].rsplit(" ", 1)[0])
            vocab.append(x)
            indices.append(idx)
        return cls(vocab, indices)

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.vocab).outputs(),
                make_constant_node(np.array(self.indices, dtype=np.int32), Type.i32),
            )
        )
        return _get_factory().create("TrieTokenizer", input_nodes).outputs()


@dataclass
class WordPieceTokenizationStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    unk_token: str = "[UNK]"
    suffix_indicator: str = "##"
    max_bytes_per_word: int = 100
    unk_token_id: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.unk_token_id = self.vocab.index(self.unk_token)
        except ValueError:
            raise UserInputError(f"Cannot find unknown token '{self.unk_token}' in the vocab")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "WordPieceTokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"],
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"],
            vocab=[token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])],
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.vocab).outputs(),
                *as_node(self.unk_token_id).outputs(),
            )
        )
        return (
            _get_factory()
            .create(
                "WordpieceTokenizer",
                input_nodes,
                {
                    "suffix_indicator": self.suffix_indicator,
                    "max_bytes_per_word": self.max_bytes_per_word,
                },
            )
            .outputs()
        )


@dataclass
class BPETokenizationStep(TokenizationModelStep):
    vocab: Union[List[str], List[bytes]] = field(repr=False)
    merges: Union[List[str], List[Tuple[bytes, bytes]]] = field(repr=False)
    unk_token: str = ""
    fuse_unk: bool = False
    suffix_indicator: str = ""
    end_suffix: str = ""
    byte_fallback: bool = False
    cache_capacity: int = MIN_CACHE_CAPACITY
    added_tokens: Optional[Union[Dict[str, int], Dict[bytes, int]]] = None

    def finalize(self) -> None:
        if self.added_tokens is None:
            return

        pipeline = self.get_pipeline()

        vocab_set = set(self.vocab)
        for (
            token,
            idx,
        ) in sorted(self.added_tokens.items(), key=lambda x: (x[1], x[0])):
            if token not in vocab_set:
                if pipeline.is_byte_level:
                    token = apply_bytes_to_unicode(token)
                if isinstance(idx, str):
                    assert True
                if idx >= len(self.vocab):
                    self.vocab.append(token)

        added_tokens = sorted(self.added_tokens, reverse=True)

        for split_step in pipeline.split_steps:
            split_step.skip_tokens = added_tokens

        idx = sum(
            1
            for _ in takewhile(
                lambda step: not isinstance(step, (PreTokenizatinStep, TokenizationModelStep)), pipeline.steps
            )
        )
        pipeline.steps.insert(idx, RegexSplitStep.special_tokens_splitter(added_tokens))

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "BPETokenizationStep":
        vocab = [token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])]
        added_tokens = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"] if token["id"]}
        for token_json in tokenizer_json["added_tokens"]:
            if token_json["rstrip"]:
                for new_token in generate_tokens_with_space_symbols(token_json["content"], depth=2):
                    added_tokens[new_token] = token_json["id"]

        # TODO: CVS-150387 Implement suffix_indicator.
        if tokenizer_json["model"]["continuing_subword_prefix"]:
            raise NotImplementedError("continuing_subword_prefix/suffix_indicator is not implemented yet.")

        return cls(
            unk_token=tokenizer_json["model"]["unk_token"] or "",
            fuse_unk=tokenizer_json["model"]["fuse_unk"] or False,
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"] or "",
            end_suffix=tokenizer_json["model"]["end_of_word_suffix"] or "",
            vocab=vocab,
            merges=tokenizer_json["model"]["merges"],
            added_tokens=added_tokens,
            byte_fallback=tokenizer_json["model"]["byte_fallback"],
            cache_capacity=max(
                tokenizer_json["model"].get("cache_capacity", int(len(vocab) * VOCAB_SIZE_CACHE_PROPORTION)),
                MIN_CACHE_CAPACITY,
            ),
        )

    @classmethod
    def from_tiktoken_encoding(
        cls,
        encoding: "Encoding",  # noqa
        reference_vocab: Optional[Dict[Union[str, bytes], int]] = None,
    ) -> "BPETokenizationStep":
        from .tiktoken_parser import generate_vocab_and_merges

        vocab, merges, added_tokens = generate_vocab_and_merges(encoding)
        added_tokens.update(dict(encoding._special_tokens.items()))

        if reference_vocab is not None:
            existing_indices = set(vocab.values())

            for ref_token, ref_idx in reference_vocab.items():
                if ref_idx in existing_indices:
                    continue

                vocab[ref_token] = ref_idx

        return cls(
            unk_token="",
            fuse_unk=False,
            suffix_indicator="",
            end_suffix="",
            vocab=[token for token, idx in sorted(vocab.items(), key=lambda x: x[1])],
            merges=merges,
            added_tokens=added_tokens,
            cache_capacity=max(int(len(vocab) * VOCAB_SIZE_CACHE_PROPORTION), MIN_CACHE_CAPACITY),
        )

    @property
    def merges_is_bytes(self) -> bool:
        return self.merges and not isinstance(self.merges[0], str)

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        pipeline = self.get_pipeline()
        pipeline.vocab_node_outputs = self.create_string_constant_node(self.vocab).outputs()

        if self.added_tokens:
            special_tokens_outputs = self.create_string_constant_node(self.added_tokens).outputs()
        else:
            special_tokens_outputs = []

        if special_tokens_outputs and pipeline.is_byte_level:
            special_tokens_outputs = pipeline.add_ragged_dimension(special_tokens_outputs)
            special_tokens_outputs = BytesToCharsStep().get_ov_subgraph(special_tokens_outputs)[-3:]

        input_nodes.extend(pipeline.vocab_node_outputs)
        if self.merges_is_bytes:
            left_merges, right_merges = zip(*self.merges)
            input_nodes.extend(
                (
                    *self.create_string_constant_node(left_merges).outputs(),
                    *self.create_string_constant_node(right_merges).outputs(),
                )
            )
        else:
            input_nodes.extend(self.create_string_constant_node(self.merges).outputs())

        if special_tokens_outputs:
            input_nodes.extend(
                (
                    *special_tokens_outputs,
                    *make_constant_node(np.array(list(self.added_tokens.values())), Type.i32).outputs(),
                )
            )

        return (
            _get_factory()
            .create(
                "BPETokenizer",
                input_nodes,
                {
                    "unk_token": self.unk_token,
                    "fuse_unk": self.fuse_unk,
                    "suffix_indicator": self.suffix_indicator,
                    "end_suffix": self.end_suffix,
                    "byte_fallback": self.byte_fallback,
                    "cache_capacity": self.cache_capacity,
                },
            )
            .outputs()
        )


@dataclass
class PostTokenizationStep(BasePipelineStep):
    pass


@dataclass
class TruncationStep(PostTokenizationStep):
    max_length: int
    truncate_right: bool = True
    axis: int = -1

    @classmethod
    def from_hf_json(
        cls, tokenizer_json: Dict[str, Any], num_of_added_tokens: int = 0, max_length: int = -1
    ) -> "TruncationStep":
        if max_length == -1:
            max_length = min(
                tokenizer_json["truncation"]["max_length"] - num_of_added_tokens,
                2**31 - 1 - num_of_added_tokens,
            )
        else:
            max_length = min(max_length - num_of_added_tokens, 2**31 - 1 - num_of_added_tokens)
        return cls(
            max_length=max_length,
            truncate_right=tokenizer_json["truncation"]["direction"] == "Right",
        )

    @classmethod
    def from_hf_object(cls, tokenizer: Any, num_of_added_tokens: int = 0) -> "TruncationStep":
        max_length = min(
            tokenizer.model_max_length - num_of_added_tokens,
            2**31 - 1 - num_of_added_tokens,
        )
        return cls(
            max_length=max_length,
            truncate_right=tokenizer.truncation_side == "right",
        )

    @staticmethod
    def validate_inputs(input_nodes):
        if len(input_nodes) != 3:
            raise UserInputError("Only one input ragged tensor is supported as an input for TruncationStep")

    def get_ov_subgraph(self, input_nodes: List[Output]):
        # FIXME: Truncation side (truncate_right) is ignored
        # TODO: Check if axis is the right-most dimension
        self.validate_inputs(input_nodes)

        max_length = opset.minimum(
            opset.subtract(input_nodes[1], input_nodes[0]),
            make_constant_node(self.max_length, Type.i32),
        )
        if self.truncate_right:
            return [
                input_nodes[0],
                opset.add(input_nodes[0], max_length).output(0),
                input_nodes[2],
            ]
        else:
            return [
                opset.subtract(input_nodes[1], max_length).output(0),
                input_nodes[1],
                input_nodes[2],
            ]


@dataclass
class SpecialTokenWithId:
    token: Optional[str] = None
    _token_id: Optional[int] = None

    def set_token_id(self, vocab: Optional[List[str]], is_byte_level: bool = False) -> None:
        token = apply_bytes_to_unicode(self.token) if is_byte_level else self.token
        if self._token_id is None and vocab is not None and token in vocab:
            self._token_id = vocab.index(token)

    @property
    def token_id(self) -> Optional[int]:
        return self._token_id


@dataclass
class TokenWithTypeId:
    token_type_id: Optional[int] = None


@dataclass
class AddToken(TokenWithTypeId, SpecialTokenWithId):
    enabled_by_default: bool = True
    pass


@dataclass
class Sequence(TokenWithTypeId):
    pass


@dataclass
class CombineSegmentsStep(PostTokenizationStep):
    inputs: List[TokenWithTypeId] = field(default_factory=list)
    segment_ids: Optional[List[int]] = None
    axis: int = -1
    add_special_tokens: bool = True

    def __post_init__(self):
        if self.segment_ids is not None:
            return

        segment_ids_tensor = [node.token_type_id for node in self.inputs]
        if any(segment is None for segment in segment_ids_tensor):
            segment_ids_tensor = [0] * len(self.inputs)

        self.segment_ids = segment_ids_tensor

    def finalize(self) -> None:
        pipeline = self.get_pipeline()
        self.set_tokens_ids(vocab=pipeline.vocab, is_byte_level=pipeline.is_byte_level)

    def set_tokens_ids(self, vocab: Optional[List[int]], is_byte_level: bool = False) -> None:
        for input_ in self.inputs:
            if isinstance(input_, AddToken) and input_.token_id is None:
                input_.set_token_id(vocab, is_byte_level)

    @property
    def number_of_added_tokens(self) -> int:
        return sum(1 for input_ in self.inputs if (isinstance(input_, AddToken) and input_.enabled_by_default))

    @classmethod
    def from_hf_json_template_postprocessor(
        cls, post_processor_dict: Dict[str, Any], number_of_inputs: int = 1, add_special_tokens: bool = True
    ) -> "CombineSegmentsStep":
        inputs: List[TokenWithTypeId] = []
        if number_of_inputs == 1:
            post_processor = post_processor_dict["single"]
        else:
            post_processor = post_processor_dict["pair"]

        for template_dict in post_processor:
            if "SpecialToken" in template_dict:
                step = AddToken(
                    token=template_dict["SpecialToken"]["id"],
                    token_type_id=template_dict["SpecialToken"]["type_id"],
                    enabled_by_default=add_special_tokens
                )
                inputs.append(step)
            elif "Sequence" in template_dict:
                inputs.append(Sequence(token_type_id=template_dict["Sequence"]["type_id"]))
        return cls(inputs, add_special_tokens=add_special_tokens)

    @classmethod
    def from_hf_json_bert_postprocessor(
        cls, post_processor_dict: Dict[str, Any], number_of_inputs: int = 1, add_special_tokens: bool = True
    ) -> "CombineSegmentsStep":
        inputs: List[TokenWithTypeId] = []
        inputs.append(
            AddToken(
                token=post_processor_dict["cls"][0],
                token_type_id=0,
                enabled_by_default=add_special_tokens
            )
        )
        inputs.append(Sequence(token_type_id=0))
        inputs.append(
            AddToken(
                token=post_processor_dict["sep"][0],
                token_type_id=0,
                enabled_by_default=add_special_tokens
            )
        )
        if number_of_inputs == 2:
            inputs.append(Sequence(token_type_id=1))
            inputs.append(
                AddToken(
                    token=post_processor_dict["sep"][0],
                    token_type_id=1,
                    enabled_by_default=add_special_tokens
                )
            )
        return cls(inputs, add_special_tokens=add_special_tokens)

    @classmethod
    def from_hf_json_roberta_processor(
        cls, post_processor_dict: Dict[str, Any], number_of_inputs: int = 1, add_special_tokens: bool = True
    ) -> "CombineSegmentsStep":
        if number_of_inputs == 2:
            raise UserInputError("Two inputs not supported for RoBERTa processor")

        inputs: List[TokenWithTypeId] = [Sequence(token_type_id=0)]

        inputs.insert(0, AddToken(token=post_processor_dict["cls"][0], token_type_id=0, enabled_by_default=add_special_tokens))
        inputs.append(AddToken(token=post_processor_dict["sep"][0], token_type_id=0, enabled_by_default=add_special_tokens))
        return cls(inputs, add_special_tokens=add_special_tokens)

    def validate_inputs(self, input_nodes: List[Output]) -> None:
        number_of_sequence_inputs = sum(1 for input_ in self.inputs if isinstance(input_, Sequence))
        if number_of_sequence_inputs != len(input_nodes) / 3:
            raise UserInputError(
                f"Number of input nodes: {len(input_nodes)}, must be equal to {number_of_sequence_inputs}"
            )

    def get_ov_subgraph(self, input_nodes):
        self.validate_inputs(input_nodes)

        op_inputs = []
        input_nodes_iter = iter(input_nodes)

        segment_ids = []
        segment_index = 0
        for (key, token_id), group_iter in groupby(self.inputs, key=lambda input: (type(input), getattr(input, 'token_id', None))):
            if key is Sequence:
                for sequence in group_iter:
                    op_inputs.extend(islice(input_nodes_iter, 3))
                segment_ids.append(self.segment_ids[segment_index])
                segment_index += 1
            elif key is AddToken:
                ids = [node._token_id for node in group_iter]
                
                segment_ids.append(self.segment_ids[segment_index])
                segment_index += len(ids)
                
                op_inputs.extend(make_constant_node(0, Type.i32).outputs())

                # We need to keep end values even if special tokens are not added,
                # because potentially we can turn on adding special tokens in OV GenAI.
                op_inputs.extend(opset.select(make_constant_node(self.add_special_tokens, Type.boolean), 
                                         make_constant_node(len(ids), Type.i32), 
                                         make_constant_node(0, Type.i32)).outputs())
                
                op_inputs.append(make_constant_node(np.array(ids), Type.i32).output(0))
            else:
                raise UserInputError(f"Unexpected node type in CombineSegments: {key}")

        op_inputs.append(make_constant_node(segment_ids, Type.i32).output(0))
        return _get_factory().create("CombineSegments", op_inputs).outputs()


@dataclass
class PaddingStep(PostTokenizationStep, SpecialTokenWithId):
    pad_right: bool = True
    token_type_id: Optional[int] = None
    max_length: int = -1
    axis: int = -1
    pad_to_max_length: bool = False

    @classmethod
    def from_hf_json(
        cls,
        tokenizer_json: Dict[str, Any],
        pad_to_max_length: bool = False,
        max_length: int = -1,
        pad_right: bool = True,
    ) -> "PaddingStep":
        padding_dict = tokenizer_json["padding"]
        padding_strategy = padding_dict.get("strategy", {})
        if max_length == -1 and isinstance(padding_strategy, dict) and "Fixed" in padding_strategy:
            max_length = padding_strategy["Fixed"]

        return cls(
            token=padding_dict["pad_token"],
            _token_id=padding_dict["pad_id"],
            pad_right=pad_right,
            token_type_id=padding_dict["pad_type_id"],
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )

    @staticmethod
    def validate_inputs(input_nodes: List[Output]) -> None:
        # Suppose input_nodes may have multiple tuples each with 3 tensors represented decomposed ragged tensors
        # We suppose that all ragged tensors represent the same structure and produce the mask only once
        if len(input_nodes) % 3 != 0 or len(input_nodes) < 3:
            raise UserInputError(
                f"Number of input nodes should be divisible by 3 and bigger or equal 3. Got {len(input_nodes)}"
            )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        self.validate_inputs(input_nodes)

        outputs = []

        if not self.pad_to_max_length or self.max_length == -1 or self.max_length >= 2**31:
            # Calculate max_length as the maximum ragged length
            max_length = opset.reduce_max(
                opset.subtract(input_nodes[1], input_nodes[0]),
                make_constant_node(0, Type.i32),
            )
        else:
            max_length = make_constant_node(self.max_length, Type.i32)

        names = [TOKEN_IDS_INPUT_NAME, TOKEN_TYPE_IDS_INPUT_NAME][: len(input_nodes) // 3]
        for idx, name in enumerate(names):
            cur_outputs = (
                _get_factory()
                .create(
                    "RaggedToDense",
                    input_nodes[3 * idx : 3 * (idx + 1)]
                    + max_length.outputs()
                    + make_constant_node(self.token_id or 0, Type.i32).outputs(),
                    {
                        "pad_right": self.pad_right,
                        "pad_max_length": self.pad_to_max_length,
                    },
                )
                .outputs()
            )
            cur_outputs[0].tensor.add_names({name})

            outputs.append(cur_outputs[0])
            if idx == 0:
                mask = opset.convert(cur_outputs[1], "i32").output(
                    0
                )  # TODO: Change RaggedToDense to generate mask of any type

        mask.tensor.add_names({ATTENTION_MASK_INPUT_NAME})
        outputs.append(mask)

        return outputs


@dataclass
class DecodingStep(BasePipelineStep):
    pass


@dataclass
class VocabDecoderStep(DecodingStep):
    vocab: Optional[List[str]] = None
    skip_tokens: Optional[List[int]] = None

    def finalize(self) -> None:
        pipeline = self.get_pipeline()
        if pipeline is None and self.skip_tokens is None:
            self.skip_tokens = []
        elif self.skip_tokens is None:
            self.skip_tokens = pipeline.skip_tokens

    def get_vocab_node_outputs(self) -> Optional[List[Output]]:
        return self.get_pipeline().vocab_node_outputs if self.get_pipeline() is not None else None

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        if self.vocab is None:
            vocab_outputs = self.get_vocab_node_outputs()
        else:
            vocab_outputs = self.create_string_constant_node(self.vocab).outputs()
        input_nodes.extend(vocab_outputs)
        return _get_factory().create("VocabDecoder", input_nodes, {"skip_tokens": self.skip_tokens}).outputs()


@dataclass
class CharsToBytesStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return _get_factory().create("CharsToBytes", input_nodes, {}).outputs()


@dataclass
class FuseStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        *input_nodes, chars_node = input_nodes
        return _get_factory().create("FuzeRagged", input_nodes, {}).outputs() + [chars_node]

@dataclass
class UTF8ValidateStep(DecodingStep):
    mode: UTF8ReplaceMode = UTF8ReplaceMode.IGNORE
    
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        replace_mode = True if self.mode is UTF8ReplaceMode.REPLACE else False
        return _get_factory().create("UTF8Validate", input_nodes, {"replace_mode": replace_mode}).outputs()
    
@dataclass
class ByteFallbackStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        if len(input_nodes) == 5:
            ragged_dims, input_nodes = input_nodes[:2], input_nodes[2:]
        else:
            ragged_dims = []

        return ragged_dims + _get_factory().create("ByteFallback", input_nodes).outputs()


@dataclass
class RegexDecodingStep(DecodingStep):
    regex_search_pattern: str
    replace_term: str

    @classmethod
    def clean_up_tokenization_spaces(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r" ([\\.\\?\\!,])| ('[ms])| (') | ('[rv]e)| (n't)",
            replace_term=r"\1",
        )

    @classmethod
    def parse_replace_dict(cls, replace_dict: Dict[str, Any]) -> "RegexDecodingStep":
        pattern = replace_dict.get("pattern", {}).get("String")
        content = replace_dict.get("content")
        if pattern is None or content is None:
            raise ValueError(f"Replace Decoding Op with this parameters: `{replace_dict}` does not support yet.")

        return cls(regex_search_pattern=pattern, replace_term=content)

    @classmethod
    def parse_strip_dict(cls, replace_dict: Dict[str, Any]) -> "RegexDecodingStep":
        content = replace_dict.get("content")
        if content is None:
            raise ValueError(f"Replace Decoding Op with this parameters: `{replace_dict}` does not support yet.")

        return cls(regex_search_pattern=f"^{content}", replace_term="")

    @classmethod
    def strip_forward_space(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r"^ ",
            replace_term="",
        )

    @classmethod
    def strip_forward_space_before_not_space(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r"(^ )([^ ])",
            replace_term=r"\2",
        )

    @classmethod
    def replace_end_of_word_suffix(cls, suffix: str = "</w>") -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=suffix,
            replace_term=" ",
        )

    @classmethod
    def replace_continuing_subword_prefix(cls, prefix: str = "##") -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=prefix,
            replace_term="",
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        if len(input_nodes) == 5:
            ragged_dims, input_nodes = input_nodes[:2], input_nodes[2:]
        else:
            ragged_dims = []

        input_nodes.extend(
            (
                *self.create_string_constant_node(self.regex_search_pattern).outputs(),
                *self.create_string_constant_node(self.replace_term).outputs(),
            )
        )
        return ragged_dims + _get_factory().create("RegexNormalization", input_nodes).outputs()

    @classmethod
    def replace_sp_spaces(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern="▁",
            replace_term=" ",
        )


@dataclass
class TokenizerPipeline:
    steps: List[BasePipelineStep] = field(default_factory=list)
    vocab: Optional[List[str]] = field(default=None, repr=False)
    skip_tokens: Optional[List[int]] = field(default=None, repr=False)
    number_of_inputs: int = 1
    vocab_node_outputs: Optional[List[Output]] = field(default=None, repr=False)
    finalized: bool = False

    @property
    def is_byte_level(self) -> bool:
        return any(isinstance(step, BytesToCharsStep) for step in self.pre_tokenization_steps)

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        return {type(step).__name__: step.get_config() for step in self.steps}

    @singledispatchmethod
    def add_steps(self, steps: Any) -> None:
        raise OVTypeError(f"Type {type(steps)} is not supported")

    @add_steps.register
    def _(self, steps: BasePipelineStep) -> None:
        self.steps.append(steps)
        steps.set_pipeline(self)

    @add_steps.register
    def _(self, steps: list) -> None:
        for step in steps:
            self.steps.append(step)
            step.set_pipeline(self)

    def __getitem__(self, item: int) -> BasePipelineStep:
        return self.steps[item]

    def get_tokenizer_ov_subgraph(self) -> Model:
        self.finalize()

        string_inputs = [op.Parameter(Type.string, PartialShape(["?"])) for _ in range(self.number_of_inputs)]

        processing_outputs = []
        for input_node in string_inputs:
            input_node = _get_factory().create("StringTensorUnpack", input_node.outputs()).outputs()
            for step in self.normalization_steps:
                input_node = step.get_ov_subgraph(input_node)
            input_node = self.add_ragged_dimension(input_node)

            for step in chain(self.pre_tokenization_steps, self.tokenization_steps):
                input_node = step.get_ov_subgraph(input_node)

            processing_outputs.extend(input_node)

        for step in self.post_tokenization_steps:
            processing_outputs = step.get_ov_subgraph(processing_outputs)

        model = Model(processing_outputs, string_inputs, name=TOKENIZER_NAME)
        return model

    def finalize(self) -> None:
        if self.finalized:
            return

        for step in copy(self.steps):
            step.finalize()
        self.finalized = True

    @property
    def normalization_steps(self) -> List[NormalizationStep]:
        return [step for step in self.steps if isinstance(step, NormalizationStep)]

    @property
    def pre_tokenization_steps(self) -> List[PreTokenizatinStep]:
        return [step for step in self.steps if isinstance(step, PreTokenizatinStep)]

    @property
    def split_steps(self) -> List[RegexSplitStep]:
        return [step for step in self.pre_tokenization_steps if isinstance(step, RegexSplitStep)]

    @property
    def tokenization_steps(self) -> List[TokenizationModelStep]:
        return [step for step in self.steps if isinstance(step, TokenizationModelStep)]

    @property
    def post_tokenization_steps(self) -> List[PostTokenizationStep]:
        return [step for step in self.steps if isinstance(step, PostTokenizationStep)]

    @property
    def decoding_steps(self) -> List[DecodingStep]:
        return [step for step in self.steps if isinstance(step, DecodingStep)]

    @staticmethod
    def add_ragged_dimension(input_node: List[Output]) -> List[Output]:
        shape = opset.shape_of(input_node[0])
        batch_size = opset.gather(shape, as_node(0), as_node(0))
        ragged_begins = opset.range(as_node(0), batch_size, as_node(1), output_type="i32").outputs()
        ragged_ends = opset.range(
            as_node(1), opset.add(batch_size, make_constant_node(1, Type.i64)), as_node(1), output_type="i32"
        ).outputs()
        return ragged_begins + ragged_ends + input_node

    def create_decoding_pipeline(self, input_nodes: List[Output]) -> List[Output]:
        for step in self.decoding_steps:
            pipeline_step = step.get_ov_subgraph(input_nodes)
            input_nodes = pipeline_step

        return _get_factory().create("StringTensorPack", input_nodes).outputs()

    def get_detokenizer_ov_subgraph(self) -> Model:
        self.finalize()

        if not any(isinstance(step, VocabDecoderStep) for step in self.decoding_steps):
            raise NotImplementedError("Detokenizer is not supported for this model yet!")

        input_node = op.Parameter(Type.i32, PartialShape(["?", "?"]))
        token_ids = input_node
        outputs = self.create_decoding_pipeline([token_ids])
        model = Model(outputs, [input_node], name=DETOKENIZER_NAME)
        model.output().tensor.add_names({STRING_OUTPUT_NAME})
        return model
