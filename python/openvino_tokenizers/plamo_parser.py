# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
from openvino import Model, Output, Type
from openvino.exceptions import OVTypeError
from openvino.utils.types import make_constant_node

from . import _get_factory
from .constants import ATTENTION_MASK_INPUT_NAME, TOKEN_IDS_INPUT_NAME, UTF8ReplaceMode
from .tokenizer_pipeline import (
    AddToken,
    CombineSegmentsStep,
    FuseStep,
    PaddingStep,
    RegexDecodingStep,
    RegexSplitStep,
    Sequence,
    SpecialTokensSplit,
    TokenizationModelStep,
    TokenizerPipeline,
    TruncationStep,
    UTF8ValidateStep,
    VocabDecoderStep,
)
from .utils import TokenzierConversionParams, create_string_constant_node


@dataclass
class PlamoModelStep(TokenizationModelStep):
    pieces: list[str] = field(repr=False)
    scores: list[int] = field(repr=False)
    byte_ids: list[int] = field(repr=False)

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        return (
            _get_factory()
            .create(
                "PlamoTokenizer",
                input_nodes
                + create_string_constant_node(self.pieces)
                + make_constant_node(np.asarray(self.scores, dtype=np.int32), Type.i32).outputs()
                + make_constant_node(np.asarray(self.byte_ids, dtype=np.int32), Type.i32).outputs(),
            )
            .outputs()
        )


def convert_plamo_tokenizer(tokenizer, params: TokenzierConversionParams) -> Model | tuple[Model, Model]:
    # The official tokenizer keeps the JSONL rows in ID order.
    pieces, scores, decoder_vocab = [], [], []
    byte_ids = [-1] * 256
    for token_id, (piece, score, kind, *_) in enumerate(tokenizer.data):
        pieces.append("" if kind == "BYTE" else piece)
        scores.append(round(score * 10_000))
        if kind == "BYTE":
            byte = int(piece[3:5], 16)
            byte_ids[byte] = token_id
            decoder_vocab.append(bytes([byte]))
        else:
            decoder_vocab.append(piece.encode("utf-8"))
    if -1 in byte_ids:
        raise OVTypeError("PLaMo 3 requires a complete UTF-8 byte vocabulary.")

    for token_id, token in tokenizer.added_tokens_decoder.items():
        while token_id >= len(pieces):
            pieces.append("")
            scores.append(0)
            decoder_vocab.append(b"")
        pieces[token_id] = token.content
        decoder_vocab[token_id] = token.content.encode("utf-8")

    pipeline = TokenizerPipeline()
    pipeline.add_steps(SpecialTokensSplit.from_hf_tokenizer(tokenizer))
    # PFN's tag split runs before the repeat split. Python's Unicode whitespace
    # class also includes U+001C through U+001F.
    pipeline.add_steps(RegexSplitStep(r"<\|plamo:[^|\s\x{1C}-\x{1F}]{0,64}\|>", behaviour="isolate", mergeable=False))
    runs = []
    if tokenizer.break_around_repeated_chars_threshold is not None:
        runs.append(rf"([^\n])\1{{{tokenizer.break_around_repeated_chars_threshold - 1},}}")
    if tokenizer.break_around_consecutive_spaces_threshold is not None:
        runs.append(rf" {{{tokenizer.break_around_consecutive_spaces_threshold},}}")
    if runs:
        pipeline.add_steps(RegexSplitStep("|".join(runs), behaviour="isolate", mergeable=False))
    pipeline.add_steps(RegexSplitStep("\uee00", behaviour="remove", mergeable=False))
    pipeline.add_steps(PlamoModelStep(pieces, scores, byte_ids))

    segments = [Sequence(token_type_id=0)]
    if tokenizer.add_bos_token:
        segments.insert(0, AddToken(_token_id=tokenizer.bos_token_id, token_type_id=0))
    if tokenizer.add_eos_token:
        segments.append(AddToken(_token_id=tokenizer.eos_token_id, token_type_id=0))
    combine = CombineSegmentsStep(segments, add_special_tokens=params.add_special_tokens)
    special_token_slots = combine.number_of_added_tokens
    truncation = TruncationStep.from_hf_object(
        tokenizer,
        num_of_added_tokens=special_token_slots if params.add_special_tokens else 0,
        truncation=params.truncation,
    )
    # GenAI adds the serialized special-token slots back to the i32 limit,
    # including slots whose insertion is disabled by default.
    truncation.max_length = min(truncation.max_length, np.iinfo(np.int32).max - special_token_slots)
    pipeline.add_steps(
        [
            truncation,
            combine,
            PaddingStep(
                _token_id=tokenizer.pad_token_id,
                pad_right=tokenizer.padding_side == "right",
                max_length=tokenizer.model_max_length,
                pad_to_max_length=params.use_max_padding,
            ),
        ]
    )
    model = pipeline.get_tokenizer_ov_subgraph()
    encoder = Model(
        [model.output(TOKEN_IDS_INPUT_NAME), model.output(ATTENTION_MASK_INPUT_NAME)],
        model.get_parameters(),
        model.get_friendly_name(),
    )
    encoder.add_sinks(model.get_sinks())
    if not params.with_detokenizer:
        return encoder

    pipeline.add_steps(
        [
            VocabDecoderStep(decoder_vocab, tokenizer.all_special_ids, params.skip_special_tokens),
            FuseStep(),
        ]
    )
    if params.utf8_replace_mode not in (None, UTF8ReplaceMode.DISABLE):
        pipeline.add_steps(UTF8ValidateStep(params.utf8_replace_mode))
    cleanup = params.clean_up_tokenization_spaces
    if cleanup is None:
        cleanup = tokenizer.clean_up_tokenization_spaces
    if cleanup:
        pipeline.add_steps(RegexDecodingStep.clean_up_tokenization_spaces())
    return encoder, pipeline.get_detokenizer_ov_subgraph()
