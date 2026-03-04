#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Quick sanity-check CLI for a single HuggingFace tokenizer.

Usage:
    check_tokenizer <hf_repo_id> [options]

Steps performed:
  [1] Load the HF tokenizer
  [2] Convert it to OpenVINO (tokenizer + detokenizer)
  [3] Run the full test-string suite and compare outputs

On success each step prints a single ✓ line.
On failure the step prints ✗ plus the relevant context (exception, failing
strings with expected/actual output diff).
"""

import argparse
import sys
import textwrap
import traceback
from typing import Optional

import numpy as np

# ── test strings ─────────────────────────────────────────────────────────────

ENG_STRINGS = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop.0987654321 - eng, but with d1gits: 123",
    "USER: <image>\nWhat is in the image? ASSISTANT:",
    "What is OpenVINO?",
    (
        "If I have 100 million dollars, what kinds of projects should I invest to maximize "
        "my benefits in background of a growing number of artificial intelligence technologies?"
    ),
]
MULTILINGUAL_STRINGS = [
    "Тестовая строка!",
    "Testzeichenfolge?",
    "Tester, la chaîne...",
    "測試字符串",
    "سلسلة الاختبار",
    "מחרוזת בדיקה",
    "Сынақ жолы á",
    "رشته تست",
    "介绍下清华大学",
]
EMOJI_STRINGS = [
    "😀",
    "😁😁",
    "🤣🤣🤣😁😁😁😁",
    "🫠",
    "🤷‍♂️",
    "🤦🏼‍♂️",
]
MISC_STRINGS = [
    "",
    b"\x06".decode(),
    " ",
    " " * 10,
    " " * 256,
    "\n",
    " \t\n",
]

ALL_TEST_STRINGS = ENG_STRINGS + MULTILINGUAL_STRINGS + EMOJI_STRINGS + MISC_STRINGS

# ── helpers ───────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}", file=sys.stderr)


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{BOLD}[{n}/{total}]{RESET} {msg}")


def _truncate(s: str, max_len: int = 80) -> str:
    s = repr(s)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _array_summary(arr) -> str:
    flat = arr.reshape(-1).tolist()
    if len(flat) <= 12:
        return str(flat)
    return str(flat[:6]) + " ... " + str(flat[-3:])


def _compare_outputs(hf_out: dict, ov_out, skip_missing: bool = False) -> list[str]:
    """
    Return a list of mismatch descriptions, empty list means all match.
    ov_out can be an OVDict (keyed by tensor name) or a plain dict.
    """
    issues = []
    for key, hf_val in hf_out.items():
        try:
            ov_val = ov_out[key]  # OVDict supports lookup by tensor name
        except (KeyError, RuntimeError):
            if skip_missing:
                continue
            issues.append(f"output key '{key}' missing from OV result")
            continue
        if ov_val.shape != hf_val.shape:
            issues.append(
                f"shape mismatch for '{key}': HF {hf_val.shape} vs OV {ov_val.shape}\n"
                f"    HF: {_array_summary(hf_val)}\n"
                f"    OV: {_array_summary(ov_val)}"
            )
        elif not np.all(ov_val == hf_val):
            issues.append(
                f"value mismatch for '{key}':\n"
                f"    HF: {_array_summary(hf_val)}\n"
                f"    OV: {_array_summary(ov_val)}"
            )
    return issues


# ── main steps ────────────────────────────────────────────────────────────────

def step_load_tokenizer(repo_id: str, use_fast: bool, trust_remote_code: bool):
    """Step 1 – load the HF tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    cls_name = type(tokenizer).__name__
    _ok(f"Loaded  {BOLD}{repo_id}{RESET}  →  {cls_name}")
    return tokenizer


def step_convert(hf_tokenizer, with_detokenizer: bool, use_sentencepiece_backend: bool):
    """Step 2 – convert to OpenVINO."""
    from openvino import Core
    from openvino_tokenizers import convert_tokenizer

    result = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=with_detokenizer,
        use_sentencepiece_backend=use_sentencepiece_backend,
    )

    core = Core()
    if with_detokenizer:
        ov_tok_model, ov_detok_model = result
        ov_tok = core.compile_model(ov_tok_model)
        ov_detok = core.compile_model(ov_detok_model)
        _ok("Converted tokenizer + detokenizer")
        return ov_tok, ov_detok
    else:
        ov_tok = core.compile_model(result)
        _ok("Converted tokenizer")
        return ov_tok, None


def step_test_tokenizer(
    hf_tokenizer,
    ov_tokenizer,
    ov_detokenizer: Optional[object],
    add_special_tokens: bool,
    skip_special_tokens: bool,
    skip_missing_outputs: bool,
) -> int:
    """
    Step 3 – run the test suite.

    Returns the number of failing strings (0 = all good).
    """
    failures: list[tuple[str, list[str]]] = []

    for test_str in ALL_TEST_STRINGS:
        input_list = [test_str]

        # ── tokenizer comparison ─────────────────────────────────────────────
        hf_out = hf_tokenizer(
            input_list,
            return_tensors="np",
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        ov_out = ov_tokenizer(input_list)

        issues = _compare_outputs(dict(hf_out), ov_out, skip_missing=skip_missing_outputs)

        # ── detokenizer comparison ───────────────────────────────────────────
        if ov_detokenizer is not None:
            token_ids = hf_out["input_ids"]
            hf_decoded = hf_tokenizer.batch_decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )
            ov_decoded = ov_detokenizer(token_ids.astype("int32"))["string_output"].tolist()
            if hf_decoded != ov_decoded:
                issues.append(
                    f"detokenizer mismatch:\n"
                    f"    HF: {hf_decoded}\n"
                    f"    OV: {ov_decoded}"
                )

        if issues:
            failures.append((test_str, issues))

    total = len(ALL_TEST_STRINGS)
    passed = total - len(failures)

    if not failures:
        _ok(f"All {total} strings matched")
    else:
        _fail(f"{passed}/{total} strings matched — {len(failures)} failed")
        for test_str, issues in failures:
            print(f"\n  {YELLOW}Input:{RESET} {_truncate(test_str)}", file=sys.stderr)
            for issue in issues:
                indented = textwrap.indent(issue, "    ")
                print(f"{RED}{indented}{RESET}", file=sys.stderr)

    return len(failures)


# ── CLI entry point ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_tokenizer",
        description=(
            "Quick sanity-check for a HuggingFace tokenizer:\n"
            "  [1] Load HF tokenizer\n"
            "  [2] Convert to OpenVINO\n"
            "  [3] Compare outputs on the standard test suite\n\n"
            "Exit code: 0 = all steps succeeded, 1 = any step failed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo_id", help="HuggingFace model id or local path to a tokenizer directory.")
    parser.add_argument(
        "--use-fast-false",
        "--use_fast_false",
        action="store_true",
        default=False,
        help="Pass use_fast=False to AutoTokenizer.from_pretrained (legacy tokenizer).",
    )
    parser.add_argument(
        "--trust-remote-code",
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to AutoTokenizer.from_pretrained.",
    )
    parser.add_argument(
        "--no-detokenizer",
        "--no_detokenizer",
        action="store_true",
        default=False,
        help="Skip detokenizer conversion and testing.",
    )
    parser.add_argument(
        "--use-sentencepiece-backend",
        "--use_sentencepiece_backend",
        action="store_true",
        default=False,
        help="Use the SentencePiece backend during conversion.",
    )
    parser.add_argument(
        "--no-special-tokens",
        "--no_special_tokens",
        action="store_true",
        default=False,
        help="Encode test strings without special tokens.",
    )
    parser.add_argument(
        "--no-skip-special-tokens",
        "--no_skip_special_tokens",
        dest="skip_special_tokens",
        action="store_false",
        default=True,
        help="Decode test tokens with skip_special_tokens=False (default is True, matching the OV detokenizer default).",
    )
    parser.add_argument(
        "--skip-missing-outputs",
        "--skip_missing_outputs",
        action="store_true",
        default=False,
        help="Ignore OV outputs that are absent in the HF result (e.g. token_type_ids).",
    )
    return parser


def check_tokenizer() -> None:
    parser = build_parser()
    args = parser.parse_args()

    total_steps = 3
    exit_code = 0

    # ── Step 1 ────────────────────────────────────────────────────────────────
    _step(1, total_steps, f"Loading HF tokenizer  '{args.repo_id}'")
    try:
        hf_tokenizer = step_load_tokenizer(
            repo_id=args.repo_id,
            use_fast=not args.use_fast_false,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        _fail("Failed to load tokenizer:")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    _step(2, total_steps, "Converting to OpenVINO")
    try:
        ov_tokenizer, ov_detokenizer = step_convert(
            hf_tokenizer=hf_tokenizer,
            with_detokenizer=not args.no_detokenizer,
            use_sentencepiece_backend=args.use_sentencepiece_backend,
        )
    except Exception:
        _fail("Conversion failed:")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    n_strings = len(ALL_TEST_STRINGS)
    _step(3, total_steps, f"Testing against {n_strings} strings")
    try:
        n_failures = step_test_tokenizer(
            hf_tokenizer=hf_tokenizer,
            ov_tokenizer=ov_tokenizer,
            ov_detokenizer=ov_detokenizer,
            add_special_tokens=not args.no_special_tokens,
            skip_special_tokens=args.skip_special_tokens,  # True by default
            skip_missing_outputs=args.skip_missing_outputs,
        )
        if n_failures:
            exit_code = 1
    except Exception:
        _fail("Testing raised an unexpected exception:")
        traceback.print_exc(file=sys.stderr)
        exit_code = 1

    print()
    sys.exit(exit_code)
