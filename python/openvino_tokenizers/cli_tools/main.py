# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Unified CLI entry point for OpenVINO Tokenizers.

Usage:
    openvino_tokenizers convert <hf_repo_id> [options]   – convert a HF tokenizer to OpenVINO
    openvino_tokenizers check   <hf_repo_id> [options]   – sanity-check a HF tokenizer
"""

import argparse


def main() -> None:
    from .check_tokenizer import _configure_parser as _cfg_check
    from .check_tokenizer import run as _run_check
    from .convert_tokenizer import _configure_parser as _cfg_convert
    from .convert_tokenizer import run as _run_convert

    parser = argparse.ArgumentParser(
        prog="openvino_tokenizers",
        description="OpenVINO Tokenizers CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    sub_convert = subparsers.add_parser(
        "convert",
        help="Convert a HuggingFace tokenizer to OpenVINO format.",
        description="Converts tokenizers from Huggingface Hub to OpenVINO Tokenizer model.",
    )
    _cfg_convert(sub_convert)
    sub_convert.set_defaults(func=_run_convert)

    sub_check = subparsers.add_parser(
        "check",
        help="Sanity-check a HuggingFace tokenizer through OpenVINO.",
        description=(
            "Quick sanity-check for a HuggingFace tokenizer:\n"
            "  [1] Load HF tokenizer\n"
            "  [2] Convert to OpenVINO\n"
            "  [3] Compare outputs on the standard test suite\n\n"
            "Exit code: 0 = all steps succeeded, 1 = any step failed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _cfg_check(sub_check)
    sub_check.set_defaults(func=_run_check)

    args = parser.parse_args()
    args.func(args)
