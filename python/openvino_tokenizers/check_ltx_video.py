#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Simple reproducer for Lightricks/LTX-Video tokenizer checks.

This script uses the same test strings as the check utility and runs
HF + OpenVINO tokenization in sequence. If a segfault exists in OV tokenization,
it should reproduce while iterating over these strings.
"""

import argparse
from typing import Any

from openvino import Core
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer


# Test strings copied from cli_tools/check_tokenizer.py
ENG_STRINGS = [
	# "Eng... test, string?!",
	# "Multiline\nstring!\nWow!",
	# "A lot\t w!",
	# "A lot\t\tof whitespaces!",
	# "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
	# "Eng, but with d1gits: 123; 0987654321, stop.0987654321 - eng, but with d1gits: 123",
	# "USER: <image>\nWhat is in the image? ASSISTANT:",
	# "What is OpenVINO?",
	# (
	# 	"If I have 100 million dollars, what kinds of projects should I invest to maximize "
	# 	"my benefits in background of a growing number of artificial intelligence technologies?"
	# ),
	(
		"Write an epic travel diary where an engineer, a poet, and a chef cross seven cities in seven nights, "
		"and in each city they must solve one unusual challenge: rebuild a clocktower using only recycled brass, "
		"compose a lullaby for a sleepless market, design a dinner menu for astronauts who miss home, map hidden "
		"canals beneath an old library, negotiate peace between rival street orchestras, restore a broken weather "
		"vane that predicts memories instead of storms, and finally present a public workshop explaining every "
		"decision, every tradeoff, every failed attempt, and every lesson learned, while also listing materials, "
		"budgets, timelines, contingency plans, and a final reflection on teamwork, creativity, responsibility, "
		"and how small practical choices can change the future of an entire neighborhood."
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

# ALL_TEST_STRINGS = ENG_STRINGS + MULTILINGUAL_STRINGS + EMOJI_STRINGS + MISC_STRINGS
ALL_TEST_STRINGS = ENG_STRINGS


def _extract_input_ids(ov_out: Any) -> Any:
	"""Return input_ids from OV output dict-like object."""
	if "input_ids" in ov_out:
		return ov_out["input_ids"]
	first_key = next(iter(ov_out.keys()))
	return ov_out[first_key]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Simple LTX tokenizer reproducer.")
	parser.add_argument("--repo-id", default="Lightricks/LTX-Video")
	parser.add_argument("--subfolder", default="tokenizer")
	parser.add_argument("--max-strings", type=int, default=0, help="0 means all strings.")
	parser.add_argument("--no-special-tokens", action="store_true", default=False)
	return parser


def main() -> None:
	args = build_parser().parse_args()

	print(f"Loading HF tokenizer: {args.repo_id} (subfolder={args.subfolder})")
	hf_tokenizer = AutoTokenizer.from_pretrained(
		args.repo_id,
		subfolder=args.subfolder,
		trust_remote_code=True,
	)

	print("Converting to OpenVINO tokenizer + detokenizer")
	ov_tokenizer_model, _ = convert_tokenizer(hf_tokenizer, with_detokenizer=True, max_length=10024)
	ov_tokenizer = Core().compile_model(ov_tokenizer_model)

	test_strings = ALL_TEST_STRINGS if args.max_strings == 0 else ALL_TEST_STRINGS[: args.max_strings]
	print(f"Running {len(test_strings)} test strings")

	for i, text in enumerate(test_strings, start=1):
		preview = repr(text)
		if len(preview) > 120:
			preview = preview[:117] + "..."
		
		print(f"[{i:02d}/{len(test_strings):02d}] {preview}")

		hf_out = hf_tokenizer(
			text,
			add_special_tokens=not args.no_special_tokens,
			return_tensors="np",
		)
		ov_out = ov_tokenizer([text])
		ov_ids = _extract_input_ids(ov_out)

		hf_len = int(hf_out["input_ids"].shape[-1])
		ov_len = int(ov_ids.shape[-1])
		print(f"  HF len={hf_len}, OV len={ov_len}")

	print("Done")


if __name__ == "__main__":
	main()
