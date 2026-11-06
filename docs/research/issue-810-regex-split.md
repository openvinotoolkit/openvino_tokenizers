# RegexSplit adjacent-delimiter semantics (issue #810)

Research date: 2026-11-04. Upstream comparisons below use Hugging Face Tokenizers commit `301277072263fb3830eb66eec7e47adadf76f1dc`, the revision linked by [issue #810](https://github.com/openvinotoolkit/openvino_tokenizers/issues/810).

## Problem and authoritative behavior

Issue #810 reports that repeated delimiter matches disappear in `mergedwithnext` and overlap earlier text in `mergedwithprevious`. Its reproduction includes both offline toy BPE tokenizers and `jhu-clsp/mmBERT-small`. The issue's proposed fix is explicitly untested; its description is useful evidence of the reported failure, while HF source owns the compatibility contract. [Issue report](https://github.com/openvinotoolkit/openvino_tokenizers/issues/810)

HF merges a delimiter with the immediately neighboring non-delimiter span in the selected direction. It does not merge an entire delimiter run into neighboring text. For pattern `-` and input `the-final--countdown`:

| Behavior | Pieces |
| --- | --- |
| MergedWithPrevious | `the-`, `final-`, `-`, `countdown` |
| MergedWithNext | `the`, `-final`, `-`, `-countdown` |

HF implements the two directions by folding the match-span sequence forward or backward, tracking whether the preceding processed span was a match. [Contract](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/tokenizer/normalizer.rs#L74), [implementation](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/tokenizer/normalizer.rs#L727)

`invert=True` flips each span's match flag without combining spans. Consequently, adjacent regex matches become separate non-delimiter spans. With pattern `a`, input `aab`, and inversion enabled, previous mode yields `a`, `ab`; next mode yields `a`, `a`, `b`. Merely resetting the old held-back offset after a merge does not address a held non-delimiter being overwritten by another non-delimiter. These examples follow directly from the flag inversion and merge rules. [Pattern partition and inversion](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/tokenizer/pattern.rs#L63), [Split's inversion dispatch](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/pre_tokenizers/split.rs#L96)

Metaspace replaces spaces, optionally prepends its marker, and uses `MergedWithNext` when splitting is enabled. Its upstream repeated-space test expects `▁Hey`, `▁`, `▁`, `▁friend!` for `Hey   friend!`. This supplies a direct primary-source reason to test the conversion pipeline as well as the native operation. [Metaspace implementation](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/pre_tokenizers/metaspace.rs#L122), [multiple_spaces test](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/pre_tokenizers/metaspace.rs#L234)

## Implementation design

The current change in [src/regex_split.cpp](../../src/regex_split.cpp) handles unlimited merge modes with one pending span containing byte offsets and its delimiter flag. For each incoming span:

1. If no span is pending, retain the incoming span.
2. Merge and emit when pending/current flags are non-delimiter/delimiter for previous mode, or delimiter/non-delimiter for next mode. Clear pending state after emission.
3. Otherwise emit the pending span and retain the incoming span.
4. Emit any pending span once after scanning finishes.

This is a forward-streaming implementation of the directional merge contract. It preserves consecutive delimiters and consecutive non-delimiters, including inverted input. State belongs to each ragged input piece; spans use byte offsets, so a multi-byte marker such as `▁` is handled without character-index conversion. The emitter initializes the skip flag for each emitted span, including emissions caused by flushing pending state. Skipped pieces remain outside this merge path. [Current implementation](../../src/regex_split.cpp)

A standalone Python design check compared this pending-span algorithm with a transcription of HF's directional fold for all Boolean span-class sequences of lengths 1 through 8 in both modes: 1,020 comparisons agreed. This checks the abstract merge rule only; it is not a compiled OpenVINO test or a substitute for the implementation's regression tests.

## Scope boundaries and existing limitations

The new path is restricted to `max_splits == -1`. HF Split exposes pattern, behavior, and inversion, with no corresponding split-limit field. Locally, the older finite-limit path extends an output to the string end when its counter reaches the limit but continues scanning and emitting. The adjacency change preserves that existing path rather than defining new finite-limit semantics. [HF Split fields](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/pre_tokenizers/split.rs#L27), [local implementation](../../src/regex_split.cpp)

Zero-width matching is another existing difference. HF includes empty regex match events in its span sequence and removes empty resulting pieces at the pre-tokenizer layer. The local `get_next_match` helper treats an empty match as termination. The current change retains that behavior; comprehensive zero-width compatibility would require separate iterator-progress handling. [HF regex-span construction](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/tokenizer/pattern.rs#L63), [empty-piece filtering](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/tokenizer/pre_tokenizer.rs#L86), [local matcher](../../src/regex_split.cpp)

Regression coverage should distinguish individual adjacent matches from a grouped match: `-` over `--` produces two delimiter events, while `-+` produces one. Useful cases include leading, interior, trailing, and delimiter-only runs; both merge directions and inversion values; multiple runs; UTF-8 markers; multi-character delimiters; no match; existing empty-input behavior; independent ragged pieces; and skipped special tokens. Offline toy-tokenizer conversion and the issue's pinned published tokenizer complement direct split-piece assertions. These are coverage recommendations, not claims that any particular test has run.

## Local verification, 2026-11-04

Both builds used fresh `git archive HEAD` snapshots of commit `47eb7817`; only the fixed snapshot received the modified `src/regex_split.cpp`. Both were built with GCC, Ninja, Release, and identical dependencies: Python 3.12.8, OpenVINO `2026.5.0-23715-b63f0947bd9`, Hugging Face tokenizers 0.23.3 and transformers 5.22.1. Build directories are `/tmp/ov810-baseline-build` and `/tmp/ov810-fixed-build`; PCRE2 and SentencePiece source copies came from `/tmp/ov809-build/_deps`.

For each variant, the extension was selected with `OV_TOKENIZER_PREBUILD_EXTENSION_PATH=/tmp/ov810-${variant}-build/src/libopenvino_tokenizers.so`, `PYTHONPATH=$PWD/python:$PWD`, and `/tmp/ov809-venv/bin/python -m pytest`:

| Selection | Baseline | Fixed |
| --- | --- | --- |
| `tests/regex_split_adjacent_test.py -q --tb=short` | 19 failed, 5 passed | 24 passed |
| `tests/layer_tests.py tests/check_tokenizer_test.py -q -k 'not charsmap and not unigram' --tb=short` | 4 failed, 139 passed, 5 deselected | 4 failed, 139 passed, 5 deselected |
| `tests/tokenizers_test.py -q -k '(roberta-base or bloom or deberta-base or galactica or ModernBERT) and (test_hf_bpe_tokenizers_outputs or test_bpe_detokenizer)' --tb=short` | 2 failed, 1168 passed, 16391 deselected | 2 failed, 1168 passed, 16391 deselected |

The four failures in both layer runs are existing Unicode normalization threshold mismatches: NFC 17395 vs 17325 expected, NFD 17806 vs 17736, NFKC 17225 vs 17224, and NFKD 17620 vs 17619. The two failures in both model runs are BLOOM's whitespace-heavy case, with and without added tokens. These identical results do not establish a new regression. The pytest session hook can alter exit status and tracked statistics, so reports were inspected and `tests/stats.json` and `tests/pass_rates.json` restored byte-for-byte.

The regression suite checks both merge directions and inversion on all 126 nonempty `a`/`-` strings of length 1 through 6, Unicode and multicharacter patterns, empty strings, skip markers, UTF-8 byte offsets and batches, repeat inference, saved-model reload, and three offline toy tokenizer conversions.

The published `jhu-clsp/mmBERT-small` tokenizer at revision `abc32620dd4f6ab06f5fbe905dc25f310618e09f` matched HF on all four strings from the issue with the fixed build; all four differed on baseline. Reproduction script and full logs are under `/home/linux/.cache/openvino-tokenizers-810/`. Ruff check, Ruff format check, and `git diff --check` passed.

Scope limits: finite `max_splits` uses the original path and retains the existing adjacent-delimiter problem; zero-width matching is unchanged. Full model zoo, performance, non-Linux platforms and the complete project suite were not tested.

### Follow-up review

Review identified an additional compatibility case: when a batch contains an empty piece and nonempty text, the original `mergedwithnext` path emits the empty piece. The initial pending-span implementation omitted it. The fix now keeps empty pieces on the original path and adds a mixed-batch regression for both merge modes. After rebuilding, the combined new and existing RegexSplit selection reported **59 passed, 114 deselected** with the fixed library, versus **19 failed, 40 passed, 114 deselected** on the original baseline. The results in the table above describe the earlier run before this follow-up.
