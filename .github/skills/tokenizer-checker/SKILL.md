---
name: tokenizer-checker
description: "Validate a HuggingFace tokenizer with OpenVINO Tokenizers and OpenVINO GenAI. Use when: checking if a tokenizer converts and works correctly, verifying tokenizer/detokenizer accuracy, checking GenAI Tokenizer compatibility, gating tokenizer enablement pipeline."
argument-hint: "model_id (e.g. zai-org/GLM-4.7)"
---

# OpenVINO Tokenizer Checker

Pass/fail gate for HuggingFace tokenizer compatibility with OpenVINO Tokenizers. Runs the `openvino_tokenizers check` CLI and produces a structured result that downstream skills (diagnostics, fixers) and the orchestrator agent can consume.

## When to Use

- Verify a HuggingFace tokenizer converts to OpenVINO and matches HF outputs
- Check if a newly supported tokenizer works end-to-end with OpenVINO GenAI
- Gate the tokenizer enablement pipeline — first step before diagnostics or fixing
- Re-verify after fixes have been applied

## Inputs

The user must provide:

- **model_id**: HuggingFace model identifier or local path (e.g. `zai-org/GLM-4.7`)

Optional flags the user may request (pass through to the CLI):

- `--trust-remote-code` — required for some models with custom tokenizer code
- `--no-detokenizer` — skip detokenizer conversion and testing
- `--use-sentencepiece-backend` — use SentencePiece backend during conversion
- `--no-special-tokens` — encode without special tokens
- `--no-skip-special-tokens` — decode keeping special tokens
- `--skip-missing-outputs` — ignore HF outputs absent in OV result (e.g. token_type_ids)
- `--use-fast-false` — load the legacy (slow) tokenizer

## Prerequisites

Activate the Python virtual environment before running any commands.

1. **Locate the virtual environment** — check for common directories at the repository root: `.venv/`, `venv/`, `env/`. Use `list_dir` to find it. If none is found, ask the user for its location.
2. **Activate** based on the current platform:
   - **Linux/macOS**: `source <venv_path>/bin/activate`
   - **Windows (cmd)**: `<venv_path>\Scripts\activate.bat`
   - **Windows (PowerShell)**: `<venv_path>\Scripts\Activate.ps1`

## Procedure

### Step 1: Run the tokenizer check

Run from the repository root:

```
openvino_tokenizers check <model_id> [flags]
```

This executes:
- **[1/5] Load HF tokenizer** — downloads and loads the tokenizer via `AutoTokenizer.from_pretrained`
- **[2/5] Convert to OpenVINO** — converts to OV tokenizer + detokenizer models
- **[3/5] Test against 31 strings** — compares HF vs OV encode/decode on English, multilingual, emoji, and edge-case strings
- **[4/5] GenAI Tokenizer encode + decode** — tests `openvino_genai.Tokenizer` encode/decode with and without special tokens (skipped if `openvino_genai` is not installed)
- **[5/5] GenAI padding + pair inputs** — checks batch padding and pair-input behaviour. For tokenizers-backend tokenizers (`PreTrainedTokenizerFast` / `TokenizersBackend`), mismatches are reported as errors and affect the exit code. For other tokenizers, mismatches are reported as warnings only (skipped if `openvino_genai` is not installed)

### Step 2: Parse the Output

Read both stdout and stderr from the command. Extract:

1. **Exit code**: 0 = all hard steps passed, 1 = any hard step failed.
2. **Per-step results**: each step prints `✓` (pass) or `✗` (fail) with context.
3. **Step 5 warnings**: `⚠` lines that do NOT affect the exit code for non-tokenizers-backend tokenizers.

Classify each failing test string into one of these categories based on the [test string definitions in check_tokenizer.py](../../python/openvino_tokenizers/cli_tools/check_tokenizer.py):

| Category | Strings |
|----------|---------|
| english | `"Eng... test, string?!"`, `"Multiline\nstring!..."`, `"What is OpenVINO?"`, etc. |
| multilingual | `"Тестовая строка!"`, `"測試字符串"`, `"سلسلة الاختبار"`, etc. |
| emoji | `"😀"`, `"🤣🤣🤣😁😁😁😁"`, `"🫠"`, `"🤷‍♂️"`, `"🤦🏼‍♂️"` |
| whitespace_edge | `""`, `" "`, `" " * 10`, `"\n"`, `"\x06"`, etc. |

Classify the nature of each mismatch:

| Mismatch Type | How to Identify |
|--------------|-----------------|
| `conversion_error` | Step [2/5] prints `✗` — conversion itself failed with an exception |
| `token_id_mismatch` | `"value mismatch for 'input_ids'"` in step [3/5] or [4/5] |
| `shape_mismatch` | `"shape mismatch for ..."` in step [3/5] or [4/5] |
| `missing_output` | `"output key '...' missing from OV result"` in step [3/5] |
| `decode_mismatch` | `"detokenizer mismatch"` in step [3/5] or `"decode mismatch"` in step [4/5] |
| `padding_mismatch` | `"padding mismatch"` in step [5/5] |
| `pair_encode_mismatch` | `"pair encode mismatch"` in step [5/5] |

### Step 3: Produce Structured Result

After parsing, produce a result block in exactly this format:

```
## Result
- status: PASS | FAIL
- conversion_ok: true | false
- tokenizer_test_failures: <number of failing strings in step 3>
- detokenizer_failures: <number of detokenizer mismatches in step 3>
- genai_test_failures: <number of failing strings in step 4, or "skipped">
- genai_advanced_issues: <number of issues in step 5, or "skipped">
- genai_advanced_strict: true | false (whether step 5 issues are errors or warnings)
- failing_categories: [<list of categories with failures>]
- failure_types: [<list of mismatch types observed>]
- error_details: |
    <copy the relevant ✗ lines and mismatch details from stderr, indented>
```

**Examples:**

All pass:
```
## Result
- status: PASS
- conversion_ok: true
- tokenizer_test_failures: 0
- detokenizer_failures: 0
- genai_test_failures: 0
- genai_advanced_issues: 0
- genai_advanced_strict: true
- failing_categories: []
- failure_types: []
- error_details: none
```

Partial failure:
```
## Result
- status: FAIL
- conversion_ok: true
- tokenizer_test_failures: 3
- detokenizer_failures: 0
- genai_test_failures: 3
- genai_advanced_issues: 1
- genai_advanced_strict: true
- failing_categories: [emoji, whitespace_edge]
- failure_types: [token_id_mismatch]
- error_details: |
    Input: '😀'
      value mismatch for 'input_ids':
        HF: [1, 155, 234]
        OV: [1, 155, 235]
    Input: ''
      value mismatch for 'input_ids':
        HF: [1]
        OV: [1, 0]
```

Conversion failure:
```
## Result
- status: FAIL
- conversion_ok: false
- tokenizer_test_failures: 0
- detokenizer_failures: 0
- genai_test_failures: skipped
- genai_advanced_issues: skipped
- genai_advanced_strict: false
- failing_categories: []
- failure_types: [conversion_error]
- error_details: |
    OVTypeError: Normalizer type 'NewFancyNormalizer' is not supported
```

### Step 4: Report to User

**If `status: PASS`:**
- State that the tokenizer is fully compatible.
- Note whether GenAI steps were tested or skipped (if `openvino_genai` is not installed).
- Note any step-5 warnings if present (when `genai_advanced_strict: false` and `genai_advanced_issues > 0`).

**If `status: FAIL`:**
- Present the structured result block.
- Summarize the key findings:
  1. **Which step failed** — conversion, tokenizer comparison, detokenizer, GenAI encode/decode, or GenAI advanced.
  2. **Pattern in failing categories** — helps identify the root cause area:
     - Only emoji → multi-byte / surrogate handling issue
     - Only multilingual → Unicode/encoding issue
     - Only whitespace_edge → edge-case handling issue
     - All categories → fundamental conversion or model issue
  3. **Nature of the mismatch** — token ID, shape, decode, etc.
- If invoked as part of the enablement pipeline, the orchestrator will hand off the result to the `tokenizer-diagnostics` skill for root cause analysis.

### Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** modify `model_id` — pass it exactly as provided by the user.
- **NEVER** call internal Python functions directly — always use the `openvino_tokenizers` CLI commands.
