---
name: tokenizer-checker
description: "Validate a HuggingFace tokenizer with OpenVINO Tokenizers and OpenVINO GenAI. Use when: checking if a tokenizer converts and works correctly, verifying tokenizer/detokenizer accuracy, testing normalization steps, checking GenAI Tokenizer compatibility."
argument-hint: "model_id (e.g. zai-org/GLM-4.7)"
---

# OpenVINO Tokenizer Checker

Validates that a HuggingFace tokenizer converts to OpenVINO correctly and produces matching outputs for encoding, decoding, normalization, and GenAI compatibility.

## When to Use

- Verify a HuggingFace tokenizer converts to OpenVINO and matches HF outputs
- Check if a newly supported tokenizer works end-to-end with OpenVINO GenAI
- Diagnose which test categories (English, multilingual, emoji, whitespace) fail
- Test normalization steps individually to isolate mismatches

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
- **[5/5] GenAI padding + pair inputs** — soft-checks batch padding and pair-input behaviour; warnings only, does not affect exit code (skipped if `openvino_genai` is not installed)

### Step 2: Run the normalization check

```
openvino_tokenizers check_normalization <model_id> [flags]
```

This executes:
- **[1/3] Load HF tokenizer** — same as above
- **[2/3] Parse normalizer pipeline** — extracts individual normalizer steps from `tokenizer.json` and prints the HF → OV mapping
- **[3/3] Test normalizer steps** — tests each normalizer step independently, then tests the full stacked pipeline

### Step 3: Interpret Results

Both commands print `✓` / `✗` per step and exit with code 0 (all passed) or 1 (any failure).

**Pass criteria:**
- Exit code 0 for each command
- All test strings matched in step 3 of `check`
- All normalizer steps matched in step 3 of `check_normalization`

**Failure output includes:**
- The input string that failed
- Expected (HF) vs actual (OV) values — token IDs, decoded text, or normalized text
- Shape mismatches, value mismatches, or missing output keys

**Warnings (step 5 of `check`):**
- Batch padding mismatches across different configurations (longest, max_length, left/right padding)
- Pair-input encode mismatches
- These do NOT affect the exit code but should be reported

### Step 4: Report Results

Provide a structured report to the user:

**If all steps pass:**
- State that the tokenizer is fully compatible
- Note whether GenAI steps were tested or skipped (if `openvino_genai` is not installed)
- Note any step-5 warnings if present

**If any step fails, build a failure report covering:**

1. **Which step failed** — conversion, tokenizer comparison, detokenizer, GenAI encode/decode, or normalization
2. **Which string categories failed** — identify patterns:
   - English strings only → basic tokenization issue
   - Multilingual strings → Unicode/encoding issue
   - Emoji strings → multi-byte / surrogate handling issue
   - Empty/whitespace strings → edge-case handling issue
   - All strings → fundamental conversion issue
3. **Nature of the mismatch** — token ID mismatch, shape mismatch, missing output key, decode mismatch, or normalization mismatch
4. **Normalization isolation** — if `check_normalization` identifies a specific normalizer step as the root cause, report which step type (e.g. `NFC`, `Lowercase`, `Precompiled`) and its parameters

### Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** modify `model_id` — pass it exactly as provided by the user.
- **NEVER** call internal Python functions directly — always use the `openvino_tokenizers` CLI commands.