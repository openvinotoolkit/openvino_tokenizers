---
name: tokenizer-diagnostics
description: "Diagnose tokenizer conversion issues for OpenVINO Tokenizers. Use when: tokenizer-checker reports failures, need to pinpoint root cause location (Python conversion vs C++ operation), identify which pipeline stage diverges, determine whether to use tokenizer-fix-python or tokenizer-fix-cpp skill."
argument-hint: "model_id (e.g. zai-org/GLM-4.7)"
---

# OpenVINO Tokenizer Diagnostics

Pinpoints the root cause of tokenizer conversion failures by analyzing the pipeline stage-by-stage. Determines whether the issue is in the Python conversion layer or the C++ operation implementation, and identifies the exact pipeline stage that diverges.

## When to Use

- The `tokenizer-checker` skill reported `status: FAIL`
- Need to understand **where** a tokenizer mismatch originates before fixing it
- Want to see how HF `tokenizer.json` pipeline maps to OV pipeline steps
- Need to identify unsupported normalizer/pre-tokenizer/decoder types

## Inputs

**Required:**
- **model_id**: HuggingFace model identifier or local path (e.g. `meta-llama/Llama-3-8B`)

**Optional (from tokenizer-checker result):**
- `failure_types` — helps focus the diagnosis (e.g. `[token_id_mismatch]`, `[conversion_error]`)
- `failing_categories` — narrows which test strings to inspect
- CLI flags: `--trust-remote-code`, `--use-fast-false`

## Prerequisites

Activate the Python virtual environment before running any commands.

1. **Locate the virtual environment** — check for common directories at the repository root: `.venv/`, `venv/`, `env/`. Use `list_dir` to find it. If none is found, ask the user for its location.
2. **Activate** based on the current platform:
   - **Linux/macOS**: `source <venv_path>/bin/activate`
   - **Windows (cmd)**: `<venv_path>\Scripts\activate.bat`
   - **Windows (PowerShell)**: `<venv_path>\Scripts\Activate.ps1`

## Procedure

### Step 1: Run the diagnose CLI

Run from the repository root:

```
openvino_tokenizers diagnose <model_id> [flags]
```

This executes 5 steps:
- **[1/5] Load HF tokenizer** — downloads and loads via `AutoTokenizer.from_pretrained`
- **[2/5] Map pipeline** — extracts `tokenizer.json` sections (normalizer, pre_tokenizer, model, post_processor, decoder) and maps each HF step to its OV equivalent. Flags unsupported types with `⚠ UNSUPPORTED`.
- **[3/5] Test normalization** — tests each normalizer step individually, then tests the combined pipeline. Reuses `check_normalization` logic.
- **[4/5] Test pre-tokenization** — compares HF `backend_tokenizer.pre_tokenizer` output with OV pre-tokenization behavior.
- **[5/5] Full pipeline comparison** — runs full encode + decode comparison to identify the first point of divergence.

The command prints a **Diagnosis Summary** at the end with structured fields.

### Step 2: Run normalization check (if needed)

If step 3 of `diagnose` shows normalization failures, run the dedicated normalization checker for more detail:

```
openvino_tokenizers check_normalization <model_id> [flags]
```

This gives per-step HF→OV mapping with detailed mismatch output for each normalizer step.

### Step 3: Inspect the pipeline mapping

If step 2 of `diagnose` flags unsupported types or the pipeline mapping reveals gaps, inspect the relevant code:

**For unsupported types** — check whether the type exists in the appropriate map in [hf_parser.py](../../python/openvino_tokenizers/hf_parser.py):
- `TransformersTokenizerPipelineParser.normalizers_map` — normalizer types
- `TransformersTokenizerPipelineParser.pre_tokenization_map` — pre-tokenizer types
- `TransformersTokenizerPipelineParser.post_tokenization_map` — post-processor types
- `TransformersTokenizerPipelineParser.decoding_map` — decoder types
- Tokenization model types are checked in `tokenization_model()` method

**For conversion errors** — read the traceback from `diagnose` output. Common patterns:
- `OVTypeError: ... type '...' is not supported` → missing map entry (Python fix)
- `KeyError` in `parse_*` functions → unexpected `tokenizer.json` structure (Python fix)
- Conversion succeeds but outputs differ → C++ operation bug or incorrect Python step parameters

### Step 4: Determine root cause location

Use the Diagnosis Summary from step 1:

| Summary Field | Interpretation |
|--------------|----------------|
| `root_cause_location: python` | Fix needed in `hf_parser.py` or `tokenizer_pipeline.py` |
| `root_cause_location: cpp` | Fix needed in C++ operation under `src/` |
| `root_cause_location: both` | Fix Python first, then C++ |
| `root_cause_location: none` | No issues found |
| `unsupported_types: [X, Y]` | Types X, Y need new handlers in hf_parser.py |
| `affected_stages: [normalization]` | Issue isolated to normalizer operations |
| `affected_stages: [encode]` | Token ID mismatch — could be pre-tokenizer, tokenizer model, or post-processor |
| `affected_stages: [decode]` | Detokenizer issue — check decoder pipeline |

**Decision rules:**

1. **Unsupported types exist** → `root_cause_location: python`. The type needs a new handler in the parser map and possibly a new pipeline step class.

2. **Normalization fails, full pipeline also fails** → `root_cause_location: cpp`. The Python mapping is correct but the C++ operation produces wrong results.

3. **Normalization passes, full pipeline fails** → `root_cause_location: python`. The issue is in pre-tokenization, tokenization model, post-processing, or decoding pipeline construction.

4. **Only normalization fails** → `root_cause_location: cpp`. Individual normalizer step works differently in C++ than in HF.

### Step 5: Produce diagnosis report

After all analysis, produce a structured report:

```
## Diagnosis
- root_cause_location: python | cpp | both | none
- affected_stages: [<list of affected stages>]
- unsupported_types: [<list of unsupported HF types>]
- normalization_failures: <count>
- pre_tokenization_failures: <count>
- full_pipeline_failures: <count>
- description: <human-readable summary of the root cause>
- suggested_fix_skill: tokenizer-fix-python | tokenizer-fix-cpp | none
- details: |
    <copy the relevant diagnostic output, including pipeline mapping,
     failing test strings, and mismatch details>
```

### Step 6: Generate minimal reproducer (when applicable)

If the issue is well-isolated, create a minimal Python script that demonstrates the mismatch. Use this template:

```python
#!/usr/bin/env python3
"""Minimal reproducer for <model_id> tokenizer mismatch in <stage>."""
import numpy as np
from transformers import AutoTokenizer
from openvino import Core
from openvino_tokenizers import convert_tokenizer

# Load
hf_tok = AutoTokenizer.from_pretrained("<model_id>")
ov_tok_model, ov_detok_model = convert_tokenizer(hf_tok, with_detokenizer=True)
ov_tok = Core().compile_model(ov_tok_model)

# Test
test_string = "<failing_input>"
hf_out = hf_tok([test_string], return_tensors="np", truncation=True)
ov_out = ov_tok([test_string])

print(f"HF ids:  {hf_out['input_ids'].tolist()}")
print(f"OV ids:  {ov_out['input_ids'].tolist()}")
print(f"Match:   {np.array_equal(hf_out['input_ids'], ov_out['input_ids'])}")
```

Save the reproducer to inform the fixer skill or for human review.

## Key Code References

- **CLI diagnose tool**: `python/openvino_tokenizers/cli_tools/diagnose_tokenizer.py`
- **CLI normalization check**: `python/openvino_tokenizers/cli_tools/check_normalization.py`
- **HF parser & maps**: `python/openvino_tokenizers/hf_parser.py` → `TransformersTokenizerPipelineParser`
- **Pipeline step classes**: `python/openvino_tokenizers/tokenizer_pipeline.py`
- **C++ operations**: `src/*.cpp` / `src/*.hpp`

## Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** modify `model_id` — pass it exactly as provided by the user.
