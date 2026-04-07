---
name: tokenizer-fix-python
description: "Fix Python conversion issues in OpenVINO Tokenizers. Use when: tokenizer-diagnostics reports root_cause_location=python, unsupported types need new handlers in hf_parser.py, pipeline step mapping or merging is incorrect, tokenizer_pipeline.py step classes need fixes."
argument-hint: "model_id and diagnosis details (e.g. unsupported_types: [NewType], affected_stages: [pre_tokenizer])"
---

# OpenVINO Tokenizer Python Fixer

Fixes issues in the Python conversion layer that maps HuggingFace tokenizer pipelines to OpenVINO Tokenizers.

## When to Use

- `tokenizer-diagnostics` reported `root_cause_location: python`
- Common issue categories:
  - **Unsupported type**: A HuggingFace step type is missing from the parser maps
  - **Incorrect mapping**: A supported type produces wrong OV steps (wrong params, missing steps)
  - **Merge bug**: `merge_regex_split_steps` incorrectly combines incompatible `RegexSplitStep` instances during `finalize()`
  - **Missing pipeline step class**: A new OV operation needs a Python step wrapper

## Inputs

**Required:**
- **model_id**: HuggingFace model identifier
- **diagnosis**: Output from the `tokenizer-diagnostics` skill, particularly:
  - `unsupported_types` — which HF types need handlers
  - `affected_stages` — which pipeline stages diverge
  - `description` — human-readable root cause summary

**Helpful context (from tokenizer-diagnostics output):**
- The HF→OV pipeline mapping (Step 2 output)
- Finalized pipeline showing step merges
- Pre-tokenization expected vs actual token splits

## Prerequisites

Activate the Python virtual environment before running any commands.

1. **Locate the virtual environment** — check for common directories at the repository root: `.venv/`, `venv/`, `env/`. Use `list_dir` to find it. If none is found, ask the user for its location.
2. **Activate** based on the current platform:
   - **Linux/macOS**: `source <venv_path>/bin/activate`
   - **Windows (cmd)**: `<venv_path>\Scripts\activate.bat`
   - **Windows (PowerShell)**: `<venv_path>\Scripts\Activate.ps1`

## Architecture Overview

### Conversion Pipeline

```
HuggingFace tokenizer
    │
    ▼
TransformersTokenizerPipelineParser (hf_parser.py)
    │  reads tokenizer.json
    │  maps each section via type→handler dicts
    ▼
TokenizerPipeline (tokenizer_pipeline.py)
    │  list of *Step dataclasses
    │  finalize() → merges steps, transforms vocab
    ▼
OV Model (via get_tokenizer_ov_subgraph)
    │  each Step.get_ov_subgraph() creates C++ ops
    ▼
Compiled OpenVINO model
```

### Key Files

| File | Purpose |
|------|---------|
| `python/openvino_tokenizers/hf_parser.py` | Parser with type→handler maps for all pipeline sections |
| `python/openvino_tokenizers/tokenizer_pipeline.py` | Pipeline step dataclasses, `TokenizerPipeline`, `finalize()` |
| `python/openvino_tokenizers/convert_tokenizer.py` | Entry point: dispatches to fast/sentencepiece/tiktoken converters |
| `src/ov_extension.cpp` | C++ operation registration (lists all available ops) |

### Parser Maps (hf_parser.py)

The parser uses four type→handler dictionaries. Each handler takes a `step_dict` (from `tokenizer.json`) and returns one or more `*Step` instances:

| Map | Pipeline Section | Approx Line |
|-----|-----------------|-------------|
| `normalizers_map` | `tokenizer.json["normalizer"]` | ~L194 |
| `pre_tokenization_map` | `tokenizer.json["pre_tokenizer"]` | ~L229 |
| `post_tokenization_map` | `tokenizer.json["post_processor"]` | ~L279 |
| `decoding_map` | `tokenizer.json["decoder"]` | ~L367 |

The tokenization model (BPE/WordPiece/Unigram/WordLevel) is handled by the `tokenization_model()` method (~L265), not via a map.

### Step Class Hierarchy (tokenizer_pipeline.py)

```
BasePipelineStep
├── NormalizationStep  (normalizer ops)
│   ├── RegexNormalizationStep
│   ├── NormalizeUnicode
│   ├── CaseFoldStep
│   └── CharsmapStep
├── PreTokenizatinStep  (splitting ops)
│   ├── RegexSplitStep
│   ├── WhitespaceSplitStep
│   └── BytesToCharsStep
├── TokenizationModelStep  (vocabulary lookup)
│   ├── BPETokenizationStep
│   ├── WordPieceTokenizationStep
│   ├── UnigramModelStep
│   └── VocabEncoderStep
├── PostTokenizationStep  (combine/truncate/pad)
│   ├── CombineSegmentsStep
│   ├── TruncationStep
│   └── PaddingStep
└── DecodingStep  (detokenizer ops)
    ├── VocabDecoderStep
    ├── CharsToBytesStep
    ├── FuseStep
    └── ByteFallbackStep
```

### Available C++ Operations (src/ov_extension.cpp)

RegexNormalization, RegexSplit, BPETokenizer, WordpieceTokenizer, BytesToChars,
CharsToBytes, CombineSegments, VocabEncoder, VocabDecoder, TrieTokenizer,
Truncate, FuzeRagged, ByteFallback, SpecialTokensSplit, CharsMapNormalization,
CaseFold, NormalizeUnicode, UnigramTokenizer, UTF8Validate,
SentencepieceTokenizer, SentencepieceDetokenizer, SentencepieceStreamDetokenizer,
RaggedToDense, RaggedToSparse, RaggedToRagged, StringToHashBucket,
StringTensorPack, StringTensorUnpack, RaggedTensorPack, EqualStr

## Procedure

### Step 1: Extract the tokenizer.json structure

Download and inspect the HF tokenizer's pipeline definition:

```python
from tokenizers import Tokenizer
import json

tok = Tokenizer.from_pretrained("<model_id>")
tj = json.loads(tok.to_str())

# Inspect the failing section
print(json.dumps(tj["pre_tokenizer"], indent=2))  # or normalizer, decoder, etc.
```

Alternatively, use `openvino_tokenizers diagnose <model_id>` output which already maps all sections.

### Step 2: Identify the fix category

Based on the diagnosis, determine which category the fix falls into:

#### Category A: Missing type in parser map

**Symptom**: `unsupported_types: [SomeType]` in diagnosis output.

**Fix pattern**:

1. Read the HF tokenizers source or `tokenizer.json` spec to understand what `SomeType` does
2. Find or create the corresponding OV step class in `tokenizer_pipeline.py`
3. Add the handler to the appropriate map in `hf_parser.py`:

```python
# In hf_parser.py — add to the appropriate map:
normalizers_map["NewType"] = lambda step_dict: SomeStep(
    param1=step_dict.get("param1", "default"),
    param2=step_dict["param2"],
)
```

If a new step class is needed:

```python
# In tokenizer_pipeline.py — add a new step:
@dataclass
class NewNormalizationStep(NormalizationStep):
    param1: str = ""

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        # Use an existing C++ op
        input_nodes.extend(create_string_constant_node(self.param1))
        return (
            _get_factory()
            .create("RegexNormalization", input_nodes, {"global_replace": True})
            .outputs()
        )
```

The `get_ov_subgraph` pattern:
1. Extend `input_nodes` with constant parameters (strings via `create_string_constant_node`, scalars via `make_constant_node`)
2. Call `_get_factory().create("<OpName>", input_nodes, {attributes_dict})` to create the OV C++ operation
3. Return `.outputs()` — the list of output nodes

#### Category B: Incorrect parameter parsing

**Symptom**: Conversion succeeds but outputs differ. Diagnosis shows no unsupported types but pre-tokenization or full pipeline tests fail.

**Fix pattern**:

1. Read the HF `tokenizer.json` for the failing step and compare with the parser handler
2. Common issues:
   - Reading wrong key from `step_dict` (e.g. `"String"` vs `"Regex"` in pattern dicts)
   - Missing boolean flags that affect behavior
   - Default values that don't match HF defaults

Example — how handler reads a pattern dict:
```python
# HF tokenizer.json may have:  {"pattern": {"String": "abc"}}  or  {"pattern": {"Regex": "a.*c"}}
pattern = step_dict["pattern"].get("String") or step_dict["pattern"]["Regex"]
```

#### Category C: Pipeline merge bug (merge_regex_split_steps)

**Symptom**: Diagnosis shows `⚠ Pre-tokenization merge: N steps → M steps` and pre-tokenization test fails. The merged regex pattern in the finalized pipeline produces different splits than running the original patterns sequentially.

**Where**: `TokenizerPipeline.merge_regex_split_steps()` in `tokenizer_pipeline.py` (~L1498).

**How it works**: The method iterates over `RegexSplitStep` instances and tries to combine them via `RegexSplitStep.__add__`, which joins patterns with `|` (OR). It merges when `invert`, `behaviour`, and `max_splits` all match.

**Common fix approaches**:

1. **Prevent merge for incompatible patterns**: Add a check in `RegexSplitStep.__add__` that raises `ValueError` when patterns can't be safely merged (the merge loop catches `ValueError` and keeps steps separate):

```python
def __add__(self, other: "RegexSplitStep") -> "RegexSplitStep":
    # ... existing checks ...
    # Example: prevent merging patterns with conflicting quantifiers
    if _patterns_conflict(self.split_pattern, other.split_pattern):
        raise ValueError("Patterns cannot be safely merged")
    return self.__class__(
        split_pattern="|".join((self.split_pattern, other.split_pattern)),
        ...
    )
```

2. **Skip merge entirely for specific step configurations**: Add a `mergeable` flag or check specific pattern signatures that are known to conflict.

3. **Fix the merge logic**: If the `|`-join is semantically wrong for certain pattern combinations, the patterns may need wrapping in non-capturing groups `(?:pattern1)|(?:pattern2)` or the merge should be skipped.

#### Category D: Step finalization issue

**Symptom**: Individual step mapping looks correct but the finalized pipeline differs unexpectedly.

**Where**: `TokenizerPipeline.finalize()` in `tokenizer_pipeline.py` (~L1547) and individual `Step.finalize()` methods.

**Key finalization behaviors to be aware of**:
- `BPETokenizationStep.finalize()` (~L665): Removes `BytesToCharsStep` and `CharsToBytesStep` from the pipeline when `is_byte_level=True`, absorbing byte-level encoding into the BPE vocab
- `merge_regex_split_steps()` (~L1498): Merges compatible `RegexSplitStep` instances
- `del_duplicated_split_steps()`: Removes duplicate whitespace splitters
- `update_metaspace_step_with_special_tokens()`: Patches metaspace regex when special tokens exist

### Step 3: Implement the fix

1. Read the relevant source files to understand the current implementation
2. Apply the minimal fix — do not refactor surrounding code
3. Ensure the fix handles edge cases present in the test strings:
   - Whitespace variants: `\t`, `\n`, multiple spaces
   - Empty strings
   - Unicode: emoji, CJK, accented characters, RTL
   - Mixed content: digits + letters + punctuation

### Step 4: Verify

After applying the fix, rebuild and test:

```bash
# Rebuild (triggers CMake for C++ extensions)
pip install --pre -Ue . --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

# Run targeted check
openvino_tokenizers check <model_id> [flags]

# Run diagnose to verify pre-tokenization now matches
openvino_tokenizers diagnose <model_id> [flags]

# Run unit tests for the affected operation
python -m pytest tests/layer_tests.py -k <relevant_test> -v

# Run integration tests for the model type
python -m pytest tests/tokenizers_test.py -k <relevant_test> -v

# Run the full test suite to check for regressions
python -m pytest tests/ -v
```

### Step 5: Check for regressions

The fix must not break other tokenizers. Key model sets to verify:

- **WordPiece models**: `bert-base-multilingual-cased`, `google/mobilebert-uncased`
- **BPE models**: `openai-community/gpt2`, `roberta-base`, `NousResearch/Meta-Llama-3-8B-Instruct`
- **SentencePiece models**: `NousResearch/Llama-2-13b-hf`, `microsoft/Phi-3-mini-128k-instruct`
- **TikToken models**: `Qwen/Qwen-14B-Chat`

At minimum, run `openvino_tokenizers check` against 2-3 models of the same type as the fixed tokenizer.

## Common Pitfalls

1. **Don't forget `parse()`**: `TransformersTokenizerPipelineParser.__init__` does NOT populate the pipeline — you must call `parser.parse()` first.

2. **Handler return types**: Map handlers must return either a single `*Step` instance or a `list[*Step]`. The parser wraps singles in lists internally.

3. **Regex dialect**: OV uses PCRE2 regex (via the C++ `RegexSplit` / `RegexNormalization` ops). HF's Rust tokenizers uses the `fancy-regex` crate. Key differences: Unicode property classes (`\p{L}`, `\p{N}`) are supported in both, but lookahead/lookbehind support may differ.

4. **Byte-level encoding**: `ByteLevel` pre-tokenizer in HF does TWO things: regex split AND byte→char mapping. In OV these are separate steps: `RegexSplitStep` + `BytesToCharsStep`. The `BytesToCharsStep` is later removed during `BPETokenizationStep.finalize()` which absorbs it into the vocab.

5. **Step ordering matters**: Pre-tokenization steps are applied sequentially. `merge_regex_split_steps` combines them into fewer ops for efficiency, but this changes the execution from sequential to parallel `|`-alternation.

## Security

- **NEVER** run arbitrary code from `tokenizer.json` — only read data fields
- **NEVER** install packages — assume the environment is pre-configured
- **NEVER** modify `model_id` — pass it exactly as provided
- Validate that regex patterns from `tokenizer.json` are used only via the safe OV regex API (no `eval()`, no `subprocess`)
