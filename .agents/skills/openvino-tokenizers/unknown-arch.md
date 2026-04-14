# Skill: Handle Tokenizer Failure Caused by Unknown Model Architecture

> **Scope:** This skill is **tokenizer conversion only**. It handles
> `AutoTokenizer.from_pretrained` / `convert_tokenizer` failures caused by an
> outdated Transformers version. It does **not** resolve the optimum-intel model
> support gap (model configs, patchers, custom OpenVINO ops). That is the job
> of the **Optimum-Intel agent** (skills: `optimum_debug_export` →
> `optimum_add_model_support`).
>
> **When this skill is called:**
> - Via `codingagent_tokenizers_tier1.yml` as a secondary step **after** the
>   Optimum-Intel agent has already resolved the transformers version and
>   model support issues. The tokenizers agent is responsible only for the
>   tokenizer conversion half of enablement.
> - In legacy/edge cases where the deploy pipeline routes `tokenizer_error`
>   (pure tokenizer failures unrelated to architecture support).

**Trigger (this skill is applicable when):**
- Tokenizer conversion fails with one of these signatures:
  - `KeyError: '<model_type>'` inside `convert_tokenizer` or tokenizer-specific code
  - `ValueError: The checkpoint you are trying to load has model type '<model_type>'
    but Transformers does not recognize this architecture.` — during tokenizer load
  - `OSError: <model_id> does not appear to have a file named tokenizer*.json`
    combined with the above
- OR: `AutoTokenizer.from_pretrained` throws before any tokenizer-specific code runs.

**Not applicable (hand off to Optimum-Intel agent):**
- Export failures from `optimum-cli export openvino` — even if caused by old transformers
- `model_type` not in `TasksManager` of optimum-intel (`optimum_unsupported_arch`)
- All errors in `optimum/exporters/openvino/` traceback paths

> This skill is one component of the **Autonomous Experiment Loop** defined in
> `openvino_tokenizers.agent.md`. Always log each attempt to `experiments_log.json`
> and follow the Surrender Protocol from that agent when you run out of hypotheses.

## Context

When this skill is triggered during tokenizer conversion, the model architecture
may require a Transformers version newer than the one currently installed.
This is NOT a tokenizer-specific bug — the standard fix is upgrading Transformers.

**Important:** For `unknown_arch_transformers_too_old` routed from `classify_error.py`,
the Optimum-Intel agent runs first and resolves the full model support (transformers
upgrade + export pipeline). This tokenizers skill is subsequently responsible for
confirming that the tokenizer conversion also works under the resolved environment.

Known model families that have needed newer Transformers (updated as encountered):

| `model_type` | Requires (approx.) | Notes |
|---|---|---|
| `qwen3_5` | `transformers>=4.57.0.dev0` | Hybrid DeltaNet + GQA, multimodal |
| _(add new entries here as discovered)_ | | |

## Steps

### 0. Consume Cross-Agent Artifacts (run FIRST when called after Optimum-Intel agent)

When invoked after the Optimum-Intel agent, a `transformers_override` may
already be recorded in the shared manifest artifact. Apply it **before** any
other step — do not re-discover what the previous agent already solved.

```bash
# Bootstrap from the shared manifest (picks up transformers_override, patches, etc.)
if [ -f meat_manifest.json ]; then
  python scripts/collect_artifacts.py bootstrap --manifest meat_manifest.json | bash
fi

# Also check the direct tokenizer artifact if present
if [ -f tokenizers-artifact/artifact-description.md ]; then
  cat tokenizers-artifact/artifact-description.md
fi

# Verify what transformers version is now active
python -c "import transformers; print(transformers.__version__)"
```

If the manifest already contains a working `transformers_install` URL, use it:
```bash
# From manifest: transformers_install = "git+https://github.com/huggingface/transformers@main"
# or a specific version like "transformers==4.57.0"
pip install --quiet "$transformers_install"
```

Proceed directly to **Step 3a (tokenizer conversion)** if the transformers
version is already resolved by the manifest. Skip Steps 1–3.

Log this as `attempt_id: exp-000-consume-artifacts` in `experiments_log.json`.

### 1. Upstream Pattern Search (run when this skill is primary responder)

Search for existing fixes in public repos before writing new code:

```bash
MODEL_TYPE="<model_type>"

# openvino_tokenizers PRs/issues
curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/search/issues?q=${MODEL_TYPE}+repo:openvinotoolkit/openvino_tokenizers&per_page=5" \
  | python -c "import json,sys;[print(i['number'],i['state'],i['title'],i['html_url']) for i in json.load(sys.stdin).get('items',[])]"

# transformers changelog / PRs for this model_type
curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/search/issues?q=${MODEL_TYPE}+repo:huggingface/transformers+label:New+model&per_page=5" \
  | python -c "import json,sys;[print(i['number'],i['state'],i['title'],i['html_url']) for i in json.load(sys.stdin).get('items',[])]"

# Local git history in the ref clone
cd "${TOKENIZERS_DIR:-/tmp/openvino-tokenizers-ref}"
git log --oneline --all --grep="$MODEL_TYPE" | head -20
```

If a relevant merged PR exists in `openvino_tokenizers` → fetch and apply it
to the experimental source clone before attempting anything else:  it may be
a direct fix.

Log this as `attempt_id: exp-000-upstream-search` in `experiments_log.json`.

### 1. Identify the failing `model_type`

Extract from the error traceback:
```python
import re, pathlib

log_text = pathlib.Path("error.log").read_text()
m = re.search(r"model type `(\w+)`", log_text) or \
    re.search(r"KeyError: '(\w+)'", log_text)
model_type = m.group(1) if m else None
print(f"Unknown model_type: {model_type}")
```

If `model_type` is found → continue.
If not → this is a different tokenizer error; escalate to the orchestrator
with `error_class=tokenizer_error`.

### 2. Check Latest Stable PyPI Release First

Before reaching for git-HEAD, verify whether the latest *released* transformers
version already contains the fix — this is a lighter-weight upgrade.

```bash
# Get latest released version from PyPI
LATEST=$(pip index versions transformers 2>/dev/null \
  | grep -oP '(?<=Available versions: )[\.\d]+' | head -1)
INSTALLED=$(pip show transformers | grep '^Version' | awk '{print $2}')
echo "Installed: $INSTALLED  |  Latest PyPI: $LATEST"
```

If `$INSTALLED != $LATEST`, upgrade and re-probe:

```bash
pip install --quiet "transformers==$LATEST"
python - <<'EOF'
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained("$MODEL_ID", trust_remote_code=False)
    print(f"PyPI latest OK: {cfg.model_type}")
except Exception as e:
    print(f"Still failing after PyPI upgrade: {e}")
EOF
```

If the probe passes → retry tokenizer conversion (Step 3a), record
`transformers_install: "transformers==$LATEST"` in `transformers_dep.json`.

### 3. If Latest PyPI Still Fails — Try git-HEAD

```bash
# Install transformers from git (do NOT upgrade other packages)
pip install --quiet \
  "git+https://github.com/huggingface/transformers@main" \
  --no-deps           # avoid accidentally dragging incompatible versions
```

Then probe:
```python
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained("$MODEL_ID", trust_remote_code=False)
    print(f"Config loaded OK: {cfg.model_type}")
    supported = True
except Exception as e:
    print(f"Still failing: {e}")
    supported = False
```

> **Important:** If `--no-deps` causes an `ImportError` in transformers itself,
> retry without `--no-deps`:
> ```bash
> pip install --quiet "git+https://github.com/huggingface/transformers@main"
> ```

### 4a. If git Transformers works → retry tokenizer conversion

```bash
optimum-cli export openvino \
  --model "$MODEL_ID" \
  --task text-generation-with-past \
  --weight-format int4 \
  "$OUTPUT_DIR"
```

Check whether `openvino_tokenizer.xml` / `openvino_detokenizer.xml` were created.

Validate round-trip:
```python
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer
from openvino import Core

tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)

ie = Core()
compiled_tok = ie.compile_model(ov_tokenizer)
compiled_detok = ie.compile_model(ov_detokenizer)

test_texts = ["Hello, world!", "Привет, мир!", "你好世界"]
for text in test_texts:
    ids = tokenizer(text, return_tensors="np").input_ids
    ov_ids = compiled_tok({"string_input": [text]})["input_ids"]
    assert (ids == ov_ids).all(), f"Token ID mismatch for: {text}"

print("Round-trip OK")
```

If round-trip passes → proceed to Step 4.
If round-trip fails → record as `tokenizer_error` and escalate with details.

### 4b. If git Transformers still fails — Patch the Source

`--trust-remote-code` is **never** an acceptable mitigation — treat any
requirement for it as an immediate Surrender trigger.

If git-HEAD transformers still does not recognise the `model_type`, the feature
has **not been merged upstream yet** — but this does not mean we must surrender.

Invoke the **`optimum_patch_transformers`** skill
(`skills/optimum-intel/patch-transformers.md`) which will:

1. Search transformers GitHub for an open/merged PR for this `model_type`.
2. If found → cherry-pick the PR onto a local transformers clone.
3. If not found → extract the config class from the HF model repo and register
   it in the local transformers clone.
4. Install from the patched local clone and test.
5. On success → generate a `git format-patch` via `scripts/generate_git_patch.py`.
6. Post the patch to the GitHub issue as a comment and attach it to the
   cross-agent artifact.
7. Signal `transformers_source_patched=true` to the orchestrator.

Only trigger the **Surrender Protocol** from `openvino_tokenizers.agent.md`
if the `optimum_patch_transformers` skill itself exhausts all strategies
(no config file in model repo, PR cherry-pick conflicts unresolvable, etc.).

### 5. Create a dependency patch

If Step 4a succeeded (git transformers worked + round-trip passed), record
the dependency and produce the cross-agent artifact:

Write `agent-results/openvino-tokenizers/patches/transformers_dep.json`:
```json
{
  "model_type": "<model_type>",
  "transformers_install": "git+https://github.com/huggingface/transformers@main",
  "transformers_version_used": "<x.y.z.devN from: pip freeze | grep transformers>",
  "reason": "not_in_stable_release"
}
```

If changes to `openvino_tokenizers` source were also needed (beyond the
transformers upgrade), generate a `git format-patch` from the experimental
source clone:
```bash
cd "${TOKENIZERS_SRC:-/tmp/openvino-tokenizers-src}"
git add -A
git commit -m "fix(<model_type>): add tokenizer conversion support"
git format-patch HEAD~1 -o agent-results/openvino-tokenizers/patches/ \
  --stdout > agent-results/openvino-tokenizers/patches/<attempt_id>-<model_type>-tokenizer.patch
```

Then log this attempt and assemble the cross-agent artifact per the
protocol in `openvino_tokenizers.agent.md` (Cross-Agent Artifact Protocol).

### 6. Signal back to Optimum-Intel agent

**Critical:** If `agent-results/openvino-tokenizers/patches/transformers_dep.json` was created, the
orchestrator MUST re-run the **Optimum-Intel agent** (or at minimum the
deployer) with `transformers_override` set to the git URL.

Write to the job output:
```bash
echo "transformers_override=git+https://github.com/huggingface/transformers@main" >> "$GITHUB_OUTPUT"
echo "requires_optimum_recheck=true" >> "$GITHUB_OUTPUT"
echo "model_type=<model_type>" >> "$GITHUB_OUTPUT"
```

This allows the orchestrator to trigger a Pass-2 deploy with the updated
dependency before declaring success.

### 6. Check optimum-intel compatibility

After upgrading Transformers, verify that the installed `optimum-intel` is
compatible:

```python
from optimum.exporters.tasks import TasksManager
try:
    tasks = TasksManager.get_supported_tasks_for_model_type(
        "<model_type>", exporter="openvino"
    )
    print(f"optimum-intel supports: {tasks}")
    optimum_ok = True
except KeyError:
    print("model_type not in optimum-intel → Optimum-Intel agent still needed")
    optimum_ok = False
```

If `optimum_ok=False`:
- Set `requires_optimum_new_arch=true` in the output.
- The orchestrator should invoke the **Optimum-Intel agent** with skill
  `optimum_add_model_support` even if tokenizer conversion itself succeeded.

## Decision Tree Summary

```
Tokenizer fails with "does not recognize this architecture"
  │
  ├─ Step 0: upstream_search → known fix found? → apply and test first
  │
  ├─ Install transformers from git (allowed origin: github.com/huggingface/)
  │     │
  │     ├─ Config loads OK
  │     │     │
  │     │     ├─ Round-trip OK → create dep patch → assemble cross-agent artifact
  │     │     │
  │     │     └─ Round-trip FAIL → error_class=tokenizer_error, iterate hypothesis
  │     │
  │     └─ Still fails → status=blocked, error_class=transformers_no_support
  │                       → Surrender Protocol
  │
  └─ Check optimum-intel TasksManager
        ├─ Supported → tokenizer fix alone sufficient
        └─ Not supported → flag requires_optimum_new_arch=true
```

> `--trust-remote-code` is not a branch in this tree. It is always a blocker.

## Experiment Journal

After each meaningful step (upstream search, each install attempt, each
round-trip test), append an entry to `experiments_log.json`.
Minimum fields: `attempt_id`, `hypothesis`, `outcome`, `error_summary`,
`insights`, `next_hypothesis`.

## Output Contract (additions to base agent)

| Field | Value |
|---|---|
| `error_class` | `unknown_arch_transformers_too_old` \| `transformers_no_support` \| `tokenizer_error` |
| `transformers_override` | git URL if upgrade was needed, else empty |
| `requires_optimum_recheck` | `true` if deploy must be re-run with new transformers |
| `requires_optimum_new_arch` | `true` if optimum-intel has no config for this model_type |

## Surrender Checklist

Before triggering the Surrender Protocol, verify all boxes are ticked:

- [ ] Upstream search completed (Step 0) and results documented
- [ ] `model_type` correctly identified from traceback
- [ ] git-HEAD transformers installed and tested in an isolated venv
- [ ] Round-trip test attempted (if transformers loaded OK)
- [ ] `experiments_log.json` has an entry for every attempt
- [ ] Root cause is clearly one of: `transformers_no_support`, `tokenizer_error`, security blocker
- [ ] Human handoff sections in `agent_report.md` are specific and actionable

## Notes

- Never use `--trust-remote-code` — its requirement is an immediate Surrender trigger.
- Never hard-pin a `transformers` git SHA in `requirements.txt` of this repo —
  patch files are for the CI pipeline environment only.
- Document the exact `transformers.__version__` in `transformers_dep.json` for reproducibility.
- When upgrading transformers, also verify the Rust `tokenizers` library
  version compatibility — new vocab sizes (e.g., 248320 for Qwen3.5) sometimes
  require an updated `tokenizers` package (check allowlist before installing).
