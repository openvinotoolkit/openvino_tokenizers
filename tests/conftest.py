import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from math import isclose
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update_readme",
        action="store_true",
        default=False,
        help="Update test coverage report in README.md",
    )


PASS_RATES_FILE = Path(__file__).parent / "pass_rates.json"
STATUSES_FILE = Path(__file__).parent / "stats.json"
README_FILE = Path(__file__).resolve().parents[1] / "README.md"

# todo: create a unified source of tokenizer models and types for all tests
unigram_tokenizers = [
    "camembert-base",
    "google/flan-t5-xxl",
    "BAAI/bge-reranker-v2-m3",
    "microsoft/deberta-v3-base",  # byte fallback
    "facebook/musicgen-small",
    "rinna/bilingual-gpt-neox-4b",  # t5-tokenizer
]


def _get_tokenizers_test_module():
    for name, module in sys.modules.items():
        if name.endswith("tokenizers_test"):
            return module

    try:
        import tests.tokenizers_test as tokenizers_test
    except ModuleNotFoundError:
        import tokenizers_test

    return tokenizers_test


def _get_model_id_map(tokenizers_test):
    models = [
        *tokenizers_test.wordpiece_models,
        *tokenizers_test.bpe_models,
        *tokenizers_test.sentencepiece_models,
        *tokenizers_test.tiktiken_models,
    ]
    return {model.split("/")[-1]: model for model in models}


def _get_nodeid_parts(nodeid):
    test_id = nodeid.split("::")[-1]
    if "[" not in test_id:
        return test_id, ""
    test_name, params = test_id.split("[", 1)
    return test_name, params.rsplit("]", 1)[0]


def _get_model_from_params(params, model_id_map):
    for model_id in sorted(model_id_map, key=len, reverse=True):
        if params == model_id or params.startswith(f"{model_id}-"):
            return model_id_map[model_id], params[len(model_id) :].lstrip("-")
    return None, None


def _get_tokenizer_type_from_report(test_name, model, params_tail):
    if test_name.startswith(("test_hf_wordpiece_tokenizers", "test_wordpiece_model_detokenizer")):
        return "WordPiece"
    if test_name.startswith(("test_hf_bpe_tokenizers", "test_bpe_model_tokenizer", "test_bpe_detokenizer")):
        return "BPE"
    if test_name.startswith(
        (
            "test_tiktoken_tokenizers",
            "test_hf_tiktoken_tokenizers",
            "test_tiktoken_model_tokenizer",
            "test_tiktoken_detokenizer",
        )
    ):
        return "Tiktoken"
    if test_name.startswith(("test_sentencepiece_model_tokenizer", "test_hf_sentencepiece_tokenizers", "test_sentencepiece_model_detokenizer")):
        if "sp_backend" in params_tail.split("-"):
            return "SentencePiece"
        if model in unigram_tokenizers:
            return "Unigram"
        return "BPE"
    return None


def _build_coverage_results_from_terminal_stats(session):
    import pandas as pd

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return pd.DataFrame()

    tokenizers_test = _get_tokenizers_test_module()
    model_id_map = _get_model_id_map(tokenizers_test)
    rows = []
    index = []

    for status, status_value in (("passed", 1), ("failed", 0)):
        for report in reporter.stats.get(status, []):
            if "tokenizers_test.py::" not in report.nodeid:
                continue

            test_name, params = _get_nodeid_parts(report.nodeid)
            model, params_tail = _get_model_from_params(params, model_id_map)
            if model is None:
                continue

            tokenizer_type = _get_tokenizer_type_from_report(test_name, model, params_tail)
            if tokenizer_type is None:
                continue

            rows.append(
                {
                    "Tokenizer Type": tokenizer_type,
                    "Model": model,
                    "test_string": params_tail,
                    "status": status_value,
                }
            )
            index.append(report.nodeid)

    return pd.DataFrame(rows, index=index)


def _is_xdist_worker(session):
    return hasattr(session.config, "workerinput")


def build_coverege_report(session: pytest.Session) -> None:
    results_df = _build_coverage_results_from_terminal_stats(session)
    if results_df.empty:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "Skipping README coverage report: no tokenizer results were harvested "
                "(run the full tokenizers_test.py suite with --update_readme to regenerate it)."
            )
        return

    results_df["Model"] = results_df["Model"] + ["_legacy" * value for value in results_df.index.str.contains("Slow")]
    results_df = results_df[["Tokenizer Type", "Model", "test_string", "status"]]
    grouped_by_model = results_df.groupby(["Tokenizer Type", "Model"]).agg({"status": ["mean", "count"]}).reset_index()
    grouped_by_model.columns = ["Tokenizer Type", "Model", "Output Matched, %", "Number of Tests"]
    grouped_by_model["Output Matched, %"] *= 100
    grouped_by_type = results_df.groupby(["Tokenizer Type"]).agg({"status": ["mean", "count"]}).reset_index()
    grouped_by_type.columns = ["Tokenizer Type", "Output Matched, %", "Number of Tests"]
    grouped_by_type["Output Matched, %"] *= 100

    readme_path = README_FILE
    with open(readme_path) as f:
        old_readme = f.read().split("## Test Results")[0]

    new_readme = StringIO()
    new_readme.write(old_readme)
    new_readme.write(
        "## Test Results\n\n"
        "This report is autogenerated and includes tokenizers and detokenizers tests. The `Output Matched, %` column "
        "shows the percent of test strings for which the results of OpenVINO and Huggingface Tokenizers are the same. "
        "To update the report run `pytest --update_readme tokenizers_test.py -n auto` in `tests` directory.\n\n"
        "### Output Match by Tokenizer Type\n\n"
    )
    is_pandas_2 = tuple(map(int, version("pandas").split("."))) >= (2, 0, 0)
    if is_pandas_2:
        grouped_by_type.style.format(precision=2).hide(axis="index").to_html(new_readme, exclude_styles=True)
    else:
        grouped_by_type.style.format(precision=2).hide_index().to_html(new_readme, exclude_styles=True)
    new_readme.write("\n### Output Match by Model\n\n")
    if is_pandas_2:
        grouped_by_model.style.format(precision=2).hide(axis="index").to_html(new_readme, exclude_styles=True)
    else:
        grouped_by_model.style.format(precision=2).hide_index().to_html(new_readme, exclude_styles=True)

    new_readme.write(
        "\n### Recreating Tokenizers From Tests\n\n"
        "In some tokenizers, you need to select certain settings so that their output is closer "
        "to the Huggingface tokenizers:\n"
        "- `THUDM/chatglm3-6b` detokenizer don't skips special tokens. Use `skip_special_tokens=False` "
        "during conversion\n"
        "- All tested tiktoken based detokenizers leave extra spaces. Use `clean_up_tokenization_spaces=False` "
        "during conversion\n"
    )
    with open(readme_path, "w") as f:
        f.write(new_readme.getvalue())


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: pytest.ExitCode) -> None:
    """
    Tests fail if the test pass rate decreases
    """
    # Under pytest-xdist this hook runs on every worker as well as the controller. Only
    # the controller has the aggregated terminalreporter stats and should read/write the
    # pass-rate and status files; workers must not touch them.
    if _is_xdist_worker(session):
        return

    if session.config.getoption("update_readme", default=False):
        build_coverege_report(session)

    if exitstatus != pytest.ExitCode.TESTS_FAILED:
        return

    with open(PASS_RATES_FILE) as f:
        previous_rates = json.load(f)
    with open(STATUSES_FILE) as f:
        previous_statuses = json.load(f)

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = reporter.stats

    # The pass rate must only reflect tokenizers_test.py. A full `pytest tests` run also
    # collects optional modules (e.g. onnx_contrib_test.py, which is skipped at collection
    # when onnx is not installed); counting those would skew the rate and make CI diverge
    # from a `pytest tokenizers_test.py` run.
    def is_tokenizers_test(report):
        return "tokenizers_test.py" in report.nodeid

    # Derive the common nodeid prefix from all reports rather than session.items: under
    # xdist the controller collects no items (session.items is empty there), but the test
    # reports are aggregated from all workers. Using every report (not just tokenizers_test
    # ones) mirrors the old session.items-based prefix, so a full `pytest tests` run still
    # yields a shorter prefix and does not trigger the tokenizers-only stats.json rewrite.
    all_reports = [report for category in ("passed", "failed", "skipped") for report in stats.get(category, [])]
    parent = os.path.commonprefix([report.nodeid for report in all_reports]).strip("[]")

    passed = [report for report in stats.get("passed", []) if is_tokenizers_test(report)]
    failed = [report for report in stats.get("failed", []) if is_tokenizers_test(report)]
    skipped = [report for report in stats.get("skipped", []) if is_tokenizers_test(report)]

    relevant = len(passed) + len(failed)
    if relevant == 0:
        return
    pass_rate = 1 - len(failed) / relevant

    try:
        suffix = "transformers_v4" if version("transformers").startswith("4.") else ""
    except PackageNotFoundError:
        suffix = ""
    previous = previous_rates.get(parent + suffix, 0)

    new_statuses = {}
    for stat in passed:
        new_statuses[stat.nodeid] = "passed"
    for stat in skipped:
        new_statuses[stat.nodeid] = "skipped"
    for stat in failed:
        new_statuses[stat.nodeid] = "failed"

    rewrite_statuses = parent in ("tokenizers_test.py::test_", "tests/tokenizers_test.py::test_")

    if rewrite_statuses:
        new_statuses = {test_id[len(parent) :]: status for test_id, status in sorted(new_statuses.items())}
        with open(STATUSES_FILE, "w") as stat_file:
            json.dump(new_statuses, stat_file, indent=2)

    added_tests = {test_id: status for test_id, status in new_statuses.items() if test_id not in previous_statuses}

    changed_statuses = {(test_id, status) for test_id, status in new_statuses.items() if test_id not in added_tests}
    changed_statuses = changed_statuses.difference(set(previous_statuses.items()))
    if changed_statuses:
        reporter.write_line("CHANGED STATUS:")
        for test_id, new_status in changed_statuses:
            reporter.write_line(f"{previous_statuses[test_id]}->{new_status}: {test_id}")

    if isclose(pass_rate, previous):
        session.exitstatus = pytest.ExitCode.OK
        reporter.write_line(f"New pass rate isclose to previous: {pass_rate}")
        return

    if pass_rate > previous:
        reporter.write_line(f"New pass rate {pass_rate} is bigger then previous: {previous}")
        session.exitstatus = pytest.ExitCode.OK
        previous_rates[parent + suffix] = pass_rate

        with open(PASS_RATES_FILE, "w") as pass_rate_file:
            json.dump(previous_rates, pass_rate_file, indent=4)
    else:
        reporter.write_line(f"Pass rate is lower! Current: {pass_rate}, previous: {previous}")
