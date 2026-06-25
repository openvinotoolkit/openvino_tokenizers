import json
import os
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

# todo: create a unified source of tokenizer models and types for all tests
unigram_tokenizers = [
    "camembert-base",
    "google/flan-t5-xxl",
    "BAAI/bge-reranker-v2-m3",
    "microsoft/deberta-v3-base",  # byte fallback
    "facebook/musicgen-small",
    "rinna/bilingual-gpt-neox-4b",  # t5-tokenizer
]


def build_coverege_report(session: pytest.Session) -> None:
    import pandas as pd
    from pytest_harvest import get_session_results_df

    # Use row.get so the report still builds if a run did not exercise every fixture type
    # (those *_param columns are then absent from the results frame).
    def add_tokenizer_type(row):
        if not pd.isnull(row.get("hf_wordpiece_tokenizers_param")):
            return "WordPiece"
        if not pd.isnull(row.get("hf_wordpiece_tokenizers_with_padding_sides_param")):
            return "WordPiece"
        if not pd.isnull(row.get("hf_bpe_tokenizers_param")):
            return "BPE"
        if not pd.isnull(row.get("hf_bpe_tokenizers_with_padding_sides_param")):
            return "BPE"
        if not pd.isnull(row.get("hf_sentencepiece_tokenizers_param")):
            if row.get("is_sentencepiece_backend_param"):
                return "SentencePiece"
            elif row.get("hf_sentencepiece_tokenizers_param") in unigram_tokenizers:
                return "Unigram"
            else:
                return "BPE"
        if not pd.isnull(row.get("hf_sentencepiece_tokenizers_with_padding_sides_param")):
            if row.get("is_sentencepiece_backend_param"):
                return "SentencePiece"
            elif row.get("hf_sentencepiece_tokenizers_with_padding_sides_param") in unigram_tokenizers:
                return "Unigram"
            else:
                return "BPE"
        if not pd.isnull(row.get("hf_tiktoken_tokenizers_param")):
            return "Tiktoken"
        if not pd.isnull(row.get("hf_tiktoken_tokenizers_with_padding_sides_param")):
            return "Tiktoken"

    results_df = get_session_results_df(session)

    # get_session_results_df can come back empty (e.g. under xdist if the harvested worker
    # results were already cleaned up, or if no harvestable tests ran). The coverage report
    # is only meaningful for a full-coverage run, so skip it rather than crash.
    expected_param_columns = [
        "hf_wordpiece_tokenizers_param",
        "hf_wordpiece_tokenizers_with_padding_sides_param",
        "hf_bpe_tokenizers_param",
        "hf_bpe_tokenizers_with_padding_sides_param",
        "hf_sentencepiece_tokenizers_param",
        "hf_sentencepiece_tokenizers_with_padding_sides_param",
        "hf_tiktoken_tokenizers_param",
        "hf_tiktoken_tokenizers_with_padding_sides_param",
    ]
    if results_df.empty or not any(column in results_df.columns for column in expected_param_columns):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "Skipping README coverage report: no tokenizer results were harvested "
                "(run the full tokenizers_test.py suite with --update_readme to regenerate it)."
            )
        return

    # Some fixture-type columns may be missing for a partial run; add them as NaN so the
    # downstream fillna chain and column selection below do not raise KeyError.
    for column in expected_param_columns:
        if column not in results_df.columns:
            results_df[column] = pd.NA

    results_df["Tokenizer Type"] = results_df.apply(add_tokenizer_type, axis=1)
    results_df = results_df[results_df.status != "skipped"]  # filter skipped tests
    results_df.hf_wordpiece_tokenizers_param.fillna(results_df.hf_bpe_tokenizers_param, inplace=True)
    results_df.hf_wordpiece_tokenizers_param.fillna(results_df.hf_sentencepiece_tokenizers_param, inplace=True)
    results_df.hf_wordpiece_tokenizers_param.fillna(results_df.hf_tiktoken_tokenizers_param, inplace=True)
    results_df.hf_wordpiece_tokenizers_param.fillna(
        results_df.hf_wordpiece_tokenizers_with_padding_sides_param, inplace=True
    )
    results_df.hf_wordpiece_tokenizers_param.fillna(
        results_df.hf_bpe_tokenizers_with_padding_sides_param, inplace=True
    )
    results_df.hf_wordpiece_tokenizers_param.fillna(
        results_df.hf_sentencepiece_tokenizers_with_padding_sides_param, inplace=True
    )
    results_df.hf_wordpiece_tokenizers_param.fillna(
        results_df.hf_tiktoken_tokenizers_with_padding_sides_param, inplace=True
    )
    results_df.status = (results_df.status == "passed").astype(int)
    results_df = results_df.dropna(subset=["hf_wordpiece_tokenizers_param"])
    results_df["Model"] = results_df.hf_wordpiece_tokenizers_param + [
        "_legacy" * value for value in results_df.index.str.contains("Slow")
    ]
    results_df = results_df[["Tokenizer Type", "Model", "test_string", "status"]]
    grouped_by_model = results_df.groupby(["Tokenizer Type", "Model"]).agg({"status": ["mean", "count"]}).reset_index()
    grouped_by_model.columns = ["Tokenizer Type", "Model", "Output Matched, %", "Number of Tests"]
    grouped_by_model["Output Matched, %"] *= 100
    grouped_by_type = results_df.groupby(["Tokenizer Type"]).agg({"status": ["mean", "count"]}).reset_index()
    grouped_by_type.columns = ["Tokenizer Type", "Output Matched, %", "Number of Tests"]
    grouped_by_type["Output Matched, %"] *= 100

    readme_path = Path("../README.md")
    with open(readme_path) as f:
        old_readme = f.read().split("## Test Results")[0]

    new_readme = StringIO()
    new_readme.write()
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
    from pytest_harvest import is_main_process

    # Under pytest-xdist this hook runs on every worker as well as the controller. Only
    # the controller has the aggregated terminalreporter stats and should read/write the
    # pass-rate and status files; workers must not touch them.
    if not is_main_process(session):
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
