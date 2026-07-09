import argparse
import json
import platform
import random
from collections.abc import Iterable
from itertools import chain, islice
from pathlib import Path
from random import sample, shuffle
from time import perf_counter
from typing import Any, Optional

import matplotlib.pyplot as plt
import openvino as ov
import pandas as pd
import psutil
import seaborn as sns
from openvino import AsyncInferQueue, CompiledModel, InferRequest, ProfilingInfo, properties
from openvino_tokenizers import convert_tokenizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import __version__ as transformers_version


def sample_texts(
    dataset_path: str,
    num_texts: int = 1000,
) -> list[tuple[str, str]]:
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Keep the first two turns of each conversation
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in sample(dataset, k=num_texts)  # nosec
    ]
    shuffle(dataset)
    return dataset


def batch_iter(dataset: Iterable, batch: int = 1) -> Iterable[list]:
    dataset_iter = iter(dataset)
    while next_batch := list(islice(dataset_iter, batch)):
        yield next_batch


def benchmark_tokenizer_async(
    ov_tokenizer: CompiledModel, dataset: list[tuple[str, str]], batch: int = 1
) -> tuple[pd.Series, float, float]:
    def callback(
        ir: InferRequest,
        user_data: tuple[list[float], float, int],
    ) -> None:
        end = perf_counter()
        times, start, idx = user_data
        times[idx] = end - start

    prompts = list(chain.from_iterable(dataset))
    prompt_batches = list(batch_iter(prompts, batch))
    iterations = len(prompt_batches)
    async_queue = AsyncInferQueue(ov_tokenizer)
    async_queue.set_callback(callback)
    times = [0.0 for _ in range(iterations)]

    bench_start = perf_counter()
    for idx, prompt in tqdm(
        enumerate(prompt_batches), total=iterations, desc="Async benchmark"
    ):
        start = perf_counter()
        async_queue.start_async(prompt, (times, start, idx))
    async_queue.wait_all()
    elapsed = perf_counter() - bench_start

    results = pd.Series(data=times, name="OV_ASYNC_QUEUED")

    return results, len(prompts) / elapsed, iterations / elapsed


def construct_pc_series(perf_counts: list[ProfilingInfo], stats: dict[str, Any]) -> dict[str, Any]:
    for pi in perf_counts:
        if pi.status == pi.NOT_RUN:
            continue
        node_name = pi.node_name
        real_time = pi.real_time.total_seconds()
        stats[node_name] = real_time

    return stats


def benchmark_tokenizers(
    ov_tokenizer: CompiledModel,
    hf_tokenizer: PreTrainedTokenizerBase,
    dataset: list[tuple[str, str]],
    per_layer_stats: bool = False,
    batch: int = 1,
) -> pd.DataFrame:
    columns = ["prompt", "OV", "HF"]
    results = []

    # warmup
    process = psutil.Process()
    for repeat in range(1, 2):
        # print time of the first tokenization pass
        print(f"Warmup iteration {repeat}")

        mem_before_ov = process.memory_info().rss / 1024 / 1024  # MB
        ov_start = perf_counter()
        ov_tokenizer(["test " * repeat])
        ov_time = perf_counter() - ov_start
        mem_after_ov = process.memory_info().rss / 1024 / 1024  # MB
        print(f"OV warmup time: {ov_time:.6f} seconds, memory delta: {mem_after_ov - mem_before_ov:.2f} MB")
        hf_tokenizer(["test " * repeat])

    ov_input_ids = []
    ov_perf_counters = []
    prompt_batches = list(batch_iter(chain.from_iterable(dataset), batch))
    for prompt in tqdm(
        prompt_batches, total=len(prompt_batches), desc="Sync benchmark"
    ):
        res = [prompt]

        ov_start = perf_counter()
        ov_res = ov_tokenizer(prompt)
        res.append(perf_counter() - ov_start)

        if per_layer_stats:
            stats = {
                "Prompt Length": sum(len(text) for text in prompt),
                "# Tokens": ov_res["input_ids"].shape[-1],
            }
            stats = construct_pc_series(ov_tokenizer._infer_request.profiling_info, stats)

            ov_perf_counters.append(stats)

        results.append(res)
        ov_input_ids.append(ov_res["input_ids"])

    equal_ids_count = 0
    for res, ov_input_id in tqdm(zip(results, ov_input_ids), total=len(results), desc="HF benchmark"):
        prompt, *_ = res
        hf_start = perf_counter()
        hf_res = hf_tokenizer(prompt, return_tensors="np", padding=True)
        res.append(perf_counter() - hf_start)

        equal_ids_count += (hf_res["input_ids"].shape == ov_input_id.shape) and (
            hf_res["input_ids"] == ov_input_id
        ).all()

    if ov_perf_counters:
        df = pd.DataFrame(ov_perf_counters)
        model_name = hf_tokenizer.name_or_path.rsplit("/")[-1]
        df.to_csv(f"{model_name}_pc.csv", index=False)

        df_describe = df.describe(percentiles=[0.5])
        print(
            df_describe.T.sort_values("mean", ascending=False).to_string(
                float_format="{:.6f}".format, formatters={"count": "{:.0f}".format}
            )
        )

    print(f"input_ids matched for {equal_ids_count}/{len(results)} samples")
    return pd.DataFrame(results, columns=columns)


def dump_latency_stats(results: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    sorted_res = results.sort_values("Prompt Length, chars")
    sorted_res["OV/HF latency"] = sorted_res["OV"] / sorted_res["HF"]

    sorted_res.to_csv(output_dir / f"latency_res_{model_name}.csv", index=False)


def _latency_summary(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def print_stats(
    results: pd.DataFrame, async_prompt_fps: float, async_batch_fps: float, batch: int = 1
) -> dict[str, Any]:
    prompt_counts = results["prompt"].apply(len)
    prompt_count = int(prompt_counts.sum())
    batch_count = len(results)
    ov_total = results["OV"].sum()
    hf_total = results["HF"].sum()
    ov_prompt_fps = prompt_count / ov_total
    hf_prompt_fps = prompt_count / hf_total
    ov_batch_fps = batch_count / ov_total
    hf_batch_fps = batch_count / hf_total

    print("Throughput:")
    print(
        f"Sync  OV: {ov_prompt_fps:.3f} prompts/s, {ov_batch_fps:.3f} batches/s; "
        f"HF: {hf_prompt_fps:.3f} prompts/s, {hf_batch_fps:.3f} batches/s; "
        f"OV/HF prompts/s: {ov_prompt_fps / hf_prompt_fps:.3f}"
    )
    print(
        f"Async OV: {async_prompt_fps:.3f} prompts/s, {async_batch_fps:.3f} batches/s "
        "(queued execution)"
    )

    latency_stats = pd.DataFrame(
        {
            "OV sync": _latency_summary(results["OV"]),
            "HF sync": _latency_summary(results["HF"]),
            "OV async queued": _latency_summary(results["OV_ASYNC_QUEUED"]),
        }
    ).T
    print("Latency, seconds:")
    print(latency_stats.to_string(float_format="{:.6f}".format))

    print("Workload:")
    print(
        f"Pairs: {prompt_count // 2}, prompts: {prompt_count}, batches: {batch_count}, "
        f"requested batch size: {batch}"
    )
    print("Prompt length stats, chars:")
    print(results["Prompt Length, chars"].describe().drop("count").to_string(float_format="{:.3f}".format))

    return {
        "prompt_count": prompt_count,
        "batch_count": batch_count,
        "ov_prompts_per_sec": float(ov_prompt_fps),
        "hf_prompts_per_sec": float(hf_prompt_fps),
        "async_ov_prompts_per_sec": float(async_prompt_fps),
        "ov_batches_per_sec": float(ov_batch_fps),
        "hf_batches_per_sec": float(hf_batch_fps),
        "async_ov_batches_per_sec": float(async_batch_fps),
        "latency_seconds": {
            "ov_sync": _latency_summary(results["OV"]),
            "hf_sync": _latency_summary(results["HF"]),
            "ov_async_queued": _latency_summary(results["OV_ASYNC_QUEUED"]),
        },
    }


def build_plot(results: pd.DataFrame, save_file: Optional[str] = None, **kwargs) -> plt.Figure:
    cmap = sns.cubehelix_palette(rot=-0.2, as_cmap=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    max_latency_sync = max(results["OV"].max(), results["HF"].max())

    for ax, is_log in zip(axes[:2], (False, True)):
        ax.plot([0, max_latency_sync], [0, max_latency_sync], linestyle="dashed", linewidth=1, color="r")

        if is_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("Sync latency, log scale")
        else:
            ax.set_title("Sync latency")

        sns.scatterplot(
            data=results, x="OV", y="HF", hue="Prompt Length, chars", palette=cmap, ax=ax
        )

    axes[2].set_title("Async queued latency")
    sns.scatterplot(
        data=results,
        x="Prompt Length, chars",
        y="OV_ASYNC_QUEUED",
        hue="Batch Prompts",
        palette="viridis",
        ax=axes[2],
    )

    if (title := kwargs.get("title")) is not None:
        fig.suptitle(title)

    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file)
    return fig


def main(
    checkpoint: str,
    dataset: str,
    num_pairs: int = 1000,
    batch: int = 1,
    trust: bool = False,
    dump_latency: bool = False,
    per_layer_stats: bool = False,
    tput: bool = False,
    converted_tokenizer: Optional[str] = None,
    output_dir: str = ".",
) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset

    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust)

    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    hint = properties.hint.PerformanceMode.THROUGHPUT if tput else properties.hint.PerformanceMode.LATENCY
    config = {properties.hint.performance_mode(): hint}
    if per_layer_stats:
        config[properties.enable_profiling()] = True

    core = ov.Core()
    if converted_tokenizer:
        tokenizer_model = core.read_model(Path(converted_tokenizer) / "openvino_tokenizer.xml")
    else:
        tokenizer_model = convert_tokenizer(hf_tokenizer)

    start_compile = perf_counter()
    ov_tokenizer = core.compile_model(tokenizer_model, "CPU", config)
    end_compile = perf_counter()
    print(f"Time to compile tokenizer model: {end_compile - start_compile}s")

    dataset = sample_texts(dataset_path, num_pairs)
    result_df = benchmark_tokenizers(ov_tokenizer, hf_tokenizer, dataset, per_layer_stats, batch)
    async_results, async_prompt_fps, async_batch_fps = benchmark_tokenizer_async(ov_tokenizer, dataset, batch)
    result_df = result_df.assign(OV_ASYNC_QUEUED=async_results.values)
    result_df["Batch Prompts"] = result_df["prompt"].apply(len)
    result_df["Prompt Length, chars"] = result_df["prompt"].apply(
        lambda prompts: sum(len(prompt) for prompt in prompts)
    )

    model_name = checkpoint.rsplit("/", 1)[-1]
    stats = print_stats(result_df, async_prompt_fps, async_batch_fps, batch)

    metadata = {
        "model": checkpoint,
        "model_name": model_name,
        "openvino_version": ov.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset_path,
        "requested_pairs": num_pairs,
        "sampled_pairs": len(dataset),
        "batch_size": batch,
        "performance_hint": "THROUGHPUT" if tput else "LATENCY",
        **stats,
    }
    with open(output_dir_path / f"benchmark_metadata_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if dump_latency:
        dump_latency_stats(result_df, model_name, output_dir_path)

    title = (
        f"OV vs HF Latency\n{checkpoint}, batch_size={batch}\n"
        f"OV: {stats['ov_prompts_per_sec']:.1f} prompts/s; "
        f"OV Async {stats['async_ov_prompts_per_sec']:.1f} prompts/s; "
        f"HF  {stats['hf_prompts_per_sec']:.1f} prompts/s"
    )
    build_plot(result_df, output_dir_path / f"latency_benchmark_{model_name}.jpeg", title=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO Tokenizers Benchmark")
    parser.add_argument(
        "model_id",
        type=str,
        help=(
            "The model id of a tokenizer hosted in a model repo on huggingface.co "
            "or a path to a saved Huggingface tokenizer directory"
        ),
    )
    parser.add_argument("-d", "--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument(
        "--converted_tokenizer", "--converted-tokenizer", type=str, default=None, help="Path to converted tokenizer."
    )
    parser.add_argument(
        "-n",
        "--num_pairs",
        type=int,
        default=1000,
        help="Number of prompt/completion pairs to sample from the dataset.",
    )
    parser.add_argument(
        "--trust-remote-code",
        "--trust_remote_code",
        required=False,
        action="store_true",
        help=(
            "Pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained`. It will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--dump-latency-stats",
        "--dump_latency_stats",
        required=False,
        action="store_true",
        help="Save csv file with latency stats.",
    )
    parser.add_argument(
        "--print-per-layer-stats",
        "--print_per_layer_stats",
        required=False,
        action="store_true",
        help="Print execution info for each tokenizer layer.",
    )
    parser.add_argument(
        "--tput",
        required=False,
        action="store_true",
        help="Use THROUGHPUT performance hint.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        required=False,
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=None,
        help="Random seed for data sampling",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=False,
        type=str,
        default=".",
        help="Directory for generated CSV, metadata, and plot files.",
    )

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    main(
        args.model_id,
        args.dataset,
        args.num_pairs,
        args.batch,
        trust=args.trust_remote_code,
        dump_latency=args.dump_latency_stats,
        per_layer_stats=args.print_per_layer_stats,
        tput=args.tput,
        converted_tokenizer=args.converted_tokenizer,
        output_dir=args.output_dir,
    )
