import argparse
import json
import random
from itertools import chain, islice
from random import sample, shuffle
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from openvino import AsyncInferQueue, CompiledModel, InferRequest, compile_model
from openvino.runtime import ProfilingInfo, properties
from openvino_tokenizers import convert_tokenizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def sample_texts(
    dataset_path: str,
    num_texts: int = 1000,
) -> List[Tuple[str, str]]:
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in sample(dataset, k=num_texts)
    ]
    shuffle(dataset)
    return dataset


def batch_iter(dataset: Iterable, batch: int = 1):
    dataset_iter = iter(dataset)
    while next_batch := list(islice(dataset_iter, batch)):
        yield next_batch


def benchmark_tokenizer_async(
    ov_tokenizer: CompiledModel, dataset: List[Tuple[str, str]], batch: int = 1
) -> Tuple[pd.Series, float]:
    def callback(
        ir: InferRequest,
        user_data: Tuple[List[int], float, int],
    ) -> None:
        end = perf_counter()
        times, start, idx = user_data
        times[idx] = end - start

    iterations = len(dataset) * 2 // batch
    async_queue = AsyncInferQueue(ov_tokenizer)
    async_queue.set_callback(callback)
    times = [0 for _ in range(iterations)]

    bench_start = perf_counter()
    for idx, prompt in tqdm(
        enumerate(batch_iter(chain.from_iterable(dataset), batch)), total=iterations, desc="Async benchmark"
    ):
        start = perf_counter()
        async_queue.start_async(prompt, (times, start, idx))
    async_queue.wait_all()
    elapsed = perf_counter() - bench_start

    results = pd.Series(data=times, name="OV_Async")

    return results, iterations * batch / elapsed


def construct_pc_series(perf_counts: List[ProfilingInfo], stats: Dict[str, Any]) -> Dict[str, Any]:
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
    dataset: List[Tuple[str, str]],
    per_layer_stats: bool = False,
    batch: int = 1,
) -> pd.DataFrame:
    columns = ["prompt", "OV", "HF"]
    results = []

    # warmup
    for repeat in range(1, 11):
        ov_tokenizer(["test " * repeat])
        hf_tokenizer(["test " * repeat])

    ov_perf_counters = []
    for prompt in tqdm(
        batch_iter(chain.from_iterable(dataset), batch), total=len(dataset) * 2 / batch, desc="Sync benchmark"
    ):
        res = [prompt]

        ov_start = perf_counter()
        ov_res = ov_tokenizer(prompt)
        res.append(perf_counter() - ov_start)

        hf_start = perf_counter()
        hf_tokenizer(prompt)
        res.append(perf_counter() - hf_start)

        results.append(res)

        if per_layer_stats:
            stats = {
                "Prompt Length": sum(len(text) for text in prompt),
                "# Tokens": ov_res["input_ids"].shape[-1],
            }
            stats = construct_pc_series(ov_tokenizer._infer_request.profiling_info, stats)

            ov_perf_counters.append(stats)

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

    return pd.DataFrame(results, columns=columns)


def dump_latency_stats(results: pd.DataFrame, model_name: str) -> None:
    sorted_res = results.sort_values("Prompt Length, chars")
    sorted_res["OV vs HF"] = sorted_res["OV"] / sorted_res["HF"]
    sorted_res["OV_ASYNC vs HF"] = sorted_res["OV_ASYNC"] / sorted_res["HF"]

    sorted_res.to_csv(f"latency_res_{model_name}.csv", index=False)


def print_stats(
    results: pd.DataFrame, async_fps: Optional[float] = None, batch: int = 1
) -> Tuple[float, float, float]:
    data_size = len(results) * batch
    ov_fps = data_size / results["OV"].sum()
    hf_fps = data_size / results["HF"].sum()

    print(f"Sync:  OV: {ov_fps:.3f} FPS, HF: {hf_fps:.3f} FPS, OV/HF: {ov_fps/hf_fps}")
    print(f"Async: OV: {async_fps:.3f} FPS, HF: {hf_fps:.3f} FPS, OV/HF: {async_fps/hf_fps}")
    print("Latency and prompt stats:")
    stats = results.describe().drop("count")
    print(stats)
    return ov_fps, async_fps, hf_fps


def build_plot(results: pd.DataFrame, save_file: Optional[str] = None, **kwargs) -> plt.Figure:
    cmap = sns.cubehelix_palette(rot=-0.2, as_cmap=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    max_latency_sync = max(results["OV"].max(), results["HF"].max())
    max_latency_async = max(results["OV_ASYNC"].max(), results["HF"].max())

    logs = (False, False, True, True)
    asyncs = (False, True, False, True)
    for ax, is_log, is_async in zip(axes.flatten(), logs, asyncs):
        max_latency = max_latency_async if is_async else max_latency_sync
        ax.plot([0, max_latency], [0, max_latency], linestyle="dashed", linewidth=1, color="r")

        if is_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        sns.scatterplot(
            data=results, x="OV" + "_ASYNC" * is_async, y="HF", hue="Prompt Length, chars", palette=cmap, ax=ax
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
) -> None:
    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust)

    hint = properties.hint.PerformanceMode.THROUGHPUT if tput else properties.hint.PerformanceMode.LATENCY
    config = {properties.hint.performance_mode(): hint}
    if per_layer_stats:
        config[properties.enable_profiling()] = True

    start_compile = perf_counter()
    ov_tokenizer = compile_model(convert_tokenizer(hf_tokenizer), "CPU", config)
    end_compile = perf_counter()
    print(f"Time to compile tokenizer model: {end_compile - start_compile}s")

    dataset = sample_texts(dataset, batch * num_pairs)
    result_df = benchmark_tokenizers(ov_tokenizer, hf_tokenizer, dataset, per_layer_stats, batch)
    async_results, async_fps = benchmark_tokenizer_async(ov_tokenizer, dataset, batch)
    result_df = result_df.assign(OV_ASYNC=async_results.values)
    result_df["Prompt Length, chars"] = result_df["prompt"].apply(
        lambda prompts: sum(len(prompt) for prompt in prompts)
    )

    ov_fps, async_fps, hf_fps = print_stats(result_df, async_fps, batch)
    model_name = checkpoint.rsplit("/", 1)[-1]

    if dump_latency:
        dump_latency_stats(result_df, model_name)

    title = (
        f"OV vs HF Latency\n{checkpoint}, batch_size={batch}\n"
        f"OV: {ov_fps:.1f} FPS; "
        f"OV Async {async_fps:.1f} FPS; "
        f"HF  {hf_fps:.1f} FPS"
    )
    build_plot(result_df, f"latency_benchmark_{model_name}.jpeg", title=title)


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
    )
