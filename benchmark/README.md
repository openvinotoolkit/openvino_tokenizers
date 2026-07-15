# OpenVINO Tokenizers Benchmark

This is a CLI benchmark tool that compares OpenVINO Tokenizers and Huggingface tokenizers throughput and latency. 
It is based on a [vLLM benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks). 
Use `pip install -U openvino-tokenizers[benchmark]` or `pip install -U .[benchmark]` to install dependencies.

## Usage

```shell
usage: benchmark.py [-h] [-d DATASET] [--converted_tokenizer CONVERTED_TOKENIZER] [-n NUM_PAIRS]
                    [--trust-remote-code] [--dump-latency-stats] [--print-per-layer-stats] [--tput]
                    [-b BATCH] [--seed SEED] [-o OUTPUT_DIR]
                    model_id

OpenVINO Tokenizers Benchmark

positional arguments:
  model_id              The model id of a tokenizer hosted in a model repo on huggingface.co or a path to a saved Huggingface tokenizer directory

options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Path to the dataset.
  --converted_tokenizer CONVERTED_TOKENIZER, --converted-tokenizer CONVERTED_TOKENIZER
                        Path to converted tokenizer.
  -n NUM_PAIRS, --num_pairs NUM_PAIRS
                        Number of prompt/completion pairs to sample from the dataset.
  --trust-remote-code, --trust_remote_code
                        Pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained`. It will execute code present on the Hub on your local
                        machine.
  --dump-latency-stats, --dump_latency_stats
                        Save csv file with latency stats.
  --print-per-layer-stats, --print_per_layer_stats
                        Print execution info for each tokenizer layer.
  --tput                Use THROUGHPUT performance hint.
  -b BATCH, --batch BATCH
                        Batch size
  --seed SEED           Random seed for data sampling
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for generated CSV, metadata, and plot files.
```

`--num_pairs` controls the number of sampled prompt/completion pairs. `--batch` only controls how those prompts are
grouped into batches.

The benchmark reports synchronous OpenVINO and Hugging Face throughput in prompts/s and batches/s. Async OpenVINO
results are reported as queued execution throughput and queued latency; they are not treated as directly comparable to
Hugging Face synchronous per-call latency.

Generated files are written to `--output-dir`:

- `benchmark_metadata_<model>.json`: versions, workload details, throughput, and latency percentiles.
- `latency_benchmark_<model>.jpeg`: sync latency comparison and async queued latency plots.
- `latency_res_<model>.csv`: per-batch latency data, when `--dump-latency-stats` is passed.

## Download Dataset

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
