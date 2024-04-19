# OpenVINO Tokenizers Benchmark

This is a CLI benchmark tool that compares OpenVINO Tokenizers and Huggingface tokenizers throughput and latency. 
It is based on a [vLLM benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks). 
Use `pip install -U openvino-tokenizers[benchmark]` or `pip install -U .[benchmark]` to install dependencies.

## Usage

```shell
usage: benchmark.py [-h] [-d DATASET] [-n NUM_PAIRS] [--trust-remote-code] [--log-scale] model_id

OpenVINO Tokenizers Benchmark

positional arguments:
  model_id              The model id of a tokenizer hosted in a model repo on huggingface.co or a path to a saved Huggingface tokenizer directory

options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Path to the dataset.
  -n NUM_PAIRS, --num_pairs NUM_PAIRS
                        Number of prompt/completion pairs to sample from the dataset.
  --trust-remote-code, --trust_remote_code
                        Pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained`. It will execute code present on the Hub on your local
                        machine.
  --log-scale, --log_scale
                        Use log scale for the plot.
```

## Download Dataset

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```