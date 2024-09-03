# OpenVINO Tokenizers

[![Downloads](https://static.pepy.tech/badge/openvino-tokenizers)](https://pepy.tech/project/openvino-tokenizers)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/openvino-tokenizers/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino-tokenizers)

OpenVINO Tokenizers adds text processing operations to OpenVINO.

## Features

- Perform tokenization and detokenization without third-party dependencies
- Convert a HuggingFace tokenizer into OpenVINO model tokenizer and detokenizer
- Combine OpenVINO models into a single model
- Add greedy decoding pipeline to text generation model

## Installation

(Recommended) Create and activate virtual env:
```bash
python3 -m venv venv
source venv/bin/activate
 # or
conda create --name openvino_tokenizers
conda activate openvino_tokenizers
```

### Minimal Installation

Use minimal installation when you have a converted OpenVINO tokenizer:
```bash
pip install openvino-tokenizers
 # or
conda install -c conda-forge openvino openvino-tokenizers
```

### Convert Tokenizers Installation

If you want to convert HuggingFace tokenizers into OpenVINO tokenizers:
```bash
pip install openvino-tokenizers[transformers]
 # or
conda install -c conda-forge openvino openvino-tokenizers && pip install transformers[sentencepiece] tiktoken
```

### Install Pre-release Version

Use `openvino-tokenizers[transformers]` to install tokenizers conversion dependencies.
```bash
pip install --pre -U openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

### Build and Install from Source

#### Using OpenVINO PyPI package

openvino-tokenizers build depends on [openvino](https://pypi.org/project/openvino/) package which will be automatically installed from PyPI during the build process. To install unreleased versions, you would need to install openvino package from the nightly distribution channel using `--extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly`

```bash
git clone https://github.com/openvinotoolkit/openvino_tokenizers.git
cd openvino_tokenizers
pip install . --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```
This command is the equivalent of minimal installation. Install tokenizers conversion dependencies if needed:
```bash
pip install transformers[sentencepiece] tiktoken
```
:warning: Latest commit of OpenVINO Tokenizers might rely on features that are not present in the release OpenVINO version.
Use [a nightly build](https://docs.openvino.ai/2024/get-started/install-openvino.html?VERSION=NIGHTLY) of OpenVINO or build
OpenVINO Tokenizers from a release branch if you have issues with the build process.

#### Using OpenVINO archive

Install [OpenVINO archive](https://docs.openvino.ai/2024/get-started/install-openvino.html) distribution. Use `--no-deps` to avoid OpenVINO installation from PyPI into your current environment.
`--extra-index-url` is needed to resolve build dependencies only.

```bash
source path/to/installed/openvino/setupvars.sh
git clone https://github.com/openvinotoolkit/openvino_tokenizers.git
cd openvino_tokenizers
pip install --no-deps . --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```
This command is the equivalent of minimal installation. Install tokenizers conversion dependencies if needed:
```bash
pip install transformers[sentencepiece] tiktoken
```
:warning: Latest commit of OpenVINO Tokenizers might rely on features that are not present in the release OpenVINO version.
Use [a nightly build](https://docs.openvino.ai/2024/get-started/install-openvino.html?VERSION=NIGHTLY) of OpenVINO or build
OpenVINO Tokenizers from a release branch if you have issues with the build process.

### Build and install for development

#### Using OpenVINO PyPI package

```bash
git clone https://github.com/openvinotoolkit/openvino_tokenizers.git
cd openvino_tokenizers
pip install -e .[all] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
# verify installation by running tests
cd tests/
pytest .
```

#### Using OpenVINO archive

Install [OpenVINO archive](https://docs.openvino.ai/2024/get-started/install-openvino.html) distribution. Use `--no-deps` to avoid OpenVINO installation from PyPI into your current environment.
`--extra-index-url` is needed to resolve build dependencies only.

```bash
source path/to/installed/openvino/setupvars.sh
git clone https://github.com/openvinotoolkit/openvino_tokenizers.git
cd openvino_tokenizers
pip install -e .[all] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
# verify installation by running tests
cd tests/
pytest .
```

### C++ Installation

You can use converted tokenizers in C++ pipelines with prebuild binaries.

1. Download OpenVINO archive distribution for your OS from [here](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) and extract the archive.
2. Download OpenVINO Tokenizers prebuild libraries from [here](https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/). To ensure compatibility first three numbers of OpenVINO Tokenizers version should match OpenVINO version and OS.
3. Extract OpenVINO Tokenizers archive into OpenVINO installation directory. OpenVINO Tokenizers archive maintains the structure to be aligned with OpenVINO archive:
    - Windows: `<openvino_dir>\runtime\bin\intel64\Release\`
    - MacOS_x86: `<openvino_dir>/runtime/lib/intel64/Release`
    - MacOS_arm64: `<openvino_dir>/runtime/lib/arm64/Release/`
    - Linux_x86: `<openvino_dir>/runtime/lib/intel64/`
    - Linux_arm64: `<openvino_dir>/runtime/lib/aarch64/`

After that you can add binary extension in the code with:
- `core.add_extension("openvino_tokenizers.dll")` for Windows
- `core.add_extension("libopenvino_tokenizers.dylib")` for MacOS
- `core.add_extension("libopenvino_tokenizers.so")` for Linux

and `read`/`compile` converted (de)tokenizers models.
If you use version `2023.3.0.0`, the binary extension file is called `(lib)user_ov_extension.(dll/dylib/so)`.

### Reducing the ICU Data Size

By default, all available ICU locales are supported, which significantly increases the package size. To reduce the size of the ICU libraries included in your final package, follow these steps:

1. **Use the ICU Data Configuration File**:
    - This file specifies which features and locales to include in a custom data bundle. You can find more information [here](https://unicode-org.github.io/icu/userguide/icu_data/buildtool.html#icu-data-configuration-file).

2. **Set the ICU Data Filter File as an Environment Variable**:
    - **On Unix-like systems (Linux, macOS)**:
      Set the `ICU_DATA_FILTER_FILE` environment variable to the path of your configuration file (`filters.json`):

      ```bash
      export ICU_DATA_FILTER_FILE="filters.json"
      ```

    - **On Windows**:
      Set the `ICU_DATA_FILTER_FILE` environment variable using the Command Prompt or PowerShell:

      **Command Prompt:**
      ```cmd
      set ICU_DATA_FILTER_FILE=filters.json
      ```

      **PowerShell:**
      ```powershell
      $env:ICU_DATA_FILTER_FILE="filters.json"
      ```

3. **Create a Configuration File**:
    - An example configuration file (`filters.json`) might look like this:

    ```json
    {
      "localeFilter": {
        "filterType": "language",
        "includelist": [
          "en"
        ]
      }
    }
    ```

4. **Configure OpenVINO Tokenizers**:
    - When building OpenVINO tokenizers, set the following CMake option during the project configuration:

    ```bash
    -DBUILD_FAST_TOKENIZERS=ON
    ```
   - Example for a pip installation path:
   ```bash
   ICU_DATA_FILTER_FILE=</path/to/filters.json> pip install git+https://github.com/openvinotoolkit/openvino_tokenizers.git --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly --config-settings=override=cmake.options.BUILD_FAST_TOKENIZERS=ON
   ```

By following these instructions, you can effectively reduce the size of the ICU libraries in your final package.

### Build OpenVINO Tokenizers without FastTokenizer Library

If a tokenizer doesn't use `CaseFold`, `UnicodeNormalization` or `Wordpiece` operations, you can drastically reduce package binary size by building OpenVINO Tokenizers without FastTokenizer dependency with this flag:

```bash
-DENABLE_FAST_TOKENIZERS=OFF
```

This option can also help with building for platform that is supported by FastTokenizer, for example `Android x86_64`.

Example for a pip installation path:
```bash

pip install git+https://github.com/openvinotoolkit/openvino_tokenizers.git --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly --config-settings=override=cmake.options.ENABLE_FAST_TOKENIZERS=OFF
```

## Usage

:warning: OpenVINO Tokenizers can be inferred on a `CPU` device only.

### Convert HuggingFace tokenizer

OpenVINO Tokenizers ships with CLI tool that can convert tokenizers from Huggingface Hub
or Huggingface tokenizers saved on disk:

```shell
convert_tokenizer codellama/CodeLlama-7b-hf --with-detokenizer -o output_dir
```

There is also `convert_tokenizer` function that can convert tokenizer python object.

```python
import numpy as np
from transformers import AutoTokenizer
from openvino import compile_model, save_model
from openvino_tokenizers import convert_tokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ov_tokenizer = convert_tokenizer(hf_tokenizer)

compiled_tokenzier = compile_model(ov_tokenizer)
text_input = ["Test string"]

hf_output = hf_tokenizer(text_input, return_tensors="np")
ov_output = compiled_tokenzier(text_input)

for output_name in hf_output:
    print(f"OpenVINO {output_name} = {ov_output[output_name]}")
    print(f"HuggingFace {output_name} = {hf_output[output_name]}")
# OpenVINO input_ids = [[ 101 3231 5164  102]]
# HuggingFace input_ids = [[ 101 3231 5164  102]]
# OpenVINO token_type_ids = [[0 0 0 0]]
# HuggingFace token_type_ids = [[0 0 0 0]]
# OpenVINO attention_mask = [[1 1 1 1]]
# HuggingFace attention_mask = [[1 1 1 1]]

# save tokenizer for later use
save_model(ov_tokenizer, "openvino_tokenizer.xml")

loaded_tokenizer = compile_model("openvino_tokenizer.xml")
loaded_ov_output = loaded_tokenizer(text_input)
for output_name in hf_output:
    assert np.all(loaded_ov_output[output_name] == ov_output[output_name])
```

### Connect Tokenizer to a Model

To infer and convert the original model, install torch or torch-cpu to the virtual environment.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openvino import compile_model, convert_model
from openvino_tokenizers import convert_tokenizer, connect_models

checkpoint = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
hf_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

text_input = ["Free money!!!"]
hf_input = hf_tokenizer(text_input, return_tensors="pt")
hf_output = hf_model(**hf_input)

ov_tokenizer = convert_tokenizer(hf_tokenizer)
ov_model = convert_model(hf_model, example_input=hf_input.data)
combined_model = connect_models(ov_tokenizer, ov_model)
compiled_combined_model = compile_model(combined_model)

openvino_output = compiled_combined_model(text_input)

print(f"OpenVINO logits: {openvino_output['logits']}")
# OpenVINO logits: [[ 1.2007061 -1.4698029]]
print(f"HuggingFace logits {hf_output.logits}")
# HuggingFace logits tensor([[ 1.2007, -1.4698]], grad_fn=<AddmmBackward0>)
```

### Use Extension With Converted (De)Tokenizer or Model With (De)Tokenizer

Import `openvino_tokenizers` will add all tokenizer-related operations to OpenVINO,
after which you can work with saved tokenizers and detokenizers.

```python
import numpy as np
import openvino_tokenizers
from openvino import Core

core = Core()

# detokenizer from codellama sentencepiece model
compiled_detokenizer = core.compile_model("detokenizer.xml")

token_ids = np.random.randint(100, 1000, size=(3, 5))
openvino_output = compiled_detokenizer(token_ids)

print(openvino_output["string_output"])
# ['sc�ouition�', 'intvenord hasient', 'g shouldwer M more']
```

### Text generation pipeline

```python
import numpy as np
from openvino import compile_model, convert_model
from openvino_tokenizers import add_greedy_decoding, convert_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


model_checkpoint = "JackFram/llama-68m"
hf_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
hf_model = AutoModelForCausalLM.from_pretrained(model_checkpoint, use_cache=False)

# convert hf tokenizer
text_input = ["Quick brown fox jumped "]
ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
compiled_tokenizer = compile_model(ov_tokenizer)

# transform input text into tokens
ov_input = compiled_tokenizer(text_input)
hf_input = hf_tokenizer(text_input, return_tensors="pt")

# convert Pytorch model to OpenVINO IR and add greedy decoding pipeline to it
ov_model = convert_model(hf_model, example_input=hf_input.data)
ov_model_with_greedy_decoding = add_greedy_decoding(ov_model)
compiled_model = compile_model(ov_model_with_greedy_decoding)

# generate new tokens
new_tokens_size = 10
prompt_size = ov_input["input_ids"].shape[-1]
input_dict = {
    output.any_name: np.hstack([tensor, np.zeros(shape=(1, new_tokens_size), dtype=np.int_)])
    for output, tensor in ov_input.items()
}
for idx in range(prompt_size, prompt_size + new_tokens_size):
    output = compiled_model(input_dict)["token_ids"]
    input_dict["input_ids"][:, idx] = output[:, idx - 1]
    input_dict["attention_mask"][:, idx] = 1
ov_token_ids = input_dict["input_ids"]

hf_token_ids = hf_model.generate(
    **hf_input,
    min_new_tokens=new_tokens_size,
    max_new_tokens=new_tokens_size,
    temperature=0,  # greedy decoding
)

# decode model output
compiled_detokenizer = compile_model(ov_detokenizer)
ov_output = compiled_detokenizer(ov_token_ids)["string_output"]
hf_output = hf_tokenizer.batch_decode(hf_token_ids, skip_special_tokens=True)
print(f"OpenVINO output string: `{ov_output}`")
# OpenVINO output string: `['Quick brown fox was walking through the forest. He was looking for something']`
print(f"HuggingFace output string: `{hf_output}`")
# HuggingFace output string: `['Quick brown fox was walking through the forest. He was looking for something']`
```

### TensorFlow Text Integration

OpenVINO Tokenizers include converters for certain TensorFlow Text operations.
Currently, only the MUSE model is supported.
Here is an example of model conversion and inference:

```python
import numpy as np
import tensorflow_hub as hub
import tensorflow_text  # register tf text ops
from openvino import convert_model, compile_model
import openvino_tokenizers  # register ov tokenizer ops and translators


sentences = ["dog",  "I cuccioli sono carini.", "私は犬と一緒にビーチを散歩するのが好きです"]
tf_embed = hub.load(
    "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/"
    "TensorFlow2/variations/multilingual/versions/2"
)
# convert model that uses Sentencepiece tokenizer op from TF Text
ov_model = convert_model(tf_embed)
ov_embed = compile_model(ov_model, "CPU")

ov_result = ov_embed(sentences)[ov_embed.output()]
tf_result = tf_embed(sentences)

assert np.all(np.isclose(ov_result, tf_result, atol=1e-4))
```

### RWKV Tokenizer

```python
from urllib.request import urlopen

from openvino import compile_model
from openvino_tokenizers import build_rwkv_tokenizer


rwkv_vocab_url = (
    "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/tokenizer/rwkv_vocab_v20230424.txt"
)

with urlopen(rwkv_vocab_url) as vocab_file:
    vocab = map(bytes.decode, vocab_file)
    tokenizer, detokenizer = build_rwkv_tokenizer(vocab)

tokenizer, detokenizer = compile_model(tokenizer), compile_model(detokenizer)

print(tokenized := tokenizer(["Test string"])["input_ids"])  # [[24235 47429]]
print(detokenizer(tokenized)["string_output"])  # ['Test string']
```

## Supported Tokenizer Types

| Huggingface <br/>Tokenizer Type | Tokenizer Model Type | Tokenizer | Detokenizer |
|---------------------------------|----------------------|----------|------------|
| Fast                            | WordPiece            | ✅        | ❌          |
|                                 | BPE                  | ✅        | ✅          |
|                                 | Unigram              | ❌         | ❌          |
| Legacy                          | SentencePiece .model | ✅        | ✅          |
| Custom                          | tiktoken             | ✅        | ✅          |
| RWKV                            | Trie                 | ✅        | ✅          |

## Test Results

This report is autogenerated and includes tokenizers and detokenizers tests. The `Output Matched, %` column shows the percent of test strings for which the results of OpenVINO and Huggingface Tokenizers are the same. To update the report run `pytest --update_readme tokenizers_test.py` in `tests` directory.

### Output Match by Tokenizer Type

<table>
  <thead>
    <tr>
      <th >Tokenizer Type</th>
      <th >Output Matched, %</th>
      <th >Number of Tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >BPE</td>
      <td >95.57</td>
      <td >5932</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >88.23</td>
      <td >6534</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >99.19</td>
      <td >494</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >99.10</td>
      <td >1327</td>
    </tr>
  </tbody>
</table>

### Output Match by Model

<table>
  <thead>
    <tr>
      <th >Tokenizer Type</th>
      <th >Model</th>
      <th >Output Matched, %</th>
      <th >Number of Tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-j-6b</td>
      <td >95.29</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-neo-125m</td>
      <td >95.29</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-neox-20b</td>
      <td >95.82</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/pythia-12b-deduped</td>
      <td >95.82</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >KoboldAI/fairseq-dense-13B</td>
      <td >96.65</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >NousResearch/Meta-Llama-3-8B-Instruct</td>
      <td >100.00</td>
      <td >241</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >Salesforce/codegen-16B-multi</td>
      <td >96.08</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >Xenova/gpt-4o</td>
      <td >100.00</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >ai-forever/rugpt3large_based_on_gpt2</td>
      <td >94.51</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >bigscience/bloom</td>
      <td >97.49</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >databricks/dolly-v2-3b</td>
      <td >95.82</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >deepseek-ai/deepseek-coder-6.7b-instruct</td>
      <td >100.00</td>
      <td >257</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/bart-large-mnli</td>
      <td >95.29</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/galactica-120b</td>
      <td >95.82</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/opt-66b</td>
      <td >96.65</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >gpt2</td>
      <td >95.29</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >laion/CLIP-ViT-bigG-14-laion2B-39B-b160k</td>
      <td >75.29</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >microsoft/deberta-base</td>
      <td >96.65</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >roberta-base</td>
      <td >95.29</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >sentence-transformers/all-roberta-large-v1</td>
      <td >95.29</td>
      <td >255</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >stabilityai/stablecode-completion-alpha-3b-4k</td>
      <td >95.82</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >stabilityai/stablelm-2-1_6b</td>
      <td >100.00</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >stabilityai/stablelm-tuned-alpha-7b</td>
      <td >95.82</td>
      <td >239</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >tiiuae/falcon-7b</td>
      <td >94.51</td>
      <td >255</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >NousResearch/Llama-2-13b-hf</td>
      <td >96.65</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >NousResearch/Llama-2-13b-hf_legacy</td>
      <td >100.00</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >NousResearch/Llama-2-13b-hf_sp_backend</td>
      <td >100.00</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm2-6b_legacy</td>
      <td >100.00</td>
      <td >153</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm3-6b_legacy</td>
      <td >50.97</td>
      <td >155</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >camembert-base</td>
      <td >52.30</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >camembert-base_legacy</td>
      <td >76.15</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >codellama/CodeLlama-7b-hf</td>
      <td >96.65</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >codellama/CodeLlama-7b-hf_legacy</td>
      <td >96.65</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >codellama/CodeLlama-7b-hf_sp_backend</td>
      <td >94.98</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >facebook/musicgen-small</td>
      <td >84.52</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >facebook/musicgen-small_legacy</td>
      <td >79.92</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/Phi-3-mini-128k-instruct</td>
      <td >95.85</td>
      <td >241</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/Phi-3-mini-128k-instruct_legacy</td>
      <td >95.85</td>
      <td >241</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/Phi-3-mini-128k-instruct_sp_backend</td>
      <td >94.19</td>
      <td >241</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/deberta-v3-base</td>
      <td >96.65</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/deberta-v3-base_legacy</td>
      <td >100.00</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >mlx-community/quantized-gemma-7b-it</td>
      <td >99.17</td>
      <td >241</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >mlx-community/quantized-gemma-7b-it_legacy</td>
      <td >99.17</td>
      <td >241</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >mlx-community/quantized-gemma-7b-it_sp_backend</td>
      <td >100.00</td>
      <td >241</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >rinna/bilingual-gpt-neox-4b</td>
      <td >80.75</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >rinna/bilingual-gpt-neox-4b_legacy</td>
      <td >86.61</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >t5-base</td>
      <td >85.77</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >t5-base_legacy</td>
      <td >81.17</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlm-roberta-base</td>
      <td >96.23</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlm-roberta-base_legacy</td>
      <td >96.23</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlnet-base-cased</td>
      <td >65.27</td>
      <td >239</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlnet-base-cased_legacy</td>
      <td >59.41</td>
      <td >239</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >Qwen/Qwen-14B-Chat</td>
      <td >100.00</td>
      <td >255</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >THUDM/glm-4-9b</td>
      <td >98.33</td>
      <td >239</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >ProsusAI/finbert</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >bert-base-multilingual-cased</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >bert-base-uncased</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >cointegrated/rubert-tiny2</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >distilbert-base-uncased-finetuned-sst-2-english</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >google/electra-base-discriminator</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >google/mobilebert-uncased</td>
      <td >100.00</td>
      <td >91</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >jhgan/ko-sbert-sts</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >prajjwal1/bert-mini</td>
      <td >100.00</td>
      <td >91</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >rajiv003/ernie-finetuned-qqp</td>
      <td >100.00</td>
      <td >91</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >rasa/LaBSE</td>
      <td >88.79</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >sentence-transformers/all-MiniLM-L6-v2</td>
      <td >100.00</td>
      <td >107</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >squeezebert/squeezebert-uncased</td>
      <td >100.00</td>
      <td >91</td>
    </tr>
  </tbody>
</table>

### Recreating Tokenizers From Tests

In some tokenizers, you need to select certain settings so that their output is closer to the Huggingface tokenizers:
- `THUDM/chatglm2-6b` detokenizer always skips special tokens. Use `skip_special_tokens=True` during conversion
- `THUDM/chatglm3-6b` detokenizer don't skips special tokens. Use `skip_special_tokens=False` during conversion
- All tested tiktoken based detokenizers leave extra spaces. Use `clean_up_tokenization_spaces=False` during conversion
