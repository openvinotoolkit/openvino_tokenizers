#!/usr/bin/env bash
set -euo pipefail

cd /home/epavel/devel/openvino_tokenizers
source /home/epavel/opt/envs/py310-tokenizers/bin/activate

export OV_TOKENIZER_PREBUILD_EXTENSION_PATH=/home/epavel/devel/openvino_tokenizers/build-Release/src/libopenvino_tokenizers.so
export PYTHONPATH=/home/epavel/opt/openvino_master/python:/home/epavel/devel/openvino_tokenizers/python/:
export LD_LIBRARY_PATH=/home/epavel/opt/openvino_master/runtime/3rdparty/tbb/lib:/home/epavel/opt/openvino_master/runtime/lib/intel64:

python -m openvino_tokenizers check Lightricks/LTX-Video --subfolder tokenizer --trust-remote-code 2>&1
