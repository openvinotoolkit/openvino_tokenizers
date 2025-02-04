# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools
import os
import site
import sys
from itertools import chain
from pathlib import Path
from typing import Callable, Optional

import openvino
from openvino.utils.node_factory import NodeFactory


_ext_name = "openvino_tokenizers"
if sys.platform == "win32":
    _ext_name = f"{_ext_name}.dll"
elif sys.platform == "darwin":
    _ext_name = f"lib{_ext_name}.dylib"
elif sys.platform == "linux":
    _ext_name = f"lib{_ext_name}.so"
else:
    sys.exit(f"Error: extension does not support the platform {sys.platform}")

# when the path to the extension set manually
_extension_path = os.environ.get("OV_TOKENIZER_PREBUILD_EXTENSION_PATH")
if _extension_path and Path(_extension_path).is_file():
    # when the path to the extension set manually
    _ext_path = Path(_extension_path)
else:
    site_packages = chain((Path(__file__).parent.parent,), site.getusersitepackages(), site.getsitepackages())
    _ext_path = next(
        (
            ext
            for site_package in map(Path, site_packages)
            if (ext := site_package / __name__ / "lib" / _ext_name).is_file()
        ),
        _ext_name,  # Case when the library can be found in the PATH/LD_LIBRAY_PATH
    )

del _ext_name

# patching openvino
old_core_init = openvino.Core.__init__
old_factory_init = openvino.utils.node_factory.NodeFactory.__init__
old_fe_init = openvino.frontend.frontend.FrontEnd.__init__


@functools.wraps(old_core_init)
def new_core_init(self, *args, **kwargs):
    old_core_init(self, *args, **kwargs)
    self.add_extension(str(_ext_path))  # Core.add_extension doesn't support Path object


@functools.wraps(old_factory_init)
def new_factory_init(self, *args, **kwargs):
    old_factory_init(self, *args, **kwargs)
    self.add_extension(_ext_path)


@functools.wraps(old_fe_init)
def new_fe_init(self, *args, **kwargs):
    old_fe_init(self, *args, **kwargs)
    self.add_extension(str(_ext_path))


openvino.Core.__init__ = new_core_init
openvino.frontend.frontend.FrontEnd.__init__ = new_fe_init


def _get_factory_callable() -> Callable[[], NodeFactory]:
    factory = {}

    def inner(opset_version: Optional[str] = None) -> NodeFactory:
        nonlocal factory
        if opset_version not in factory:
            openvino.utils.node_factory.NodeFactory.__init__ = new_factory_init
            factory[opset_version] = NodeFactory() if opset_version is None else NodeFactory(opset_version)

        return factory[opset_version]

    return inner


def _get_opset_factory_callable() -> Callable[[], NodeFactory]:
    # factory without extensions
    factory = {}

    def inner(opset_version: Optional[str] = None) -> NodeFactory:
        nonlocal factory
        if opset_version not in factory:
            openvino.runtime.utils.node_factory.NodeFactory.__init__ = old_factory_init
            factory[opset_version] = NodeFactory() if opset_version is None else NodeFactory(opset_version)

        return factory[opset_version]

    return inner


_get_factory = _get_factory_callable()
_get_opset_factory = _get_opset_factory_callable()

# some files uses _get_factory function
from .__version__ import __version__  # noqa
from .build_tokenizer import build_rwkv_tokenizer  # noqa
from .convert_tokenizer import convert_tokenizer  # noqa
from .utils import add_greedy_decoding, connect_models  # noqa
