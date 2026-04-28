# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Tests for custom TF op translators added in openvino_tokenizers:
#   - AsString  (translate_as_string)
#
# Each test builds a tiny TF SavedModel containing the op under test,
# converts it with openvino using the tokenizers extension, executes
# it on the OV CPU device, and compares the result to TensorFlow's own
# output.

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import openvino as ov

# -----------------------------------------------------------------------
# Locate the tokenizers extension .so built in the local tree
# -----------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_LOCAL_SO = _REPO_ROOT / "build" / "build_local" / "src" / "libopenvino_tokenizers.so"

# Fall back: accept any libopenvino_tokenizers.so we can find below the repo
def _find_extension() -> str | None:
    if _BUILD_LOCAL_SO.exists():
        return str(_BUILD_LOCAL_SO)
    for candidate in sorted(_REPO_ROOT.rglob("libopenvino_tokenizers.so")):
        return str(candidate)
    return None

_EXTENSION_PATH = _find_extension()

pytestmark = pytest.mark.skipif(
    _EXTENSION_PATH is None,
    reason="libopenvino_tokenizers.so not found – build the project first",
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _convert_and_run(saved_model_dir: str, inputs: dict) -> dict:
    """Convert a TF SavedModel with the tokenizers extension and run it.

    ``inputs`` is a name→value dict matching the TF function's argument names.
    The function looks up OV input names and re-orders the tensors accordingly
    so the call is robust to OV reordering TF arguments.
    """
    model = ov.convert_model(saved_model_dir, extension=_EXTENSION_PATH)
    core = ov.Core()
    core.add_extension(_EXTENSION_PATH)
    compiled = core.compile_model(model, "CPU")

    # Build an {ov_input_index: value} dict so we match by name, not position.
    ov_input_map = {}
    for ov_inp in compiled.inputs:
        # OV uses the TF argument name as the tensor name
        ov_name = ov_inp.get_any_name()
        if ov_name in inputs:
            ov_input_map[ov_inp] = inputs[ov_name]

    # Fall back to positional if no names matched (e.g. single-input models)
    if not ov_input_map:
        ov_input_map = list(inputs.values())

    results = compiled(ov_input_map)
    return results


# -----------------------------------------------------------------------
# AsString tests
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def as_string_int64_model(tmp_path_factory):
    """SavedModel: f(x: int64) -> AsString(x)"""
    tf = pytest.importorskip("tensorflow")

    tmp_dir = str(tmp_path_factory.mktemp("as_string_i64"))

    class AsStringI64(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64)])
        def __call__(self, x):
            return tf.strings.as_string(x)

    tf.saved_model.save(AsStringI64(), tmp_dir)
    return tmp_dir


@pytest.fixture(scope="module")
def as_string_float32_model(tmp_path_factory):
    """SavedModel: f(x: float32) -> AsString(x)"""
    tf = pytest.importorskip("tensorflow")

    tmp_dir = str(tmp_path_factory.mktemp("as_string_f32"))

    class AsStringF32(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def __call__(self, x):
            return tf.strings.as_string(x)

    tf.saved_model.save(AsStringF32(), tmp_dir)
    return tmp_dir


@pytest.mark.parametrize("values,dtype", [
    ([0, 1, -1, 42, 9999, -12345], np.int64),
    ([1, 2, 3], np.int64),
])
def test_as_string_int64(as_string_int64_model, values, dtype):
    tf = pytest.importorskip("tensorflow")
    x = np.array(values, dtype=dtype)
    tf_out = tf.strings.as_string(tf.constant(x)).numpy()

    ov_results = _convert_and_run(as_string_int64_model, {"x": x})
    ov_out = ov_results[0]

    assert ov_out.shape == tf_out.shape, f"Shape mismatch: {ov_out.shape} vs {tf_out.shape}"
    for ov_val, tf_val in zip(ov_out.flatten(), tf_out.flatten()):
        assert ov_val == tf_val.decode(), f"Value mismatch: OV={ov_val!r}  TF={tf_val!r}"


@pytest.mark.parametrize("values", [
    [1.0, -2.5, 0.0, 3.14159],
])
def test_as_string_float32(as_string_float32_model, values):
    tf = pytest.importorskip("tensorflow")
    x = np.array(values, dtype=np.float32)
    # AsString for floats: OV and TF may format floats differently,
    # so we compare parsed float values with a tolerance rather than
    # exact string equality.
    tf_out = tf.strings.as_string(tf.constant(x)).numpy()

    ov_results = _convert_and_run(as_string_float32_model, {"x": x})
    ov_out = ov_results[0]

    assert ov_out.shape == tf_out.shape
    for ov_val, tf_val in zip(ov_out.flatten(), tf_out.flatten()):
        ov_f = float(ov_val)
        tf_f = float(tf_val.decode())
        assert abs(ov_f - tf_f) < 1e-5, f"Float mismatch: OV={ov_val!r}  TF={tf_val!r}"


# -----------------------------------------------------------------------
# NumericToString unit-test (evaluate() without TF)
# -----------------------------------------------------------------------

def test_numeric_to_string_evaluate():
    """Test NumericToString custom op evaluate() directly without TF."""
    import openvino.runtime.op as rt_op
    from openvino import op as ov_op, Type, PartialShape, Model, Tensor

    # Build a tiny model: Constant(int64) -> NumericToString
    # We can't import NumericToString from Python directly; instead we use
    # convert_model on a small TF graph that exercises the same code path.
    # (If the Python binding for NumericToString ever exists, update here.)
    tf = pytest.importorskip("tensorflow")
    import tempfile, os

    values = [0, 1, -1, 42, -9999]
    x = tf.constant(values, dtype=tf.int64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        class M(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64)])
            def __call__(self, x):
                return tf.strings.as_string(x)
        tf.saved_model.save(M(), tmp_dir)

        model = ov.convert_model(tmp_dir, extension=_EXTENSION_PATH)

    core = ov.Core()
    core.add_extension(_EXTENSION_PATH)
    compiled = core.compile_model(model, "CPU")
    result = compiled([np.array(values, dtype=np.int64)])[0]

    expected = [str(v) for v in values]
    for got, exp in zip(result.flatten(), expected):
        assert got == exp, f"NumericToString: got {got!r}, expected {exp!r}"
