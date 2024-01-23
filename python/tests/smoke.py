import numpy
import openvino.runtime.opset8 as ov
import openvino_extension
from openvino.runtime.utils.node_factory import NodeFactory

def test_extension_load():
    factory = NodeFactory()
    # factory.add_extension(library_path)
    data = ov.parameter([1, 2], dtype=numpy.float32)
    identity = factory.create("Identity", data.outputs())

    del identity