import json
from typing import Iterable, Tuple, List

import numpy as np
import openvino as ov
from openvino import PartialShape, Type
from openvino import opset15 as opset
from openvino.runtime.passes import AnyInput, Manager, Matcher, ModelPass, WrapType
from openvino.utils.types import make_constant_node

from . import _get_factory
from .constants import PROCESSED_POST_PROCESSOR_NAME

# TODO:
# check opsets15
# add comments
# add input names
# Check why inputs differ when we pass [[string]] from the single input [string] and why ++ sample fails
# Check __call__ method in python, utils dipatch
# exit with warning, exit when postprocessing is not present
# add coments for matchers

[4], []

class ModifyCombineSegmentsForPairInput(ModelPass):
    """
    Changes the ov.Model so that it accepts paired input.

    Concatenate both input, process, tokenize and then split them near the end before CombineSegments node.
    If any constant SpecialToken depends on sequence inputs thay are zeroed.
    Truncation logic for the inputs is also modified to support max_length. By default, if final length
    exceeds max_length, it's truncated to max_length/2 for each input and then combined.
    """
    
    def run_on_model(self, model: ov.Model):
        parameters = model.get_parameters()
        if len(parameters) != 1:
            return False

        target_inputs = list(parameters[0].output(0).get_target_inputs())
        str_unpack: ov.Node = target_inputs[0].get_node()
        if str_unpack.get_type_name() != "StringTensorUnpack":
            return False

        # Replace single Parameter with concatenated pair of Parameters.
        new_parameters = [ov.op.Parameter(Type.string, PartialShape([-1])) for i in range(2)]
        new_input = opset.concat(new_parameters, axis=0)
        str_unpack.input(0).replace_source_output(new_input.output(0))

        model.replace_parameter(0, new_parameters[0])
        model.add_parameters([new_parameters[1]])

        combine_seg = None
        for op in model.get_ops():
            if op.get_type_name() == "CombineSegments":
                combine_seg = op
        if not combine_seg:
            return False

        num_segments = int(combine_seg.get_input_size() - 1) // 3

        inputs = []
        input_signature = [[]] * num_segments

        post_processor = json.loads(model.get_rt_info(PROCESSED_POST_PROCESSOR_NAME).value)
        
        for i in range(num_segments):
            if isinstance(combine_seg.input_value(3 * i).node, ov.op.Constant):
                # Constant input
                if not isinstance(combine_seg.input_value(3 * i + 2).node, ov.op.Constant):
                    return False
                input_signature[i] = combine_seg.input_value(3 * i + 2).node.get_data().item(0)
                inputs.extend(
                    [
                        combine_seg.input_value(3 * i),
                        combine_seg.input_value(3 * i + 1),
                        combine_seg.input_value(3 * i + 2),
                    ]
                )
            else:
                # Sequence input
                input_signature[i] = -1

                trunc_node = combine_seg.input_value(3 * i).node

                begin = trunc_node.input_value(0)
                end = trunc_node.input_value(1)
                data = trunc_node.input_value(2)
                inputs.extend([begin, end, data])
                trunc_values = trunc_node.input_values()[3:]

        assert post_processor["single"]["ids"] == input_signature, (
            "Input signature from rt_info does not match to the CombineSegments node inputs."
        )
        assert post_processor["pair"]["ids"][:len(input_signature)] == input_signature, (
            "Paried inputs are allowed only when it's widening the single input"
        )
        assert len(np.nonzero(np.array(post_processor["pair"]["ids"]) <= -1)[0]) == 2, (
            "Only 2 inputs are allowed for the paired input"
        )
        assert len(np.nonzero(np.array(post_processor["single"]["ids"]) <= -1)[0]) == 1, (
            "Incorrect input signature, extending only singe input is supported"
        )

        param_1_shape = opset.shape_of(new_parameters[0], output_type="i32")
        param_2_shape = opset.shape_of(new_parameters[1], output_type="i32")
        final_size = opset.shape_of(begin, output_type="i32")

        # For the first input begins_1/ends_1, it's a slice till the Parameter_1 shape.
        begins_1 = opset.slice(begin, start=[0], stop=param_1_shape, step=[1], name='begins_1')
        ends_1 = opset.slice(end, start=[0], stop=param_1_shape, step=[1], name='ends_1')
        # For the second input begins_2/ends_2, slice till the end.
        begins_2 = opset.slice(begin, start=param_1_shape, stop=final_size, step=[1], name='begins_2')
        ends_2 = opset.slice(end, start=param_1_shape, stop=final_size, step=[1], name='ends_2')
        # data is left unchanged

        # If inputs_2 is empty, we need to zero the second dimension of the broadcasted begins and ends tensors.
        # TODO: indeed for ends it should've been 1 but for thix bug CSV-160624 we set to zero for the moment.
        equal_node = opset.equal(param_2_shape, make_constant_node([0], Type.i32), name="is_paired_input")
        begins_2 = opset.select(equal_node, make_constant_node([0], Type.i32), begins_2)
        ends_2 = opset.select(equal_node, make_constant_node([0], Type.i32), ends_2)

        first_input = [begins_1.output(0), ends_1.output(0), data]
        second_input = [begins_2.output(0), ends_2.output(0), data]

        first_input_idx = input_signature.index(-1)
        signature_to_extend = post_processor["pair"]["ids"][len(input_signature):]
        
        # Since we add additional special tokens the max_length for truncation should be reduced.
        trunc_values[0] = make_constant_node(trunc_values[0].node.data - (len(signature_to_extend) - 1), Type.i32).output(0)

        trunc = _get_factory().create("Truncate", [*first_input, *second_input, *trunc_values])
        first_input, second_input = trunc.outputs()[:3], trunc.outputs()[3:6]

        new_inputs = inputs.copy()
        # Replace original input with concatenated Parameter_1, Parameter_2 with only Parameter_1 input.
        # [bos, concat(sequence_1, sequnce_2), eos] -> [bos, sequence_1, eos]
        new_inputs[3 * first_input_idx : 3 * first_input_idx + 3] = first_input

        #  if original input_signature [bos, sequence_1, eos]
        #  if pair_signature [bos, sequence_1, eos, sequence_2, eos]
        # then singature_to_extend = [sequence_2, eos]
        for value in signature_to_extend:
            if value <= -1:
                # we ensured previous that only one additional input is possible
                new_inputs.extend(second_input)
                continue

            if not isinstance(value, Iterable):
                value = [value]
            assert all(map(lambda x: isinstance(x, (int)), value)), (
                "input signature ids should be value or a list of values"
            )

            added_spec_begins = make_constant_node(0, Type.i32).output(0)
            added_spec_ends = make_constant_node(len(value), Type.i32).output(0)
            added_spec_data = make_constant_node(value, Type.i32).output(0)

            # If ends for the sequence_2 is nullified, we should nullify special_tokens constant as well
            added_spec_ends = opset.multiply(
                added_spec_ends,
                opset.select(equal_node, make_constant_node(0, Type.i32), make_constant_node(1, Type.i32)),
            ).output(0)
            new_spec_tokens = [added_spec_begins, added_spec_ends, added_spec_data]

            new_inputs.extend(new_spec_tokens)

        # For the added inputs segment ids should be 1.
        new_segment_ids = make_constant_node(post_processor["pair"]["type_ids"], Type.i32).output(0)
        new_inputs.extend([new_segment_ids])

        new_combine_segments = _get_factory().create("CombineSegments", new_inputs)
        ov.utils.replace_node(combine_seg, new_combine_segments)

        return True


def add_second_input(model: ov.Model):
    """
    Extends inplace the input of the model to a pair of inputs.
    """
    manager = Manager()
    manager.register_pass(ModifyCombineSegmentsForPairInput())
    manager.run_passes(model)
