import openvino as ov
import json
import numpy as np

from openvino import PartialShape, Type
from openvino import opset15 as opset
from openvino.runtime.passes import AnyInput, Manager, Matcher, MatcherPass, WrapType, ModelPass
from openvino.utils.types import make_constant_node
from typing import Iterable

from . import _get_factory
from .constants import PROCESSED_POST_PROCESSOR_NAME


class ModifyCombineSegmentsForPairInput(ModelPass):
    """
    Changes the ov.Model so that it accepts paired input.

    Concatenate both input, process, tokenize and then split them near the end before CombineSegments node.
    If any constant SpecialToken depends on sequence inputs thay are zeroed.
    Truncation logic for the inputs is also modified to support max_length. By default, if final length
    exceeds max_length, it's truncated to max_length/2 for each input and then combined.
    """

    def __init__(self):
        super().__init__()

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
        if num_segments not in [2, 3]:
            return False

        num_sequences = 0
        inputs = []
        input_signature = [[]] * num_segments

        const_pattern = WrapType("opset1::Constant")
        min_pattern = WrapType("opset1::Minimum", [AnyInput(), const_pattern])

        sub_trunc_pattern = WrapType("opset15::Subtract", [AnyInput(), min_pattern])
        sub_trunc_matcher = Matcher(sub_trunc_pattern, "SubTruncationMatcher")

        add_trunc_pattern = WrapType("opset15::Add", [AnyInput(), min_pattern])
        add_trunc_matcher = Matcher(add_trunc_pattern, "AddTruncationMatcher")
        post_processor = json.loads(model.get_rt_info(PROCESSED_POST_PROCESSOR_NAME).value)

        for i in range(num_segments):
            if isinstance(combine_seg.input_value(3 * i).node, ov.op.Constant):
                # Constant input
                if not isinstance(combine_seg.input_value(3 * i + 2).node, ov.op.Constant):
                    return False
                input_signature[i] = combine_seg.input_value(3 * i + 2).node.get_data().item(0)
            else:
                # Sequence input
                input_signature[i] = -1
                num_sequences += 1

                begin = combine_seg.input_value(3 * i)
                end = combine_seg.input_value(3 * i + 1)
                data = combine_seg.input_value(3 * i + 2)

                is_left_trunc = sub_trunc_matcher.match(combine_seg.input_value(3 * i))
                is_right_trunc = add_trunc_matcher.match(combine_seg.input_value(3 * i + 1))

            inputs.extend(
                [
                    combine_seg.input_value(3 * i),
                    combine_seg.input_value(3 * i + 1),
                    combine_seg.input_value(3 * i + 2),
                ]
            )

        assert num_sequences == 1, "Incorrect input signature, extending only singe input is supported"
        assert post_processor["single"]["ids"] == input_signature, (
            "Input signature from rt_info does not match to the CombineSegments node inputs."
        )
        assert post_processor["pair"]["ids"][: len(input_signature)] == input_signature, (
            "Paried inputs are allowed only when it's widening the single input"
        )
        assert len(np.nonzero(np.array(post_processor["pair"]["ids"]) <= -1)[0]) == 2, (
            "Only 2 inputs are allowed for the paired input"
        )

        param_1_shape = opset.shape_of(new_parameters[0], output_type="i32")
        param_2_shape = opset.shape_of(new_parameters[1], output_type="i32")
        final_size = opset.shape_of(begin, output_type="i32")

        # For the first input begins_1/ends_1, it's a slice till the Parameter_1 shape.
        begins_1 = opset.slice(begin, start=[0], stop=param_1_shape, step=[1])
        ends_1 = opset.slice(end, start=[0], stop=param_1_shape, step=[1])
        # For the second input begins_2/ends_2, slice till the end.
        begins_2 = opset.slice(begin, start=param_1_shape, stop=final_size, step=[1])
        ends_2 = opset.slice(end, start=param_1_shape, stop=final_size, step=[1])

        equal_node = opset.equal(param_2_shape, make_constant_node([0], Type.i32), name="is_paired_input")

        # If inputs_2 is empty, we need to zero the second dimension of the broadcasted begins and ends tensors.
        # TODO: indeed for ends it should've been 1 but for thix bug CSV-160624 we set to zero for the moment.
        begins_2 = opset.select(equal_node, make_constant_node([0], Type.i32), begins_2)
        ends_2 = opset.select(equal_node, make_constant_node([0], Type.i32), ends_2)

        first_input = [begins_1.output(0), ends_1.output(0), data]
        second_input = [begins_2.output(0), ends_2.output(0), data]

        # Add truncations. Default HF scheme.
        if is_left_trunc or is_right_trunc:
            # This is a common part for both truncations
            max_length_const = (begin if is_left_trunc else end).node.input_value(1).node.input_value(1)
            # max_length_const = combine_seg.input_value(3*i).node.input_value(1).node.input_value(1)
            half_max_length = opset.divide(max_length_const, make_constant_node(2, Type.i32))
            half_plus_mod = opset.add(half_max_length, opset.mod(max_length_const, make_constant_node(2, Type.i32)))

            gt = opset.greater(
                opset.add(first_input[1].node - first_input[0].node, second_input[1].node - second_input[0].node),
                max_length_const,
            )

        if is_left_trunc:
            first_input[0] = opset.select(gt, opset.subtract(first_input[1], half_plus_mod), first_input[0]).output(0)
            second_input[0] = opset.select(
                gt, opset.subtract(second_input[1], half_max_length), second_input[0]
            ).output(0)
        elif is_right_trunc:
            first_input[1] = opset.select(gt, opset.add(first_input[0], half_plus_mod), first_input[1]).output(0)
            second_input[1] = opset.select(gt, opset.add(second_input[0], half_max_length), second_input[1]).output(0)

        new_inputs = inputs.copy()
        first_input_idx = input_signature.index(-1)
        new_inputs[3 * first_input_idx : 3 * first_input_idx + 3] = first_input

        singature_to_extend = post_processor["pair"]["ids"][len(input_signature) :]
        for value in singature_to_extend:
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
