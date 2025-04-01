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
    
    def add_truncation(
            self, combine_seg: ov.Node, 
            sequence_input_idx: int, 
            first_input: List[ov.Output], 
            second_input: List[ov.Output],
            singature_to_extend: List[int]
        ) -> Tuple[List[ov.Output], List[ov.Output]]:
        size_input = WrapType("opset1::Subtract", [AnyInput(), AnyInput()])
        min_pattern = WrapType("opset1::Minimum", [size_input, WrapType("opset1::Constant")])
        sub_trunc_matcher = Matcher(WrapType("opset15::Subtract", [AnyInput(), min_pattern]), "SubTruncationMatcher")
        add_trunc_matcher = Matcher(WrapType("opset15::Add", [AnyInput(), min_pattern]), "AddTruncationMatcher")

        begin = combine_seg.input_value(3 * sequence_input_idx)
        end = combine_seg.input_value(3 * sequence_input_idx + 1)
        is_left_trunc = sub_trunc_matcher.match(begin)
        is_right_trunc = add_trunc_matcher.match(end)
        if not (is_left_trunc ^ is_right_trunc):
            return first_input, second_input
        
        # To prevent side effects.
        first_input = first_input.copy()
        second_input = second_input.copy()

        # Get original begins, ends before truncation.
        # if is_right_trunc:
        #     first_input[1] = add_trunc_matcher.get_pattern_value_map()[size_input].node.input_value(1)
        # else:
        #     first_input[0] = sub_trunc_matcher.get_pattern_value_map()[size_input].node.input_value(0)


        # Get the value of const from pattern CombindeSegmen's begin/end <-- Add/Sub <-- Minimum <-- Constant
        # By matcher we ensured that graph is correct and we can safely get the value.
        max_length_const = (begin if is_left_trunc else end).node.input_value(1).node.input_value(1).node
        
        # Number of added tokens in paired inputs. 
        num_added = np.nonzero(np.array(singature_to_extend) > 0)[0]
        max_length_const = make_constant_node(max_length_const.data - num_added, Type.i32)

        # If max length is odd, we prefer to add remaining 1 to the first half.
        half_max_length = opset.divide(max_length_const, make_constant_node(2, Type.i32))
        half_plus_mod = opset.add(half_max_length, opset.mod(max_length_const, make_constant_node(2, Type.i32)))

        # Get original begins and ends before truncation
        #  of const from pattern CombindeSegmen's begin/end <-- Add/Sub <-- Minimum <-- Constant
        first_len = first_input[1].node - first_input[0].node
        second_len = second_input[1].node - second_input[0].node
        # Whether the sum of the inputs lengths is greater than max_length.
        is_less_max_length = opset.less_equal(opset.add(first_len, second_len), max_length_const)

        # Whether the first and second inputs are greater than half_max_length
        first_less_half = opset.less_equal(first_len, half_plus_mod)
        second_less_half = opset.less_equal(second_len, half_max_length)

        # assume max_lengths = 10
        # There are 4 cases:
        # 1. sum is < max_length

        # 2. If first_len, second_len = (8, 3) then resulting should be (7, 3).
        # We need to shorten first and leave second untouched 
        # first_len = max_length - second_len
        # second_len = second_len
        
        # 3. If first_len, second_len = (3, 8) then resulting should be (3, 7).
        # We need to shorten second and leave first untouched
        # first_len = first_len
        # second_len = max_length - first_len
        
        # 4. If first_len, second_len = (6, 60) then resulting should be (5, 5).
        # shorten both 
        # first_len = max_length/2 + max_length % 2
        # second_len = max_length - first_len

        zero = make_constant_node(0, Type.i32)
        if is_right_trunc:
            # At the end we should have a single output with ends for the first and second inputs

            # True if and only if second length is less than half_max_length and sum of lengths is greater than max_length
            # shorten_first = opset.logical_and(opset.logical_not(is_less_max_length), second_less_half)
            # shortened_first_end = opset.select(shorten_first, max_length_const - second_len, zero)
            # original_second_end = opset.select(opset.logical_or(second_less_half, is_less_max_length), second_input[1], zero)

            # True if and only if first length is less than half_max_length and sum of lengths is greater than max_length
            shorten_second = opset.logical_and(opset.logical_not(is_less_max_length), first_less_half)
            original_first_end = opset.select(opset.logical_or(first_less_half, is_less_max_length), first_input[1], zero)
            shortened_second_end = opset.select(shorten_second, max_length_const - first_len + second_input[0].node, zero)

            # True if and only if both first and second lengths are greater than half_max_length
            # cut_both = opset.logical_and(opset.logical_not(first_less_half), opset.logical_not(second_less_half))
            # cut_first_end = opset.select(cut_both, first_input[0].node + half_plus_mod, zero)
            # cut_second_end = opset.select(cut_both, second_input[0].node + half_max_length, zero)

            # first_end = original_first_end + shortened_first_end + cut_first_end
            # second_end = original_second_end + shortened_second_end + cut_second_end
            first_end = original_first_end
            second_end = shortened_second_end + make_constant_node(0, Type.i32)
            first_input[1] = first_end.output(0)
            second_input[1] = second_end.output(0)
        else:
            pass
        
        return first_input, second_input

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
            else:
                # Sequence input
                input_signature[i] = -1

                size_input = WrapType("opset1::Subtract", [AnyInput(), AnyInput()])
                min_pattern = WrapType("opset1::Minimum", [size_input, WrapType("opset1::Constant")])
                sub_trunc_matcher = Matcher(WrapType("opset15::Subtract", [AnyInput(), min_pattern]), "SubTruncationMatcher")
                add_trunc_matcher = Matcher(WrapType("opset15::Add", [AnyInput(), min_pattern]), "AddTruncationMatcher")


                begin = combine_seg.input_value(3 * i)
                end = combine_seg.input_value(3 * i + 1)
                data = combine_seg.input_value(3 * i + 2)

                is_left_trunc = sub_trunc_matcher.match(begin)
                is_right_trunc = add_trunc_matcher.match(end)
                # Get original begins, ends before truncation.
                if is_right_trunc:
                    end = add_trunc_matcher.get_pattern_value_map()[size_input].node.input_value(0)
                else:
                    begin = sub_trunc_matcher.get_pattern_value_map()[size_input].node.input_value(0)

            inputs.extend(
                [
                    combine_seg.input_value(3 * i),
                    combine_seg.input_value(3 * i + 1),
                    combine_seg.input_value(3 * i + 2),
                ]
            )

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
        singature_to_extend = post_processor["pair"]["ids"][len(input_signature):]
        first_input, second_input = self.add_truncation(combine_seg, first_input_idx, first_input, second_input, singature_to_extend)

        new_inputs = inputs.copy()
        # Replace original input with concatenated Parameter_1, Parameter_2 with only Parameter_1 input.
        # [bos, concat(sequence_1, sequnce_2), eos] -> [bos, sequence_1, eos]
        new_inputs[3 * first_input_idx : 3 * first_input_idx + 3] = first_input

        #  if original input_signature [bos, sequence_1, eos]
        #  if pair_signature [bos, sequence_1, eos, sequence_2, eos]
        # then singature_to_extend = [sequence_2, eos]
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
