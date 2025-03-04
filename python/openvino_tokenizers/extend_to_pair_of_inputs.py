from typing import Iterable, Tuple, Optional, Set

from openvino import Model, PartialShape, Type, op
from openvino import op
import openvino as ov
from openvino import opset15 as opset
from openvino.utils.types import make_constant_node
import numpy as np
from . import _get_factory
from openvino import PartialShape
from openvino.runtime import opset13 as ops
from openvino.runtime.passes import Matcher, WrapType, Or, AnyInput, MatcherPass, Manager, VisualizeTree


# input [num_batched, 1-2]
# broadcast to [num_batched, 2]
# multiply begin, ends to [1, 1] if second input was 2, and to [1, 0] it it was 1.
# reshape to [num_batched * 1 (or 2)] so that the subsequent graph remains the same

# reshape both begins, ends back to [num_batches, 1-2]
# split botch begins, ends
# select for ends zero if input shape was [num_batces, 1]
# if some constant SpecialToken depends on Sequence inputs which are zerod, zero that constant ends as well
# extend truncation
# connect to CombineSegments

class AddReshapeForPairInput(MatcherPass):
    
    def __init__(self, model: ov.Model, number_of_inputs: int = 2):
        MatcherPass.__init__(self)

        str_input_pattern = WrapType("opset1::Parameter")
        str_unpack_pattern = WrapType("opset15.StringTensorUnpack", [str_input_pattern])

        def callback(matcher: Matcher) -> bool:
            str_unpack: ov.Node = matcher.get_match_root()

            new_parameter = op.Parameter(Type.string, PartialShape([-1, -1]))
            str_unpack.input(0).replace_source_output(new_parameter.output(0))
            nonlocal model
            model.replace_parameter(0, new_parameter)

            inp_shape = opset.shape_of(new_parameter, output_type='i32')
            num_batches = opset.slice(inp_shape, [0], [1], [1])
            boradcasted_shape = opset.concat([num_batches, make_constant_node([number_of_inputs], Type.i32)], 
                                             axis=0,
                                             name="broadcasted_shape")

            # Save for future use to prevent loops
            target_begins = str_unpack.output(0).get_target_inputs()
            target_ends = str_unpack.output(1).get_target_inputs()

            begins = opset.broadcast(str_unpack.output(0), boradcasted_shape)
            ends = opset.broadcast(str_unpack.output(1), boradcasted_shape)

            num_inputs_node = opset.slice(inp_shape, [1], [2], [1])
            equal = opset.equal(num_inputs_node, make_constant_node(1, Type.i32), name='is_paired_input')

            # If the number of inputs is 1, we need to zero the second dimension 
            # of the broadcasted begins and ends tensors.
            # This is done so that the second input string is empty "".
            multiplier = opset.select(equal, make_constant_node([[1, 0]], Type.i32), make_constant_node([[1, 1]], Type.i32))
            begins = opset.multiply(begins, multiplier)
            ends = opset.multiply(ends, multiplier)

            begins_ = opset.reshape(begins, make_constant_node([-1], Type.i32), special_zero=False)
            ends_ = opset.reshape(ends, make_constant_node([-1], Type.i32), special_zero=False)

            # replace begins, ends with new nodes
            for input in target_begins:
                input.replace_source_output(begins_.output(0))
            for input in target_ends:
                input.replace_source_output(ends_.output(0))
            
            # TODO: check whether we should return false in orde to not run validate_and_infer types
            return True

        self.register_matcher(Matcher(str_unpack_pattern, "AddReshapeForPairInput"), callback)


class ModifyCombineSegmentsForPairInput(MatcherPass):
    def __init__(self, model: ov.Model, max_length: int = 18, number_of_inputs: int = 2):
        MatcherPass.__init__(self)
        combine_seg_pattern = AnyInput(lambda output: output.node.get_type_name() == "CombineSegments")
        
        is_max_len = lambda x: isinstance(x.node.get_data(), np.ndarray) and x.node.get_data().item(0) == max_length
        const_pattern = WrapType("opset1::Constant", is_max_len)
        min_pattern = WrapType("opset1::Minimum", [AnyInput(), const_pattern])        
        
        sub_trunc_pattern = WrapType("opset15::Subtract", [AnyInput(), min_pattern])
        sub_trunc_matcher = Matcher(sub_trunc_pattern, "SubTruncationMatcher")
        
        sub_trunc_pattern = WrapType("opset15::Add", [AnyInput(), min_pattern])        
        add_trunc_matcher = Matcher(sub_trunc_pattern, "AddTruncationMatcher")
        
        def callback(matcher: Matcher) -> bool:
            nonlocal model, sub_trunc_matcher, add_trunc_matcher

            broadcasted_shape = None
            equual_node = None
            for node in model.get_ops():
                if node.get_friendly_name() == "broadcasted_shape":
                    broadcasted_shape = node
                if node.get_friendly_name() == "is_paired_input":
                    equual_node = node
            assert broadcasted_shape is not None
            assert equual_node is not None

            combine_seg: ov.Node = matcher.get_match_root()
            num_segments = int(combine_seg.get_input_size() - 1) // 3
            
            num_sequences = 0
            inputs = []
            for i in range(num_segments):
                if isinstance(combine_seg.input_value(3*i).node, ov.op.Constant):
                    # Constant input
                    pass
                else:
                    # Sequence input
                    num_sequences += 1

                    begin = combine_seg.input_value(3*i)
                    end = combine_seg.input_value(3*i + 1)
                    data = combine_seg.input_value(3*i + 2)

                    is_left_trunc = sub_trunc_matcher.match(combine_seg.input_value(3*i))
                    is_right_trunc = add_trunc_matcher.match(combine_seg.input_value(3*i + 1))
                
                inputs.extend([
                    combine_seg.input_value(3*i),
                    combine_seg.input_value(3*i + 1),
                    combine_seg.input_value(3*i + 2)
                ])

            assert num_sequences == 1
            
            reshape_1 = opset.reshape(begin, broadcasted_shape, special_zero=False)
            reshape_2 = opset.reshape(end, broadcasted_shape, special_zero=False)
            
            # split begins, ends
            split_1 = opset.split(reshape_1, axis=1, num_splits=number_of_inputs)
            split_2 = opset.split(reshape_2, axis=1, num_splits=number_of_inputs)
            
            # TODO: take them from post_processor_dict
            depends_on_input = 1

            added_spec_begins = make_constant_node(0, Type.i32).output(0)
            added_spec_ends = make_constant_node(1, Type.i32).output(0)

            spec_token_value = inputs[(depends_on_input + 1) * 3 + 2].node.get_data().item(0)
            added_spec_data = make_constant_node([spec_token_value], Type.i32).output(0)

            eq = opset.equal(inputs[depends_on_input * 3 + 1], make_constant_node([0], Type.i32))
            # if all elements of ends is zero
            added_spec_ends = opset.multiply(added_spec_ends, opset.select(eq, make_constant_node([0], Type.i32), make_constant_node([1], Type.i32))).output(0)


            # If inputs_2 is empty, we need to zero the second dimension of the broadcasted begins and ends tensors.
            # TODO: indeed for ends it should've been 1 but for thix bug CSV-xxxxx we zeto zero for the moment. 
            begins_2 = opset.select(equual_node, make_constant_node([0], Type.i32), opset.squeeze(split_1.output(1), [1])).output(0)
            ends_2 = opset.select(equual_node, make_constant_node([0], Type.i32), opset.squeeze(split_2.output(1), [1])).output(0)
            
            new_segment_ids = make_constant_node([0 for i in range(num_segments + 2)], Type.i32).output(0)

            first_input = [opset.squeeze(split_1.output(0), [1]).output(0), opset.squeeze(split_2.output(0), [1]).output(0), data]
            second_input = [begins_2, ends_2, data]
            

            if is_left_trunc or is_right_trunc:
                # This is a common part for both truncations
                max_length_const = (begin if is_left_trunc else end).node.input_value(1).node.input_value(1)
                # max_length_const = combine_seg.input_value(3*i).node.input_value(1).node.input_value(1)
                half_max_length = opset.divide(max_length_const, make_constant_node(2, Type.i32))
                half_plus_mod = opset.add(half_max_length, opset.mod(max_length_const, make_constant_node(2, Type.i32)))
                
                gt = opset.greater(opset.add(first_input[1].node - first_input[0].node, 
                                             second_input[1].node - second_input[0].node), 
                                             max_length_const)
            
            if is_left_trunc:
                first_input[0] = opset.select(gt, opset.subtract(first_input[1], half_plus_mod), first_input[0]).output(0)
                second_input[0] = opset.select(gt, opset.subtract(second_input[1], half_max_length), second_input[0]).output(0)
            elif is_right_trunc:
                first_input[1] = opset.select(gt, opset.add(first_input[0], half_plus_mod), first_input[1]).output(0)
                second_input[1] = opset.select(gt, opset.add(second_input[0], half_max_length), second_input[1]).output(0)

            new_inputs = [
                # add special token at BOS
                inputs[0], inputs[1], inputs[2],
                # First Sequence input
                *first_input,
                # Add special token at EOS
                inputs[6], inputs[7], inputs[8], 
                # Second Sequence input
                *second_input,
                # add special tokens at EOS
                added_spec_begins, added_spec_ends, added_spec_data,
                # segment ids
                new_segment_ids
            ]
            
            new_combine_segments = _get_factory().create("CombineSegments", new_inputs)
            ov.utils.replace_node(combine_seg, new_combine_segments)
            
            # # TODO: copy runtime info
            return True

        self.register_matcher(Matcher(combine_seg_pattern, "ModifyCombineSegmentsForPairInput"), callback)


def extend_input_to_pair(model: ov.Model, max_length: Optional[int] = None) -> ov.Model:

    manager = Manager()
    manager.register_pass(AddReshapeForPairInput(model))
    manager.register_pass(ModifyCombineSegmentsForPairInput(model))
    # manager.register_pass(VisualizeTree("after.svg"))
    manager.run_passes(model)
    return model
