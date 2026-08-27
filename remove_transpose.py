import copy
import argparse
import onnx
import numpy as np
from onnx import numpy_helper, helper


def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            if a.type == onnx.AttributeProto.STRING:
                return a.s
    return default


def set_or_replace_int_attr(node, name, value):
    kept = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.extend([helper.make_attribute(name, value)])


def build_maps(model):
    producer = {}
    consumers = {}
    for node in model.graph.node:
        for o in node.output:
            producer[o] = node
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    return producer, consumers


def get_initializer_map(model):
    return {init.name: init for init in model.graph.initializer}


def transpose_initializer_(init):
    arr = numpy_helper.to_array(init)
    arr_t = arr.T.copy()
    new_init = numpy_helper.from_array(arr_t, name=init.name)
    init.CopyFrom(new_init)


def fold_weight_transpose_for_matmul(model: onnx.ModelProto):
    producer, consumers = build_maps(model)
    init_map = get_initializer_map(model)

    nodes_to_remove = []

    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue

        perm = get_attr(node, "perm", None)
        if perm != [1, 0]:
            continue

        trans_out = node.output[0]
        trans_in = node.input[0]

        trans_consumers = consumers.get(trans_out, [])
        if not trans_consumers:
            continue

        # only handle transpose used by MatMul
        if not all(c.op_type == "MatMul" for c in trans_consumers):
            continue

        # Case A: Transpose input is DequantizeLinear(weight)
        if trans_in in producer and producer[trans_in].op_type == "DequantizeLinear":
            dq = producer[trans_in]
            dq_x = dq.input[0]

            # axis fix for DQ
            dq_axis = get_attr(dq, "axis", None)

            # maybe Q(weight) -> DQ -> Transpose
            if dq_x in producer and producer[dq_x].op_type == "QuantizeLinear":
                q = producer[dq_x]
                q_x = q.input[0]

                if q_x in init_map:
                    transpose_initializer_(init_map[q_x])

                    # If per-channel on [out,in] axis=0, after transpose should be axis=1
                    if get_attr(q, "axis", None) == 0:
                        set_or_replace_int_attr(q, "axis", 1)
                    if dq_axis == 0:
                        set_or_replace_int_attr(dq, "axis", 1)

                    for c in trans_consumers:
                        for idx, inp in enumerate(c.input):
                            if inp == trans_out:
                                c.input[idx] = trans_in

                    nodes_to_remove.append(node)

            # direct initializer -> DQ -> Transpose
            elif dq_x in init_map:
                transpose_initializer_(init_map[dq_x])

                if dq_axis == 0:
                    set_or_replace_int_attr(dq, "axis", 1)

                for c in trans_consumers:
                    for idx, inp in enumerate(c.input):
                        if inp == trans_out:
                            c.input[idx] = trans_in

                nodes_to_remove.append(node)

        # Case B: Transpose input is direct initializer
        elif trans_in in init_map:
            transpose_initializer_(init_map[trans_in])

            for c in trans_consumers:
                for idx, inp in enumerate(c.input):
                    if inp == trans_out:
                        c.input[idx] = trans_in

            nodes_to_remove.append(node)

    if nodes_to_remove:
        kept = [n for n in model.graph.node if n not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(kept)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="aimet_mixed_quant_image.onnx")
    parser.add_argument("--output", default="aimet_mixed_quant_image_no_weight_transpose.onnx")
    parser.add_argument(
        "--infer-shapes",
        action="store_true",
        help="Run shape inference before saving. Disabled by default because large ONNX files can exceed protobuf limits.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip ONNX checker after saving.",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)
    model = fold_weight_transpose_for_matmul(model)

    del model.graph.value_info[:]
    if args.infer_shapes:
        model = onnx.shape_inference.infer_shapes(model)

    onnx.save(model, args.output, save_as_external_data=True)

    if not args.skip_check:
        onnx.checker.check_model(args.output, full_check=True)

    print(f"saved to {args.output}")
