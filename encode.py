import json
import sys
import onnx
from onnx import shape_inference

PASS_THROUGH_OPS = {
    "Reshape",
    "Transpose",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Identity",
    "Cast",
}

STOP_OPS = {
    "Add",
    "Softmax",
}

def build_output_to_node(graph):
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            if out:
                output_to_node[out] = node
    return output_to_node

def trace_v_side_to_add(start_tensor, output_to_node, initializer_names):
    stack = [start_tensor]
    visited_tensors = set()
    visited_nodes = set()
    a8_tensors = set()

    while stack:
        tensor_name = stack.pop()
        if not tensor_name or tensor_name in visited_tensors:
            continue
        visited_tensors.add(tensor_name)

        if tensor_name in initializer_names:
            continue

        a8_tensors.add(tensor_name)

        producer = output_to_node.get(tensor_name)
        if producer is None:
            continue

        node_id = id(producer)
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)

        if producer.op_type in PASS_THROUGH_OPS:
            for inp in producer.input:
                if inp and inp not in initializer_names:
                    stack.append(inp)
        elif producer.op_type == "Add":
            continue
        elif producer.op_type == "Softmax":
            a8_tensors.discard(tensor_name)
            continue
        else:
            continue

    return a8_tensors

def choose_v_input(node, output_to_node, initializer_names):
    candidate_inputs = []

    for inp in node.input:
        if not inp or inp in initializer_names:
            continue
        producer = output_to_node.get(inp)
        producer_op = producer.op_type if producer is not None else None
        candidate_inputs.append((inp, producer_op))

    res_inp = None
    upstream_is_softmax = False

    for inp, producer_op in candidate_inputs:
        if producer_op == "Softmax":
            upstream_is_softmax = True
    
    if upstream_is_softmax:
        for inp, producer_op in candidate_inputs:
            if producer_op != "Softmax":
                res_inp = inp
                break
    else:
        # Choose second input (activation named as "transpose_xx")
        if len(candidate_inputs) > 1:
            res_inp = candidate_inputs[1][0]
        elif len(candidate_inputs) == 1:
            res_inp = candidate_inputs[0][0]
        else:
            res_inp = None

    # if candidate_inputs:
    #     return candidate_inputs[0][0]

    return res_inp

def build_w8a16_attention_vside_u8(onnx_path, output_json):
    model = onnx.load(onnx_path)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    initializer_names = {x.name for x in graph.initializer}
    output_to_node = build_output_to_node(graph)

    activation_names = set()
    for x in graph.input:
        if x.name not in initializer_names:
            activation_names.add(x.name)
    for x in graph.value_info:
        activation_names.add(x.name)
    for x in graph.output:
        activation_names.add(x.name)

    a8_activation_tensors = set()

    for node in graph.node:
        if "matmul" in node.name: # TODO: have to use more robust way to identify whether act * act matmul
            v_input = choose_v_input(node, output_to_node, initializer_names)
            if v_input is None:
                continue

            print(v_input)
            a8_activation_tensors |= trace_v_side_to_add(
                v_input,
                output_to_node,
                initializer_names,
            )

    activation_encodings = []
    for name in sorted(activation_names):
        if name in a8_activation_tensors:
            activation_encodings.append({
                "name": name,
                "enc_type": "PER_TENSOR",
                "dtype": "INT",
                "bw": 8,
                "is_sym": False,
                "scale": [0.03125],
                "offset": [0]
            })
        else:
            activation_encodings.append({
                "name": name,
                "enc_type": "PER_TENSOR",
                "dtype": "INT",
                "bw": 16,
                "is_sym": False,
                "scale": [0.125],
                "offset": [0]
            })

    param_encodings = []
    for init in graph.initializer:
        name_lower = init.name.lower()
        is_bias = (
            name_lower.endswith("bias")
            or name_lower.endswith(".bias")
            or "bias" in name_lower
        )

        if is_bias:
            param_encodings.append({
                "name": init.name,
                "enc_type": "PER_TENSOR",
                "dtype": "INT",
                "bw": 16,
                "is_sym": True,
                "scale": [0.125],
                "offset": [0]
            })
        else:
            param_encodings.append({
                "name": init.name,
                "enc_type": "PER_TENSOR",
                "dtype": "INT",
                "bw": 8,
                "is_sym": True,
                "scale": [0.03125],
                "offset": [0]
            })

    encodings = {
        "version": "1.0.0",
        "activation_encodings": activation_encodings,
        "param_encodings": param_encodings,
        "quantizer_args": {
            "activation_bitwidth": 16,
            "dtype": "INT",
            "is_symmetric": True,
            "param_bitwidth": 8,
            "per_channel_quantization": False,
            "quant_scheme": "post_training_tf"
        },
        "excluded_layers": []
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(encodings, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input.onnx output.encodings")
        sys.exit(1)

    build_w8a16_attention_vside_u8(sys.argv[1], sys.argv[2])