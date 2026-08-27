# add_tiny_bias_min.py
import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper

inp, out, target_name = sys.argv[1], sys.argv[2], sys.argv[3]

model = onnx.load(inp)
g = model.graph

target = None
target_idx = None
for i, n in enumerate(g.node):
    if n.name == target_name:
        target = n
        target_idx = i
        break

if target is None:
    raise RuntimeError(f"node not found: {target_name}")
if target.op_type != "MatMul":
    raise RuntimeError(f"{target_name} is {target.op_type}, not MatMul")

init_map = {x.name: x for x in g.initializer}

out_features = None
for x in target.input:
    if x in init_map:
        w = numpy_helper.to_array(init_map[x])
        out_features = w.shape[-1]
        print("weight:", x, w.shape)
        break

if out_features is None:
    raise RuntimeError("cannot infer out_features; weight is not initializer")

old_out = target.output[0]
raw_out = old_out + "_raw"

target.output[0] = raw_out

bias_name = old_out + "_tiny_bias"

# 用 tiny non-zero，避免被 constant folding / identity elimination 移除
bias = np.full((out_features,), 1e-8, dtype=np.float32)
g.initializer.append(numpy_helper.from_array(bias, bias_name))

add = helper.make_node(
    "Add",
    [raw_out, bias_name],
    [old_out],
    name=target_name + "_tiny_bias_add",
)

nodes = list(g.node)
nodes.insert(target_idx + 1, add)
del g.node[:]
g.node.extend(nodes)

onnx.checker.check_model(model)
onnx.save(model, out)

print("saved:", out)
print("bias shape:", bias.shape)
print("bias value:", bias[0])