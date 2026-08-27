import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper

inp, out, gemm_name = sys.argv[1], sys.argv[2], sys.argv[3]

model = onnx.load(inp)
g = model.graph

gemm = None
gemm_idx = None

for i, n in enumerate(g.node):
    if n.name == gemm_name:
        gemm = n
        gemm_idx = i
        break

if gemm is None:
    raise RuntimeError(f"node not found: {gemm_name}")

if gemm.op_type not in ["Gemm", "MatMul"]:
    raise RuntimeError(f"{gemm_name} is {gemm.op_type}, not Gemm/MatMul")

old_in = gemm.input[0]
new_in = old_in + "_reshape_2d"

shape_name = new_in + "_shape"
shape = np.array([1, 1024], dtype=np.int64)
g.initializer.append(numpy_helper.from_array(shape, shape_name))

reshape = helper.make_node(
    "Reshape",
    inputs=[old_in, shape_name],
    outputs=[new_in],
    name=gemm_name + "_input_reshape_2d",
)

gemm.input[0] = new_in

nodes = list(g.node)
nodes.insert(gemm_idx, reshape)

del g.node[:]
g.node.extend(nodes)

onnx.checker.check_model(model)
onnx.save(model, out)

print("saved:", out)
print("inserted reshape:", old_in, "->", new_in, "[1,1024]")