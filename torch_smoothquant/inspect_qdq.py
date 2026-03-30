# inspect_qdq.py
import onnx
from collections import Counter

model = onnx.load("./exported_onnx/image_encoder_qdq.onnx")
ops = Counter(node.op_type for node in model.graph.node)

print("Top ops:")
for k, v in ops.most_common(20):
    print(f"{k}: {v}")

print("\nQDQ nodes:")
print("QuantizeLinear:", ops.get("QuantizeLinear", 0))
print("DequantizeLinear:", ops.get("DequantizeLinear", 0))