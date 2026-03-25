import torch
from transformers import CLIPModel
import os
import torch.nn as nn

ONNX_DIR = "exported_onnx"

class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection
    
    def forward(self, x):
        output = self.text_model(x)
        x = output.pooler_output
        x = self.text_projection(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(model_name).to(device).eval()

    text_encoder = TextEncoder(model)
    text_encoder.eval()
    DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

    text_onnx_path = os.path.join(ONNX_DIR, "text_encoder.onnx")
    print(f"\nExporting Text Encoder to {text_onnx_path}...")

    torch.onnx.export(
        text_encoder,
        DUMMY_TEXT_INPUT,
        text_onnx_path,
        input_names=["text"],
        output_names=["text_embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )

if __name__ == "__main__":
    main()
