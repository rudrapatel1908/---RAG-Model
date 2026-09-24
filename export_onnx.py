from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "intfloat/multilingual-e5-small"
out_dir = Path("backend/models/e5-small-onnx")
out_dir.mkdir(parents=True, exist_ok=True)

model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("Exported to", out_dir)
