"""
One-time script: int8-quantizes the fp32 ONNX embedding model to cut
its memory/disk footprint ~4x (118M params: 470MB fp32 -> ~120MB int8).
Run once locally, commit the quantized output, and point
hybrid_search.py + Dockerfile at the new file. Original fp32 file is
left untouched on disk (not deleted) in case a quality regression
needs to be checked against it.
"""
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_DIR = Path(__file__).resolve().parent / "backend" / "models" / "e5-small-onnx"
SRC = MODEL_DIR / "model.onnx"
DST = MODEL_DIR / "model_quantized.onnx"

print(f"Quantizing {SRC} ({SRC.stat().st_size / 1e6:.1f} MB) -> {DST} ...")
quantize_dynamic(
    model_input=str(SRC),
    model_output=str(DST),
    weight_type=QuantType.QInt8,
)
print(f"Done. Quantized size: {DST.stat().st_size / 1e6:.1f} MB")
