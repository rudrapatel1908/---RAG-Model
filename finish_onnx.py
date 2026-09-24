import os
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ["HF_HOME"] = "D:/.hf_cache"
out_dir = Path("backend/models/e5-small-onnx")

# 1. Download the last remaining configuration file
print("Downloading missing config.json...")
hf_hub_download(
    repo_id="Xenova/multilingual-e5-small", 
    filename="config.json", 
    local_dir=out_dir
)

# 2. Flatten the ONNX directory structure if needed 
# (Moves model.onnx out of /onnx/ directly into /e5-small-onnx/)
nested_onnx = out_dir / "onnx" / "model.onnx"
target_onnx = out_dir / "model.onnx"

if nested_onnx.exists():
    if target_onnx.exists():
        os.remove(target_onnx)
    nested_onnx.rename(target_onnx)
    # Clean up the empty nested folder
    try:
        os.rmdir(out_dir / "onnx")
    except Exception:
        pass

print("\nCleanup complete! Verifying files...")
for f in out_dir.iterdir():
    if f.is_file():
        print(f"- {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
