import os
from pathlib import Path

# Redirect the Hugging Face cache to your D: drive to bypass C: space limits
os.environ["HF_HOME"] = "D:/.hf_cache"

from huggingface_hub import hf_hub_download

out_dir = Path("backend/models/e5-small-onnx")
out_dir.mkdir(parents=True, exist_ok=True)

repo_id = "Xenova/multilingual-e5-small"

files = [
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json"
]

print("Downloading pre-converted ONNX model components using D: drive cache...")
for f in files:
    print(f"Downloading {f}...")
    # Force direct download into local directory without symlinks
    local_path = hf_hub_download(
        repo_id=repo_id, 
        filename=f, 
        local_dir=out_dir, 
        local_dir_use_symlinks=False
    )

print("Successfully downloaded all assets to:", out_dir)
