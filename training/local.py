from huggingface_hub import snapshot_download
from pathlib import Path

# Create a local directory for the model
local_dir = Path("./models/Dream-v0-Base-7B")
local_dir.mkdir(parents=True, exist_ok=True)

print("Downloading model to local project folder...")
snapshot_download(
    repo_id="Dream-org/Dream-v0-Base-7B",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print(f"Model saved to {local_dir}")