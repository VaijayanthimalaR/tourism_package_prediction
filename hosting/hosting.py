from huggingface_hub import HfApi
from pathlib import Path
from huggingface_hub import upload_folder

import os

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))
   # Use relative paths from the repo root
data_dir = Path(__file__).parent.parent / "deployment"

if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found at {data_dir}")

api.upload_folder(
    folder_path=str(data_dir),  
    # replace with your repoid
    repo_id="vaijayanthimala07/tourism-package-predict",          # the target repo
     token=os.environ.get("HF_TOKEN"),
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
