from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
from pathlib import Path
from huggingface_hub import upload_folder
import os


repo_id = "vaijayanthimala07/tourism-package-predict"

repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

    # Use relative paths from the repo root
data_dir = Path(__file__).parent.parent / "data"  # Adjust based on actual structure

if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found at {data_dir}")

api.upload_folder(
    folder_path=str(data_dir),
    repo_id=repo_id,
    repo_type=repo_type,
    token=os.environ.get("HF_TOKEN")
)
