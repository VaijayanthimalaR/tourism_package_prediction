from huggingface_hub import HfApi
import os

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_package_prediction/deployment",    
    # replace with your repoid
    repo_id="vaijayanthimala07/tourism-package-predict",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
