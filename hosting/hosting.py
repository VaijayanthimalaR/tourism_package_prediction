from huggingface_hub import HfApi
import os

os.environ["HF_TOKEN"] = "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"
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id="vaijayanthimala07/tourism-package-predict",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
