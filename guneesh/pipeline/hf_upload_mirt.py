"""Upload the 70 MIRT fits (data/irt/mirt_<form>_K<K>/fold<f>/) to the dataset GunGG4/paiec-Qwen8Binstruct-guneesh-data.

upload_large_folder is resumable: rerunning it skips files already uploaded. Checkpoints are not uploaded.
Run from the project root:  python pipeline/hf_upload_mirt.py
"""
from huggingface_hub import HfApi

HfApi().upload_large_folder(
    repo_id="GunGG4/paiec-Qwen8Binstruct-guneesh-data", repo_type="dataset", folder_path="data",
    allow_patterns=["irt/mirt_*/fold*/*"], ignore_patterns=["**/ckpt_*", "**/*.part"], num_workers=2)
print("UPLOAD FINISHED", flush=True)
