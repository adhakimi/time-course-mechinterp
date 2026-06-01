"""Download checkpoints of allenai/OLMo-7B-0424-hf from HuggingFace.

OLMo revisions are named like `stepN-tokensXB` (e.g. `step651581-tokens2731B`).
By default, every branch is downloaded; set `STEP_INCREMENT` below to subsample
(e.g. one checkpoint every 5000 steps).
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "allenai/OLMo-7B-0424-hf"
DATA_ROOT = os.environ["DATA_ROOT"]  # set in paths.env; `source paths.env` first
DOWNLOAD_DIR = os.path.join(DATA_ROOT, "revisions_temp")

# Optional subsampling. Set to None to download every revision; otherwise keep
# `main` plus revisions whose step is a multiple of STEP_INCREMENT.
STEP_INCREMENT = None
MAX_WORKERS = 16

STEP_PATTERN = re.compile(r"^step(\d+)-tokens\d+B$")


def select_branches(branches):
    if STEP_INCREMENT is None:
        return [b.name for b in branches]

    selected = []
    for branch in branches:
        if branch.name == "main":
            selected.append(branch.name)
            continue
        m = STEP_PATTERN.match(branch.name)
        if m and int(m.group(1)) % STEP_INCREMENT == 0:
            selected.append(branch.name)
    return selected


def download_revision(name):
    print(f"Downloading {name}...")
    snapshot_download(repo_id=REPO_ID, revision=name, cache_dir=DOWNLOAD_DIR)
    return name


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    api = HfApi()
    refs = api.list_repo_refs(REPO_ID)
    branches_to_download = select_branches(refs.branches)
    print(f"Selected {len(branches_to_download)} revisions to download.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_revision, name) for name in branches_to_download]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    print(f"Completed: {result}")
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    print(f"Total revisions processed: {len(branches_to_download)}")


if __name__ == "__main__":
    main()
