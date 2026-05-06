"""
Download LivePortrait checkpoints on Windows (no bash required).

The official repo provides a `scripts/download_models.sh` shell script. This
is the equivalent in pure Python, using the official HuggingFace mirror that
the LivePortrait authors publish. Run from the face-service directory:

    python download_models.py

Files land in `third_party/LivePortrait/pretrained_weights/` to match what
the engine expects. Total ~2 GB.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Enable Rust-based parallel transfer (5-10x faster) before importing hub.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit(
        "huggingface_hub is required. Install dependencies first:\n"
        "    pip install -r requirements.txt"
    )

REPO_ID = "KwaiVGI/LivePortrait"
TARGET = Path(__file__).resolve().parent / "third_party" / "LivePortrait" / "pretrained_weights"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} → {TARGET}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(TARGET),
        local_dir_use_symlinks=False,
        max_workers=16,
        # Skip example assets to keep the download lean. Comment out to
        # fetch everything if you want the demo videos too.
        ignore_patterns=["*.mp4", "*.gif", "assets/*", "examples/*"],
    )
    print("Done.")


if __name__ == "__main__":
    main()
