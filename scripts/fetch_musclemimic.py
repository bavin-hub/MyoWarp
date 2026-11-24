"""Download the pretrained actor and selected, already-retargeted motion clips."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", help="Repeat for multiple KIT/AMASS clip names")
    parser.add_argument("--checkpoint-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    folder = ROOT / "assets/musclemimic"
    checkpoint = snapshot_download(
        "amathislab/mm-10m-2", local_dir=folder / "checkpoints/mm-10m-2",
        ignore_patterns=["*.mp4", "*.gif"],
    )
    print(f"Checkpoint: {checkpoint}")
    if args.checkpoint_only:
        return
    for clip in args.clip or ["KIT/314/walking_medium09_poses"]:
        name = f"MyoFullBody/gmr/{clip.removesuffix('.npz')}.npz"
        path = hf_hub_download(
            "amathislab/musclemimic-retargeted", name,
            repo_type="dataset", local_dir=folder / "motions",
        )
        print(f"Motion: {path}")


if __name__ == "__main__":
    main()
