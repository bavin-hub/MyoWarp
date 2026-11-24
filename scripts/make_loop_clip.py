"""Extract a steady gait cycle and create a long walking or running reference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=ROOT /
                        "assets/musclemimic/motions/MyoFullBody/gmr/KIT/314/walking_medium10_poses.npz")
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--start", type=int, help="Cycle start frame; use together with --length")
    parser.add_argument("--length", type=int, help="Cycle length in frames")
    parser.add_argument("--min-cycle", type=float, default=0.8, help="Minimum cycle duration in seconds")
    parser.add_argument("--max-cycle", type=float, default=1.6, help="Maximum cycle duration in seconds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.clip.is_file():
        parser.error(f"Source clip missing: {args.clip}. Download it with scripts/fetch_musclemimic.py "
                     "--clip KIT/314/walking_medium10_poses")
    if args.output is None:
        args.output = ROOT / "assets/musclemimic/motions/loops" / f"{args.clip.stem}_{args.seconds:g}s.npz"
    if args.clip.resolve() == args.output.resolve():
        parser.error("Output must differ from the source clip")
    from myowarp.musclemimic.model import build_model
    from myowarp.musclemimic.reference import loop_clip

    model, _ = build_model()
    try:
        path = loop_clip(args.clip, args.output, model, args.seconds, args.start, args.length,
                         t_min=args.min_cycle, t_max=args.max_cycle)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved reference: {path}")
    print("Evaluate the entire reference with:")
    print(f".venv-musclemimic/bin/python scripts/eval_musclemimic.py --clip {path} --steps 100000")


if __name__ == "__main__":
    main()
