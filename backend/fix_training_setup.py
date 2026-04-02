from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"


def mirror_healthy_labels() -> None:
    pairs = [
        (DATASET / "labels" / "train", DATASET / "labels" / "train" / "healthy"),
        (DATASET / "labels" / "val", DATASET / "labels" / "val" / "healthy"),
    ]

    for source_dir, healthy_dir in pairs:
        healthy_dir.mkdir(parents=True, exist_ok=True)

        for label_file in source_dir.glob("*.txt"):
            target = healthy_dir / label_file.name
            shutil.move(str(label_file), str(target))
            print(f"Moved {label_file.relative_to(ROOT)} -> {target.relative_to(ROOT)}")


def deploy_best_model(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Model file not found: {source}")

    target = ROOT / "best.pt"
    shutil.copy2(source, target)
    print(f"Copied {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fix YOLO dataset layout and optionally copy a trained best.pt into backend."
    )
    parser.add_argument(
        "--fix-labels",
        action="store_true",
        help="Move healthy label files into labels/train/healthy and labels/val/healthy.",
    )
    parser.add_argument(
        "--deploy",
        type=Path,
        help="Copy the chosen weights file into backend/best.pt.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.fix_labels and not args.deploy:
        parser.error("Use --fix-labels and/or --deploy.")

    if args.fix_labels:
        mirror_healthy_labels()

    if args.deploy:
        source = args.deploy
        if not source.is_absolute():
            source = (ROOT / source).resolve()
        deploy_best_model(source)


if __name__ == "__main__":
    main()
