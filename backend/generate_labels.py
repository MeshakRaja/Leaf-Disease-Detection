from pathlib import Path


DATASET_ROOT = Path("dataset")
CLASS_MAP = {
    "healthy": 0,
    "bacterial spot": 1,
    "early blight": 2,
    "late blight": 3,
    "spider mites": 4,
}


def generate_labels(split: str, class_name: str, class_id: int) -> None:
    image_dir = DATASET_ROOT / "images" / split / class_name
    label_dir = DATASET_ROOT / "labels" / split / class_name
    label_dir.mkdir(parents=True, exist_ok=True)

    label_content = f"{class_id} 0.5 0.5 1 1"

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        label_path.write_text(label_content, encoding="utf-8")
        print(f"Created: {label_path}")


def main() -> None:
    for split in ("train", "val"):
        for class_name, class_id in CLASS_MAP.items():
            generate_labels(split, class_name, class_id)

    print("Done!")


if __name__ == "__main__":
    main()
