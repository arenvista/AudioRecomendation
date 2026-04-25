from pathlib import Path
import csv
from tinytag import TinyTag

DATA_DIR = Path("audio")
METADATA_PATH = DATA_DIR / "metadata.csv"


def generate_metadata(data_dir: Path = DATA_DIR, metadata_path: Path = METADATA_PATH) -> None:
    """Generate a CSV file mapping MP3 filenames to artist labels."""
    mp3_files = sorted(path for path in data_dir.iterdir() if path.suffix.lower() == ".mp3")

    if not mp3_files:
        print(f"No MP3 files found in {data_dir}.")
        return

    with metadata_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "label"])

        for file_path in mp3_files:
            try:
                tags = TinyTag.get(str(file_path))
                label = tags.artist or "unknown"
                writer.writerow([file_path.name, label])
                print(f"Mapped {file_path.name} -> {label}")
            except Exception as exc:
                print(f"Could not process {file_path.name}: {exc}")

    print(f"\nSuccess! metadata.csv has been saved to: {metadata_path}")


if __name__ == "__main__":
    generate_metadata()
