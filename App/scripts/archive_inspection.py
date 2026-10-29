from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import re

ROOM_A_CLASSES = {"fall", "sit", "sitdown", "stand", "standup", "walk"}
ROOM_B_PREFIXES = {
    "empty": re.compile(r"^Empty(?:_Amp)?_.+\.csv$", re.I),
    "lying": re.compile(r"^Lying(?:_Amp)?_.+\.csv$", re.I),
    "sitting": re.compile(r"^Sitting(?:_Amp)?_.+\.csv$", re.I),
    "standing": re.compile(r"^Standing(?:_Amp)?_.+\.csv$", re.I),
    "walking": re.compile(r"^Walking(?:_Amp)?_.+\.csv$", re.I),
}

def list_archive_summary(zip_path):
    """Print non-macOS file totals and file-type counts for a ZIP."""
    zip_path = Path(zip_path)
    with ZipFile(zip_path) as archive:
        files = [
            item.filename.replace("\\", "/")
            for item in archive.infolist()
            if not item.is_dir()
            and "__MACOSX" not in item.filename.replace("\\", "/").split("/")
        ]
    extensions = Counter(Path(name).suffix.lower() or "[no extension]" for name in files)
    print(f"Archive: {zip_path.name}")
    print(f"Files excluding macOS metadata: {len(files)}")
    print("File types:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count}")
    return files

def inspect_room_a_counts(zip_path):
    """Count Room A PCAPs by activity and development/validation split."""
    counts = {"development": Counter(), "validation": Counter()}
    with ZipFile(zip_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = item.filename.replace("\\", "/")
            parts = path.lower().split("/")
            if not path.lower().endswith(".pcap") or "__macosx" in parts:
                continue
            if "csidata" not in parts:
                continue
            remaining = parts[parts.index("csidata") + 1:]
            activity = next((part for part in remaining if part in ROOM_A_CLASSES), None)
            if activity is None:
                continue
            split = "validation" if "validation" in remaining else "development"
            counts[split][activity] += 1

    print("Room A — Experiment-2")
    for split in ("development", "validation"):
        print(f"\n{split.title()}:")
        for activity in sorted(ROOM_A_CLASSES):
            print(f"  {activity}: {counts[split][activity]}")
        print(f"  Total: {sum(counts[split].values())}")
    return counts

def inspect_room_b_counts(zip_path):
    """Count Room B CSV captures by activity; report unmatched CSVs."""
    counts = Counter()
    unmatched = []
    matched_paths = {activity: [] for activity in ROOM_B_PREFIXES}

    with ZipFile(zip_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = item.filename.replace("\\", "/")
            parts = path.split("/")
            filename = parts[-1]
            if "__MACOSX" in parts or len(parts) < 2:
                continue
            if parts[-2].lower() != "data" or not filename.lower().endswith(".csv"):
                continue

            match = next(
                (activity for activity, pattern in ROOM_B_PREFIXES.items()
                 if pattern.fullmatch(filename)),
                None,
            )
            if match:
                counts[match] += 1
                matched_paths[match].append(path)
            else:
                unmatched.append(path)

    print("Room B — Experiment-3")
    print("Capture counts before creating validation split:")
    for activity in sorted(ROOM_B_PREFIXES):
        print(f"  {activity}: {counts[activity]}")
    print(f"  Total matched: {sum(counts.values())}")
    print(f"Unmatched CSV files: {len(unmatched)}")
    for path in unmatched:
        print(f"  {path}")
    return counts, matched_paths, unmatched

def inspect_csv_inside_zip(zip_path, member_path):
    """Inspect shape, columns, dtypes, missing/infinite values and basic ranges."""
    import pandas as pd
    import numpy as np

    with ZipFile(zip_path) as archive:
        with archive.open(member_path) as file_obj:
            frame = pd.read_csv(file_obj)

    print(f"File: {member_path}")
    print(f"Shape: {frame.shape}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Data types:\n{frame.dtypes.value_counts()}")
    print(f"Missing values: {int(frame.isna().sum().sum())}")
    numeric = frame.select_dtypes(include=[np.number])
    print(f"Infinite values: {int(np.isinf(numeric.to_numpy()).sum())}")
    print(f"Duplicate rows: {int(frame.duplicated().sum())}")
    print(f"All-zero rows: {int((numeric.fillna(0) == 0).all(axis=1).sum())}")
    print(f"Overall numeric min: {numeric.min().min()}")
    print(f"Overall numeric max: {numeric.max().max()}")
    return frame
