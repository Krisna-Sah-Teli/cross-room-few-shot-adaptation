from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import csv
import random
import re

ROOM_B_PREFIXES = {
    "empty": re.compile(r"^Empty(?:_Amp)?_.+\.csv$", re.I),
    "lying": re.compile(r"^Lying(?:_Amp)?_.+\.csv$", re.I),
    "sitting": re.compile(r"^Sitting(?:_Amp)?_.+\.csv$", re.I),
    "standing": re.compile(r"^Standing(?:_Amp)?_.+\.csv$", re.I),
    "walking": re.compile(r"^Walking(?:_Amp)?_.+\.csv$", re.I),
}

def collect_room_b_captures(zip_path):
    """Return {activity: [ZIP member paths]} for CSVs directly under a Data folder."""
    captures = defaultdict(list)
    unmatched = []
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
            activity = next(
                (name for name, pattern in ROOM_B_PREFIXES.items()
                 if pattern.fullmatch(filename)),
                None,
            )
            if activity is None:
                unmatched.append(path)
            else:
                captures[activity].append(path)

    for activity in captures:
        captures[activity] = sorted(captures[activity], key=str.lower)
    return dict(captures), unmatched

def create_room_b_split(zip_path, output_csv, validation_per_class=25, seed=42):
    """
    Deterministically select validation captures from each class.
    Uses sorted ZIP member paths and a seeded random sampler.
    Writes one row per original capture, before window generation.
    """
    captures, unmatched = collect_room_b_captures(zip_path)
    expected = set(ROOM_B_PREFIXES)
    if set(captures) != expected:
        raise ValueError(
            f"Expected classes {sorted(expected)}, found {sorted(captures)}"
        )
    if unmatched:
        raise ValueError(f"Found unmatched CSV files: {unmatched}")

    rng = random.Random(seed)
    rows = []
    for activity in sorted(captures):
        paths = captures[activity]
        if len(paths) < validation_per_class:
            raise ValueError(
                f"{activity} has only {len(paths)} captures; "
                f"cannot select {validation_per_class}"
            )
        selected_indices = set(rng.sample(range(len(paths)), validation_per_class))
        for index, member_path in enumerate(paths):
            rows.append({
                "capture_path": member_path,
                "activity": activity,
                "split": "validation" if index in selected_indices else "development",
            })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["capture_path", "activity", "split"]
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["activity"], row["capture_path"].lower())))

    print(f"Saved capture manifest: {output_csv}")
    print(f"Total captures: {len(rows)}")
    for split in ("validation", "development"):
        split_rows = [row for row in rows if row["split"] == split]
        print(f"\n{split.title()}: {len(split_rows)}")
        for activity in sorted(expected):
            n = sum(row["activity"] == activity for row in split_rows)
            print(f"  {activity}: {n}")

    return rows
