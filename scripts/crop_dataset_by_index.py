#!/usr/bin/env python3
"""
Crop dataset txt files by sample index and save trimmed copies.

The plot x-axis used by the training/inference scripts is the sample index of
valid data rows after comment/header lines are skipped. This script keeps the
original header comments and writes only the selected data-row range.

Examples:
    python3 scripts/crop_dataset_by_index.py --end 10000
    python3 scripts/crop_dataset_by_index.py --files "[1]dataset_50_25_new.txt,[4]dataset_100_25_new.txt" --start 0 --end 10000
    python3 scripts/crop_dataset_by_index.py --pattern "[[]*new.txt" --suffix "_0_10000"
"""

import argparse
from pathlib import Path

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_detection_raw.data.loader import parse_line


DEFAULT_DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"


def resolve_input_files(data_dir, files_arg, pattern):
    data_root = Path(data_dir)
    if files_arg:
        resolved = []
        for token in files_arg.split(","):
            token = token.strip()
            if not token:
                continue
            path = Path(token)
            resolved.append(path if path.is_absolute() else data_root / path)
        return resolved
    return sorted(data_root.glob(pattern))


def crop_file(src_path, dst_path, start, end):
    kept = 0
    valid_idx = 0
    header_lines = []
    selected_lines = []

    with src_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if line.lstrip().startswith("#") or not line.strip():
                if valid_idx == 0 and kept == 0:
                    header_lines.append(line)
                continue

            if parse_line(line, line_num) is None:
                continue

            if start <= valid_idx < end:
                selected_lines.append(line if line.endswith("\n") else line + "\n")
                kept += 1
            valid_idx += 1

    if start >= valid_idx:
        raise ValueError(
            f"Requested start index {start} is out of range for {src_path.name} "
            f"(valid samples: {valid_idx})"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line)
        f.write(f"# Cropped valid sample index range: [{start}, {min(end, valid_idx)})\n")
        for line in selected_lines:
            f.write(line)

    return valid_idx, kept


def main():
    parser = argparse.ArgumentParser(description="Crop dataset txt files by sample index")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Dataset directory")
    parser.add_argument(
        "--files",
        default="",
        help="Comma-separated file names or paths. Empty means use --pattern within --data-dir",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern used when --files is empty",
    )
    parser.add_argument("--start", type=int, default=0, help="Start sample index, inclusive")
    parser.add_argument("--end", type=int, default=10000, help="End sample index, exclusive")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <data-dir>/cropped_<start>_<end>",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix added before .txt. Default: _cropped_<start>_<end>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    args = parser.parse_args()

    if args.start < 0 or args.end <= args.start:
        raise ValueError("--start must be >= 0 and --end must be greater than --start")

    input_files = resolve_input_files(args.data_dir, args.files, args.pattern)
    if not input_files:
        raise ValueError("No input txt files found. Check --data-dir, --files, or --pattern.")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.data_dir) / f"cropped_{args.start}_{args.end}"
    suffix = args.suffix if args.suffix else f"_cropped_{args.start}_{args.end}"

    print(f"Cropping valid sample indices [{args.start}, {args.end})")
    print(f"Saving outputs to: {out_dir}")

    for src_path in input_files:
        if not src_path.exists():
            print(f"[WARN] Missing file: {src_path}")
            continue

        dst_name = f"{src_path.stem}{suffix}{src_path.suffix}"
        dst_path = out_dir / dst_name
        if dst_path.exists() and not args.overwrite:
            print(f"[SKIP] Exists already: {dst_path}")
            continue

        total_valid, kept = crop_file(src_path, dst_path, args.start, args.end)
        print(
            f"[OK] {src_path.name} -> {dst_name} | "
            f"kept {kept} / {total_valid} valid samples"
        )


if __name__ == "__main__":
    main()
