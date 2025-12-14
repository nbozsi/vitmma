import os
import json
import logging
import uuid
from pathlib import Path
from glob import glob
from dateutil.parser import parse
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

INPUT_PATTERN = "/data/bullflagdetector/*"
OUTPUT_DIR = "/app/output"


def unixformat(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return int(value * 1000) if isinstance(value, float) else int(value)

    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
        try:
            return int(parse(value).timestamp() * 1000)
        except ValueError:
            return None
    return None


def extract_label(value):
    labels = value.get("timeserieslabels")
    return str(labels[0]) if labels and isinstance(labels, list) else "Unknown"


def get_csv_filename(task):
    data = task.get("data", {})
    raw_name = data.get("csv") or task.get("file_upload") or task.get("csv") or "unknown.csv"
    return Path(raw_name).name


def parse_json_tasks(tasks, annotator, accumulator):
    for task in tasks:
        csv_name = get_csv_filename(task)

        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                val = res.get("value", {})
                start, end = val.get("start"), val.get("end")

                if start is not None and end is not None:
                    accumulator.append(
                        {
                            "code": annotator,
                            "csv": csv_name,
                            "start": unixformat(start),
                            "end": unixformat(end),
                            "label": extract_label(val),
                        }
                    )


def load_annotations(root_pattern):
    segments = []

    raw_paths = glob(root_pattern)
    root_paths = [Path(p) for p in raw_paths if Path(p).is_dir()]

    # Filter out excluded folders
    root_paths = [p for p in root_paths if "consensus" not in p.name and "sample" not in p.name]

    logging.info(f"Scanning {len(root_paths)} directories for annotation files...")

    for folder in root_paths:
        annotator = folder.name
        for json_path in folder.glob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)

                task_list = [data] if isinstance(data, dict) else data
                parse_json_tasks(task_list, annotator, segments)

            except Exception as e:
                logging.error(f"Failed to read {json_path}: {e}")

    return pl.DataFrame(segments)


def find_csv_path(base_folder, target_filename):
    # 1. Exact match
    if (base_folder / target_filename).exists():
        return base_folder / target_filename

    # 2. Strip UUID prefix heuristic
    clean_target = target_filename
    if "-" in target_filename and len(target_filename) > 36:
        parts = target_filename.split("-", 1)
        if len(parts[0]) >= 8:
            clean_target = parts[1]

    # 3. Recursive search
    for file_path in base_folder.rglob("*"):
        if not file_path.is_file():
            continue

        f_name = file_path.name
        if f_name == target_filename or f_name.endswith(clean_target) or clean_target.endswith(f_name):
            return file_path

    return None


def normalize_dataframe(df):
    df.columns = [c.lower() for c in df.columns]

    time_col = next((c for c in df.columns if "time" in c), df.columns[0])

    try:
        ts = pd.to_numeric(df[time_col])
        df["_ts_ms"] = ts * 1000 if ts.mean() < 2e10 else ts
    except Exception:
        df["_ts_ms"] = pd.to_datetime(df[time_col], utc=True).astype("int64") // 10**6

    df.rename(columns={time_col: "timestamp"}, inplace=True)
    return df


def build_timeseries_dataset(metadata, root_dir):
    all_segments = []
    root_path = Path(root_dir)

    logging.info("Starting time-series extraction...")

    for annotator, group in metadata.groupby("code"):
        student_folder = root_path / annotator
        if not student_folder.exists():
            continue

        for csv_name, csv_group in group.groupby("csv"):
            csv_path = find_csv_path(student_folder, csv_name)

            if not csv_path:
                logging.warning(f"File not found: '{csv_name}' in folder '{annotator}'")
                continue

            try:
                df_raw = pd.read_csv(csv_path, sep=None, engine="python")
                df_raw = normalize_dataframe(df_raw)

                for start, end, label in zip(csv_group["start"], csv_group["end"], csv_group["label"]):
                    mask = (df_raw["_ts_ms"] >= start) & (df_raw["_ts_ms"] <= end)
                    segment = df_raw[mask].copy()

                    if not segment.empty:
                        segment["segment_id"] = str(uuid.uuid4())
                        segment["label"] = label
                        segment["original_csv"] = csv_name
                        all_segments.append(segment)

            except Exception as e:
                logging.error(f"Error processing '{csv_name}': {e}")

    if not all_segments:
        logging.warning("No segments were extracted.")
        return pd.DataFrame()

    final_df = pd.concat(all_segments, ignore_index=True)
    print(f"Successfully compiled {len(final_df)} time-series segments.")
    return final_df


if __name__ == "__main__":
    # load metadata
    meta_df = load_annotations(INPUT_PATTERN)

    if not meta_df.is_empty():
        print(f"Found {len(meta_df)} annotations.")

        # Extract Data
        search_root = str(Path(INPUT_PATTERN).parent)
        ts_df = build_timeseries_dataset(meta_df.to_pandas(), search_root)

        if not ts_df.empty:
            print(f"Extracted {len(ts_df)} time-series rows.")

            # SPLIT DATA (Train/Test)
            unique_ids = ts_df["segment_id"].unique()
            if len(unique_ids) > 1:
                train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

                df_train = ts_df[ts_df["segment_id"].isin(train_ids)]
                df_test = ts_df[ts_df["segment_id"].isin(test_ids)]

                df_train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
                df_test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
                print(f"Saved train.csv ({len(train_ids)} segs) and test.csv ({len(test_ids)} segs) to {OUTPUT_DIR}")
            else:
                print("Not enough segments to split.")
        else:
            print("No time-series data extracted.")
    else:
        print("No annotations found.")
