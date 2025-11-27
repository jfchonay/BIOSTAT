import json
from pathlib import Path

import numpy as np
import pandas as pd

# Root folder that contains one subfolder per subject and Output folder
ROOT_DIR = Path("/Users/josechonay/Documents/relevant_subjects")
OUT_DIR = Path("/Users/josechonay/Documents/resampled")

# Desired sampling rate for both events + GSR
TARGET_FS = 100.0  # Hz

# Original sampling rate of markers-events
EVENT_FS = 90.0  # Hz

# Filenames inside each subject folder (change if yours differ)
MANIFEST_FILENAME = "manifest.json"
EVENTS_FILENAME = "Markers-events.csv"
GSR_JSON_FILENAME = "GSR.json"
GSR_CSV_FILENAME = "GSR.csv"


def resample_timeseries(df: pd.DataFrame, orig_fs: float, target_fs: float) -> pd.DataFrame:
    """
    Resample all numeric columns in df from orig_fs to target_fs
    using linear interpolation on a uniform time axis.
    """
    if len(df) == 0:
        return df.copy()

    # Original and new time axes (in seconds)
    t_orig = np.arange(len(df)) / orig_fs
    t_new_end = t_orig[-1]
    t_new = np.arange(0, t_new_end, 1.0 / target_fs)

    resampled = pd.DataFrame()

    for col in df.columns:
        series = df[col]
        if np.issubdtype(series.dtype, np.number):
            resampled[col] = np.interp(t_new, t_orig, series.to_numpy())
        # Non-numeric columns are ignored here. You could handle them if needed.

    return pd.DataFrame(resampled)


for subject_dir in ROOT_DIR.iterdir():
    if not subject_dir.is_dir():
        continue

    manifest_path = subject_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"[WARN] No manifest.json in {subject_dir}, skipping.")
        continue
    # ---- Read manifest and greenery flag ----
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    # -----------------------------------
    # Create output folder
    # -----------------------------------
    output_dir = OUT_DIR / f"{subject_dir.name}"
    output_dir.mkdir(exist_ok=True)

    # -----------------------------------
    # Copy manifest, add new sampling rate
    # -----------------------------------
    manifest_copy = manifest.copy()
    manifest_copy["resampled_fs"] = TARGET_FS

    with open(output_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest_copy, f, indent=2)

    print(f"  -> Wrote manifest with resampled_fs")
    # ---- Resample markers-events ----
    events_path = subject_dir / EVENTS_FILENAME
    if events_path.exists():
        print(f"  -> Resampling events: {events_path.name}")
        events_df = pd.read_csv(events_path)

        times_s = events_df["sample"] / EVENT_FS

        # New sample index at target_fs
        new_samples = (times_s * TARGET_FS).round().astype(int)
        events_df["onset"] = times_s
        events_df["sample"] = new_samples

        out_events = output_dir / f"Markers-events.csv"
        events_df.to_csv(out_events, index=False)
        print(f"     Saved: {out_events.name}")
    else:
        print(f"  [WARN] No {EVENTS_FILENAME} in {subject_dir}, skipping events.")

    # ---- Resample GSR ----
    gsr_json_path = subject_dir / GSR_JSON_FILENAME
    gsr_csv_path = subject_dir / GSR_CSV_FILENAME

    if gsr_json_path.exists() and gsr_csv_path.exists():
        print(f"  -> Resampling GSR: {gsr_csv_path.name}")
        with open(gsr_json_path, "r") as f:
            meta = json.load(f)
        gsr_fs = meta["stats_original"]["nominal_srate"]
        gsr_df = pd.read_csv(gsr_csv_path)

        gsr_resampled = resample_timeseries(gsr_df, orig_fs=gsr_fs, target_fs=TARGET_FS)

        out_gsr = output_dir / f"GSR.csv"
        gsr_resampled.to_csv(out_gsr, index=False)
        print(f"     Saved: {out_gsr.name}")
    else:
        print(f"  [WARN] Missing GSR files in {subject_dir} (need both JSON+CSV).")
