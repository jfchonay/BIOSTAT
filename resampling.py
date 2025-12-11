import json
from pathlib import Path
from scipy.signal import resample
import numpy as np
import pandas as pd


def resample_timeseries(df: pd.DataFrame, orig_fs: float, target_fs: float) -> pd.DataFrame:
    """
    Resample all numeric columns in df from orig_fs to target_fs
    using an FFT-based method (brick-wall low-pass at new Nyquist)
    Assumes rows are uniformly sampled at orig_fs.
    """
    if df.empty:
        return df.copy()
    # Number of samples in the input
    n_orig = df.shape[0]
    # Compute how many samples we want in the resampled signal.
    # This keeps the duration approximately the same.
    duration = (n_orig - 1) / orig_fs       # in seconds
    n_new = int(round(duration * target_fs)) + 1
    # Select numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        # nothing to resample
        return df.copy()
    # Convert to 2D array (time x channels)
    data = num_df.to_numpy()
    # FFT-based resampling along the time axis (axis=0)
    # This implicitly applies a brick-wall low-pass at target_fs / 2
    data_rs = resample(data, num=n_new, axis=0)
    # Build new DataFrame for numeric data
    resampled_num = pd.DataFrame(data_rs, columns=num_df.columns)
    return resampled_num


def resample_events(
    events_df: pd.DataFrame,
    orig_fs: float,
    target_fs: float) -> pd.DataFrame:
    """
    Map event sample indices from orig_fs to target_fs.
    - Preserves event onset time in seconds (up to rounding).
    - Uses the same sample-rate ratio as signal resampling.
    - Optionally drops events that fall outside [0, n_samples_out-1].
    """
    if events_df.empty:
        return events_df.copy()

    if orig_fs <= 0 or target_fs <= 0:
        raise ValueError("orig_fs and target_fs must be positive.")
    # Scale factor between sample indices
    ratio = target_fs / orig_fs
    out = events_df.copy()
    # Original integer sample indices
    sample_orig = out["sample"].astype(int).to_numpy()
    # Onset in seconds at original sampling rate (assuming 0-based samples)
    onset_s = sample_orig / orig_fs
    # New sample indices at target_fs
    sample_new = np.rint(sample_orig * ratio).astype(int)
    out["onset"] = onset_s
    out["sample"] = sample_new

    return out

if __name__ == "__main__":
    # Root folder that contains one subfolder per subject and Output folder
    ROOT_DIR = Path(r"P:\BIOSTAT\data_chunks")
    OUT_DIR = Path(r"P:\BIOSTAT\resampled_chunks")
    # Desired sampling rate for all files
    TARGET_FS = 100.0  # Hz
    # Filenames inside each subject folder
    MANIFEST_FILENAME = "manifest.json"
    EVENTS_FILENAME = "Markers-events.csv"
    BODY_FILENAME = "rigidBody.json"
    GSR_JSON_FILENAME = "GSR.json"
    GSR_CSV_FILENAME = "GSR.csv"

    for subject_dir in ROOT_DIR.iterdir():
        if not subject_dir.is_dir():
            continue

        manifest_path = subject_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            print(f"[WARN] No manifest.json in {subject_dir}, skipping.")
            continue
        # Read manifest and greenery flag
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        # Create output folder
        output_dir = OUT_DIR / f"{subject_dir.name}"
        output_dir.mkdir(exist_ok=True)
        # Copy manifest, add new sampling rate
        manifest_copy = manifest.copy()
        manifest_copy["resampled_fs"] = TARGET_FS

        with open(output_dir / MANIFEST_FILENAME, "w") as o:
            json.dump(manifest_copy, o, indent=2)

        print(f" Wrote manifest with resampled_fs")

        # Resample GSR
        gsr_json_path = subject_dir / GSR_JSON_FILENAME
        gsr_csv_path = subject_dir / GSR_CSV_FILENAME

        if gsr_json_path.exists() and gsr_csv_path.exists():
            print(f" Resampling GSR: {gsr_csv_path.name}")
            with open(gsr_json_path, "r") as f:
                meta = json.load(f)
            gsr_fs = meta["stats_original"]["estimated_srate"]
            gsr_df = pd.read_csv(gsr_csv_path)
            gsr_resampled = resample_timeseries(gsr_df, orig_fs=gsr_fs, target_fs=TARGET_FS)
            out_gsr = output_dir / f"GSR.csv"
            gsr_resampled.to_csv(out_gsr, index=False)
            print(f" Saved: {out_gsr.name}")
        else:
            print(f"  [WARN] Missing GSR files in {subject_dir} (need both JSON+CSV).")
        body_path = subject_dir / BODY_FILENAME
        if not body_path.exists():
            print(f"[WARN] No rigidBody.json in {subject_dir}, skipping.")
            continue
        # Read json where sr for events is stores
        with open(body_path, "r") as b:
            body = json.load(b)
        EVENT_FS = body["stats_original"]["estimated_srate"]
        # Resample markers-events
        events_path = subject_dir / EVENTS_FILENAME
        if events_path.exists():
            print(f"Resampling events: {events_path.name}")
            events_df = pd.read_csv(events_path)

            events_df_resampled = resample_events(
                events_df,
                orig_fs=EVENT_FS,
                target_fs=TARGET_FS
            )
            out_events = output_dir / "Markers-events.csv"
            events_df_resampled.to_csv(out_events, index=False)
            print(f"Saved: {out_events.name}")
        else:
            print(f"  [WARN] No {EVENTS_FILENAME} in {subject_dir}, skipping events.")
