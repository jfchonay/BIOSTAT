import json
from pathlib import Path
from scipy.signal import resample
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.errors import EmptyDataError, ParserError



def resample_timeseries(df: pd.DataFrame, orig_fs: float, target_fs: float) -> pd.DataFrame:
    """
    Resample all numeric + boolean columns in df from orig_fs to target_fs
    using an FFT-based method (brick-wall low-pass at new Nyquist).
    Assumes rows are uniformly sampled at orig_fs.
    """
    if df.empty:
        return df.copy()
    # Number of samples in the input
    n_orig = df.shape[0]
    # Keep the duration approximately the same
    duration = (n_orig - 1) / orig_fs  # seconds
    n_new = int(round(duration * target_fs)) + 1
    # Split numeric and boolean columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns  # covers numpy + pandas nullable bool
    # If nothing to resample, just return a copy
    if len(num_cols) == 0 and len(bool_cols) == 0:
        return df.copy()
    # Columns to resample (preserve original order)
    cols_to_resample = [c for c in df.columns if c in num_cols or c in bool_cols]
    sub = df[cols_to_resample].copy()
    # Cast boolean columns to float so they can be resampled
    if len(bool_cols) > 0:
        sub[bool_cols] = sub[bool_cols].astype(float)
    # Time × channels array
    data = sub.to_numpy()
    # FFT-based resampling along time axis (axis=0)
    data_rs = resample(data, num=n_new, axis=0)
    # Back to DataFrame
    resampled = pd.DataFrame(data_rs, columns=cols_to_resample)
    # Convert boolean columns back to bool via thresholding
    # (values should be in [0, 1]-ish; tune threshold if needed)
    for col in bool_cols:
        # >= 0.5 means "mostly True" after resampling
        resampled[col] = resampled[col] >= 0.5

    return resampled


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
    ROOT_DIR = Path(r"P:\BIOSTAT\raw_data")
    OUT_DIR = Path(r"P:\BIOSTAT\resampled")
    # Desired sampling rate for all files
    TARGET_FS = 100.0  # Hz
    # Filenames inside each subject folder
    MANIFEST_FILENAME = "manifest.json"
    BODY_FILENAME = "rigidBody.json"

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
        manifest_copy["export_time_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        # save the manifest with the updated data
        with open(output_dir / MANIFEST_FILENAME, "w") as o:
            json.dump(manifest_copy, o, indent=2)
        # get the sampling rate for the events
        body_path = subject_dir / BODY_FILENAME
        if not body_path.exists():
            print(f"[WARN] No rigidBody.json in {subject_dir}, skipping.")
            continue
        # Read json where sr for events is stores
        with open(body_path, "r") as b:
            body = json.load(b)
        EVENT_FS = body["stats_original"]["effective_srate"]
        # read every stream to re sample them
        for stream in manifest["streams"]:
            if stream["role"] == "events":
                try:
                    events_df = pd.read_csv(stream["csv"])
                except (EmptyDataError, ParserError) as e:
                    print(f"[WARN] Bad/empty events CSV: {stream['csv']} ({e})")
                    continue
                if events_df.empty:
                    print(f"[WARN] Events CSV has 0 rows: {stream['csv']}")
                    continue
                with open(stream["json"], "r") as m:
                    meta = json.load(m)
                events_df_resampled = resample_events(
                    events_df,
                    orig_fs=EVENT_FS,
                    target_fs=TARGET_FS
                )
                meta_copy = meta.copy()
                meta_copy["resampled_fs"] = TARGET_FS
                with open(output_dir / f"{stream['type']}-events.json", "w") as m_c:
                    json.dump(meta_copy, m_c, indent=2)
                out_events = output_dir / f"{stream['type']}-events.csv"
                events_df_resampled.to_csv(out_events, index=False)
            elif stream["role"] == "signal":
                try:
                    data = pd.read_csv(stream["csv"])
                except (EmptyDataError, ParserError) as e:
                    print(f"[WARN] Bad/empty signal CSV: {stream['csv']} ({e})")
                    continue
                if data.empty:
                    print(f"[WARN] Signal CSV has 0 rows: {stream['csv']}")
                    continue
                with open(stream["json"], "r") as m:
                    meta_data = json.load(m)
                DATA_FS = meta_data["stats_original"]["effective_srate"]
                data_resampled = resample_timeseries(data, orig_fs=DATA_FS, target_fs=TARGET_FS)
                meta_copy = meta_data.copy()
                meta_copy["resampled_fs"] = TARGET_FS
                with open(output_dir / f"{stream['type']}.json", "w") as m_c:
                    json.dump(meta_copy, m_c, indent=2)
                out_data = output_dir / f"{stream['type']}.csv"
                data_resampled.to_csv(out_data, index=False)
