import json
from pathlib import Path
import pandas as pd
from resampling import resample_timeseries, resample_events
from eda_digital_filtering import eda_digital_filter
from datetime import datetime


if __name__ == '__main__':
    # Root folder that contains one subfolder per subject and Output folder
    ROOT_DIR = Path(r"P:\BIOSTAT\processed\resampled_chunks")
    OUT_DIR = Path(r"P:\BIOSTAT\processed\filtered_gsr")
    # Filenames inside each subject folder
    MANIFEST_FILENAME = "manifest.json"
    EVENTS_FILENAME = "Markers-events.csv"
    EVENTS_META = "Markers-events.json"
    GSR_FILENAME = "GSR.csv"
    META_FILENAME = "GSR.json"
    # desired sampling rate
    t_fs = 15

    for subject_dir in ROOT_DIR.iterdir():
        if not subject_dir.is_dir():
            continue
        # Load the data from the events and gsr
        e_meta_path = subject_dir / EVENTS_META
        if not e_meta_path.exists():
            print("[WARN] Missing events file:", e_meta_path)
            continue
        with open(e_meta_path, "r") as e_m:
            e_meta = json.load(e_m)
        events_path = subject_dir / EVENTS_FILENAME
        if not events_path.exists():
            print("[WARN] Missing events file:", events_path)
            continue
        events_df = pd.read_csv(str(events_path), dtype=str, engine="python")
        # convert numeric cols
        for col in ["onset", "sample", "offset", "duration"]:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce")
        gsr_csv_path = subject_dir / GSR_FILENAME
        if not gsr_csv_path.exists():
            print("[WARN] Missing GSR file:", gsr_csv_path)
            continue
        gsr_df = pd.read_csv(gsr_csv_path)
        # Resample GSR and events, they have the same fs in this case
        EVENT_FS = e_meta["resampled_fs"]
        # Resample markers-events
        if events_path.exists():
            events_df = pd.read_csv(events_path)
            events_df_resampled = resample_events(
                events_df,
                orig_fs=EVENT_FS,
                target_fs=t_fs
            )
        # Create output folder and save the event files with updated metadata
        output_dir = OUT_DIR / f"{subject_dir.name}"
        output_dir.mkdir(exist_ok=True)
        out_events = output_dir / EVENTS_FILENAME
        events_df_resampled.to_csv(out_events, index=False)
        e_meta_copy = e_meta.copy()
        e_meta_copy["resampled_fs"] = t_fs
        with open(output_dir / EVENTS_META, "w") as e_m_c:
            json.dump(e_meta_copy, e_m_c, indent=2)
        # resample and invert the EDA signal
        gsr_rs = resample_timeseries(gsr_df, EVENT_FS, t_fs)
        gsr = (-gsr_df["CH1"].to_numpy())
        # Process the raw EDA signal
        try:
            gsr_filt = eda_digital_filter(gsr, sampling_rate=t_fs, cut_off=2, order=4)
        except ValueError as e:
            if "padlen" in str(e) and "length of the input vector" in str(e):
                print(f"[SKIP] subject={subject_dir.name}: {e}")
                continue
            raise
        # save data and updated JSON files
        gsr_filtered = pd.DataFrame(gsr_filt, columns=["CH1"])
        meta_path = subject_dir / META_FILENAME
        with open(meta_path, "r") as m:
            meta_data = json.load(m)
        meta_copy = meta_data.copy()
        meta_copy["resampled_fs"] = t_fs
        meta_copy["filter_apply"] = f"Low pass digital zero phase filter with cutoff at {2}Hz"
        with open(output_dir / META_FILENAME, "w") as m_c:
            json.dump(meta_copy, m_c, indent=2)
        out_data = output_dir / GSR_FILENAME
        gsr_filtered.to_csv(out_data, index=False)
        manifest_path = subject_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            print(f"[WARN] No manifest.json in {subject_dir}, skipping.")
            continue
        # Read manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest_copy = manifest.copy()
        manifest_copy["resampled_fs"] = t_fs
        manifest_copy["export_time_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        # save the manifest with the updated data
        with open(output_dir / MANIFEST_FILENAME, "w") as o:
            json.dump(manifest_copy, o, indent=2)
