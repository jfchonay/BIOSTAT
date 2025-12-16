import neurokit2 as nk
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resampling import resample_timeseries, resample_events
from eda_digital_filtering import eda_digital_filter

if __name__ == '__main__':
    # Root folder that contains one subfolder per subject and Output folder
    ROOT_DIR = Path(r"P:\BIOSTAT\test_filter\recovery")
    OUT_DIR = Path(r"P:\BIOSTAT\test_filter\filtered\recovery_2Hz")

    # Filenames inside each subject folder (change if yours differ)
    MANIFEST_FILENAME = "manifest.json"
    EVENTS_FILENAME = "Markers-events.csv"
    GSR_FILENAME = "GSR.csv"

    # Events names
    FIRST_EVENT = "Station scene  starting recovery phase"
    LAST_EVENT = "Station scene  end recovery phase  showing waypoint to debrief recovery"

    t_fs = 15

    # Parameters of the epoch
    s_b = 1 * t_fs
    s_a = 1 * t_fs

    for subject_dir in ROOT_DIR.iterdir():
        if not subject_dir.is_dir():
            continue
        # Load the data from the events and gsr
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
        # Resample GSR and events
        gsr_rs = resample_timeseries(gsr_df, 100, t_fs)
        EVENT_FS = 100
        events_path = subject_dir / EVENTS_FILENAME
        # Resample markers-events
        if events_path.exists():
            print(f"Resampling events: {events_path.name}")
            events_df = pd.read_csv(events_path)

            events_df_resampled = resample_events(
                events_df,
                orig_fs=EVENT_FS,
                target_fs=t_fs
            )
        # Check if the events exist in the file
        if not ((events_df_resampled["value"] == FIRST_EVENT).any() and
                (events_df_resampled["value"] == LAST_EVENT).any()):
            print(f"[WARN] Missing events in {subject_dir}, skipping.")
            continue
        # Invert
        gsr = (-gsr_df["CH1"].to_numpy())
        # TRYING TO PROCESS THE DATA
        # Process the raw EDA signal
        gsr_bio = eda_digital_filter(gsr, sampling_rate=t_fs, cut_off=2)
        gsr_neuro = eda_digital_filter(gsr, sampling_rate=t_fs, cut_off=3)
        # Cut data
        # Using the name of the events get the sample in which every event is found
        start_idx = events_df_resampled["sample"][events_df_resampled.index[events_df["value"] == FIRST_EVENT]]
        if len(start_idx) == 1:
            first_idx = start_idx.iloc[0]
        else:
            first_idx = start_idx.iloc[-1]
        end_idx = events_df_resampled["sample"][events_df_resampled.index[events_df["value"] == LAST_EVENT]]
        if len(end_idx) == 1:
            last_idx = end_idx.iloc[0]
        else:
            last_idx = end_idx.iloc[-1]
        start_sample = max(0, first_idx - s_b)
        end_sample = min(last_idx + s_a, len(gsr_df) - 1)  # inclusive index
        # Extract GSR segment
        gsr_event_bio = gsr_bio[start_sample:end_sample]
        gsr_event_neuro = gsr_neuro[start_sample:end_sample]
        gsr_event_raw = gsr[start_sample:end_sample]
        # time vector
        n_samples = len(gsr_event_bio)
        time_sec = np.arange(n_samples) / t_fs  # seconds
        if time_sec.size == 0:
            print(f"[WARN] No valid time points for {subject_dir}, skipping.")
            continue
        # segment is from seg_start -> seg_end, so shift by seg_start, then / fs
        # Events to use
        between_mask = (events_df_resampled["sample"] >= first_idx) & (events_df_resampled["sample"] <= last_idx)
        events_between = events_df_resampled.loc[between_mask].copy()
        # Convert event sample positions to seconds (same reference as time_sec)
        events_between["time_sec"] = (events_between["sample"] - start_sample) / t_fs
        events_between = events_between.reset_index(drop=True)
        events_between["label"] = (events_between.index + 1).astype(str)
        # save accordingly to the group they belong
        manifest_path = subject_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            print(f"[WARN] No manifest.json in {subject_dir}, skipping.")
            continue
        # Read manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        # According to group save in desired list
        sub_id = manifest.get("id")[:-5]
        # define the spacing between them
        ys = [gsr_event_raw, gsr_event_bio, gsr_event_neuro]
        # dynamic range across all signals
        global_range = max(y.max() - y.min() for y in ys)
        # spacing between stacked curves
        spacing = 0.9 * global_range  # 20% gap
        # -------- Plot GSR + event markers --------
        fig, ax = plt.subplots(figsize=(12, 4), dpi=200)  # higher dpi
        # Plot normalized GSR
        # clean signal: thin, saturated
        ax.plot(
            time_sec,
            gsr_event_bio + (3*spacing),
            linewidth=1,
            color='red',  # saturated
            alpha=0.5,
            label='2Hz (digital)'
        )
        #
        # ax.plot(
        #     time_sec,
        #     gsr_event_neuro + (2*spacing),
        #     linewidth=1,
        #     color='green',  # saturated
        #     alpha=0.5,
        #     label='3Hz (digital)'
        # )
        # raw signal: thicker, transparent
        ax.plot(
            time_sec,
            gsr_event_raw + (2*spacing),
            linewidth=1,
            color='blue',
            alpha=0.5,  # light & transparent
            label='raw'
        )
        # Grid + shared color for grid and event lines
        grid_color = "0.7"  # light gray
        ax.grid(True, which="both", linestyle="--", alpha=0.9, color=grid_color)
        # X ticks every 10 seconds
        ax.set_xticks(np.arange(0, time_sec[-1] + 15, 15))
        # Event lines
        for _, row in events_between.iterrows():
            t_ev = row["time_sec"]
            # vertical dashed line in same color as grid
            ax.axvline(t_ev, linestyle="--", linewidth=0.5, color=grid_color)
        ax.tick_params(axis="x", labelsize=6)
        # Labels, title, grid
        ax.set_title(f"Subject {sub_id} at {t_fs}Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel('')
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        # optionally save instead of showing, or both
        # out_path = subject_dir / f"{sub_id}_gsr_pre.png"
        # plt.savefig(out_path, dpi=300, bbox_inches="tight")
        # plt.show()
        out_path = OUT_DIR / f"{sub_id}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[INFO] Saved plot to {out_path}")