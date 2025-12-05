import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Root folder that contains one subfolder per subject and Output folder
ROOT_DIR = Path(r"P:\BIOSTAT\resampled")
OUT_DIR = Path(r"P:\BIOSTAT\stress_section\gsr_raw_markers")

# Filenames inside each subject folder (change if yours differ)
MANIFEST_FILENAME = "manifest.json"
EVENTS_FILENAME = "Markers-events.csv"
GSR_FILENAME = "GSR.csv"

# Events names
FIRST_EVENT = "Tutorial  starting baseline"
LAST_EVENT = "Station scene  end recovery phase  showing waypoint to debrief recovery"

# Parameters of the epoch
s_b = 1 * 100
s_a = 1 * 100

greenery_true = []
greenery_false = []

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
    gsr_df = pd.read_csv(gsr_csv_path)
    # Check if the events exist in the file
    if not ((events_df["value"] == FIRST_EVENT).any() and
            (events_df["value"] == LAST_EVENT).any()):
        print(f"[WARN] Missing events in {subject_dir}, skipping.")
        continue
    # Using the name of the events get the sample in which every event is found
    first_idx = events_df["sample"][events_df.index[events_df["value"] == FIRST_EVENT][0]]
    last_idx = events_df["sample"][events_df.index[events_df["value"] == LAST_EVENT][0]]
    start_sample = max(0, first_idx - s_b)
    end_sample   = min(last_idx + s_a, len(gsr_df) - 1)  # inclusive index
    # Extract GSR segment
    gsr_event = gsr_df["CH1"].iloc[start_sample:end_sample + 1].to_numpy()
    # Invert and z-score normalize
    gsr = stats.zscore(-gsr_event)
    # time vector
    n_samples = len(gsr)
    time_sec = np.arange(n_samples) / 100  # seconds
    if time_sec.size == 0:
        print(f"[WARN] No valid time points for {subject_dir}, skipping.")
        continue
    # Events to use
    between_mask = (events_df["sample"] >= first_idx) & (events_df["sample"] <= last_idx)
    events_between = events_df.loc[between_mask].copy()
    # Convert event sample positions to seconds (same reference as time_sec)
    events_between["time_sec"] = (events_between["sample"] - start_sample) / 100
    events_between = events_between.reset_index(drop=True)
    events_between["label"] = (events_between.index + 1).astype(str)
    # segment is from seg_start -> seg_end, so shift by seg_start, then / fs
    # save accordingly to the group they belong
    manifest_path = subject_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"[WARN] No manifest.json in {subject_dir}, skipping.")
        continue
    # Read manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    # According to group save in desired list
    sub_id = manifest.get("id")
    # -------- Plot GSR + event markers --------
    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)  # higher dpi

    # Plot normalized GSR
    ax.plot(time_sec, gsr, linewidth=0.25)

    # Grid + shared color for grid and event lines
    grid_color = "0.7"  # light gray
    ax.grid(True, which="both", linestyle="--", alpha=0.9, color=grid_color)

    # X ticks every 10 seconds
    ax.set_xticks(np.arange(0, time_sec[-1] + 30, 30))

    # Y ticks every 0.5 (based on current limits)
    ymin, ymax = ax.get_ylim()
    ytick_start = np.floor(ymin * 2) / 2.0
    ytick_end = np.ceil(ymax * 2) / 2.0
    ax.set_yticks(np.arange(ytick_start, ytick_end + 0.5, 0.5))
    ax.tick_params(axis="x", labelsize=6)

    # Event lines + labels on the time axis
    for _, row in events_between.iterrows():
        t_ev = row["time_sec"]

        # vertical dashed line in same color as grid
        ax.axvline(t_ev, linestyle="--", linewidth=0.3, color=grid_color)

        # label on the time axis (bottom)
        ax.text(
            t_ev,
            -0.1,  # slightly below axis
            row["label"],  # "1", "2", ...
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            rotation=45,
            clip_on=False,
        )
    # Labels, title, grid
    ax.set_title(f"Subject {sub_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GSR (z-scored)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # optionally save instead of showing, or both
    # out_path = subject_dir / f"{subject_id}_gsr_segment.png"
    # plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # plt.show()
    out_path = OUT_DIR / f"{sub_id}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Saved plot to {out_path}")
