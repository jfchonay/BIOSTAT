import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Root folder that contains one subfolder per subject and Output folder
ROOT_DIR = Path(r"P:\BIOSTAT\resampled")
OUT_DIR = Path(r"P:\BIOSTAT\stress_section")

# Filenames inside each subject folder (change if yours differ)
MANIFEST_FILENAME = "manifest.json"
EVENTS_FILENAME = "Markers-events.csv"
GSR_FILENAME = "GSR.csv"

# Events names
FIRST_EVENT = "Playing voice over  TutorialStartBaseline"
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
    if not any(events_df.index[events_df["value"] == FIRST_EVENT]) and any(events_df.index[events_df["value"] == LAST_EVENT]) :
        print (f"[WARN] No events in {subject_dir}, skipping.")
        continue
    # Using the name of the events get the sample in which every event is found
    first_idx = events_df["sample"][events_df.index[events_df["value"] == FIRST_EVENT][0]]
    last_idx = events_df["sample"][events_df.index[events_df["value"] == LAST_EVENT][0]]
    # Use the sample to cut the section of data from the GSR
    gsr_event = np.array(gsr_df["CH1"][first_idx-s_b:last_idx+s_a])
    # Invert the data and get the mean
    gsr = [-g for g in gsr_event]
    # save accordingly to the group they belong
    manifest_path = subject_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"[WARN] No manifest.json in {subject_dir}, skipping.")
        continue
    # Read manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    # According to group save in desired list
    greenery = manifest.get("greenery")
    if greenery == "1":
        greenery_true.append(gsr)
    elif greenery == "0":
        greenery_false.append(gsr)

# Convert lists to numpy arrays for easy statistics
green = np.array(greenery_true)   # shape = (n, 3)
neutral = np.array(greenery_false)   # shape = (n, 3)

# Compute means and standard deviations along rows
mean_g = green.mean(axis=0)  # shape (3,)
std_g  = green.std(axis=0)

mean_n = neutral.mean(axis=0)
std_n  = neutral.std(axis=0)

# X positions (one per float position)
x = np.arange(3)
# Plot
plt.figure()
# Plot Green
plt.errorbar(x, mean_g, fmt='-o', label="P1")
# Plot Neutral
plt.errorbar(x, mean_n, fmt='-o', label="P0")
# Formatting
plt.xticks(x, ["Baseline", "Stress", "Recovery"])
plt.xlabel("Event")
plt.ylabel("Average GSR")
plt.title("Mean ± SD for GSR in every event")
plt.legend()
plt.show()
