# -*- coding: utf-8 -*-
"""
Cut HMD data around an event onset and export to CSV.
- Events: CSV with at least columns ['event', 'onset'] where onset is in seconds.
- HMD: JSON time series, either with 'timestamp' per sample (sec or ms) OR with
       {'sampling_rate'/'srate', 'start_time' (optional), and 'data'}.
- Keeps HMD columns by index (default: 5:7 in MATLAB 1-based -> 4:7 in Python 0-based, i.e., 4,5,6).
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import sys

# ----------------------- Config -----------------------
m_dir = Path(r"P:\BIOSTAT\raw_data")
out_dir = Path(r"P:\BIOSTAT\nudging\hmd_epoched")
s_event = "Station scene  starting free choice phase"
e_event = "Station scene  user told operator why they chose this spot  showing waypoint to assessment point"
pre_seconds = 1.0          # seconds before onset
cols_to_keep = ['rigid_x', 'rigid_y', 'rigid_z']   # 0-based column indices to export (equivalent to MATLAB 5:7)
# ------------------------------------------------------


def find_subject_pairs(base_dir: Path, data_name, event_name):
    """
    Heuristic: For each subject subfolder, find all possible .
    """
    data = []
    events = []
    for entry in base_dir.iterdir():
        if entry.is_dir():
            # Try to find an events CSV and an JSON inside this subject folder
            all_csv = sorted(entry.glob("*.csv"))
            all_json = sorted(entry.glob("*.json"))
            if not all_csv or not all_json:
                continue
            for file_csv in all_csv:
                if data_name in file_csv.name:
                    data.append(file_csv)
                if event_name in file_csv.name:
                    events.append(file_csv)
            for file_json in all_json:
                if data_name in file_json.name:
                    data.append(file_json)
                if event_name in file_json.name:
                    events.append(file_json)
        else:
            pass
    return data, events


def load_events_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns
    # Try common column names for the event label; adjust as needed
    possible_event_cols = ["event", "Event", "value", "label", "type"]
    event_col = next((c for c in possible_event_cols if c in df.columns), None)
    if event_col is None:
        raise ValueError(f"No event label column found in {path}. "
                         f"Expected one of {possible_event_cols}.")
    if "onset" not in df.columns:
        raise ValueError(f"'onset' column not found in {path}.")
    # keep only useful columns
    return df.rename(columns={event_col: "event"})[["event", "onset"]]


def load_hmd_json(path: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with a 'timestamp' column in seconds and data columns.
    Accepts a few common shapes:
    (A) List[ {timestamp: <sec or ms>, a:..., b:...} ]
    (B) Dict with {'data': List[List or Dict]], 'sampling_rate'/'srate', optional 'start_time'}
        If 'data' is a list of lists, it becomes columns 0..N-1.
        If 'data' is a list of dicts, keys become columns.
    """
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    def ensure_seconds(ts):
        ts = np.asarray(ts, dtype=float)
        # If looks like ms (e.g., values ~1.7e12 or ~1e6 range for elapsed ms), convert to sec
        if np.nanmedian(ts) > 1e6:
            ts = ts / 1000.0
        return ts

    # Case A: list of dict samples
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
        if "timestamp" not in df.columns:
            raise ValueError(f"No 'timestamp' in JSON list at {path}")
        df["timestamp"] = ensure_seconds(df["timestamp"].values)
        # Move timestamp first
        other_cols = [c for c in df.columns if c != "timestamp"]
        return df[["timestamp"] + other_cols]

    # Case B: dict with meta + data
    if isinstance(obj, dict):
        data = obj.get("data", None)
        srate = obj.get("sampling_rate", obj.get("srate", None))
        start_time = obj.get("start_time", 0.0)

        if data is None:
            # maybe data is under a different key, try a few guesses
            for k in ["samples", "signals", "hmd", "values"]:
                if k in obj:
                    data = obj[k]
                    break

        if data is None:
            raise ValueError(f"No time-series 'data' found in {path}")

        # Build DataFrame from data
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
                # If there's already a timestamp, normalize to seconds
                if "timestamp" in df.columns:
                    df["timestamp"] = ensure_seconds(df["timestamp"].values)
                else:
                    if srate is None:
                        raise ValueError(f"No 'timestamp' and no 'sampling_rate'/'srate' in {path}")
                    n = len(df)
                    t0 = float(start_time) if start_time is not None else 0.0
                    ts = t0 + np.arange(n, dtype=float) / float(srate)
                    df.insert(0, "timestamp", ts)
                return df
            else:
                # list of lists -> numeric columns
                df = pd.DataFrame(data)
                if srate is None:
                    raise ValueError(f"No 'sampling_rate'/'srate' for list-of-lists in {path}")
                n = len(df)
                t0 = float(start_time) if start_time is not None else 0.0
                ts = t0 + np.arange(n, dtype=float) / float(srate)
                df.insert(0, "timestamp", ts)
                return df

    raise ValueError(f"Unsupported JSON structure in {path}")


def slice_window(df_hmd: pd.DataFrame, onset_sec: float, pre: float, post: float) -> pd.DataFrame:
    t_start = onset_sec - pre
    t_end = onset_sec + post
    # boolean mask on timestamps (inclusive start, inclusive end)
    m = (df_hmd["timestamp"] >= t_start) & (df_hmd["timestamp"] <= t_end)
    return df_hmd.loc[m].copy()


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    data, events = find_subject_pairs(m_dir, "rigidBody", "Markers-events")
    if not data and events:
        print(f"No subject folders with CSV+JSON found in {m_dir}", file=sys.stderr)
        return

    for idx, (subject, events_csv, events_json, subj_folder) in enumerate(events):
        try:
            events = load_events_csv(events_csv)
        except Exception as e:
            print(f"[{idx}] {subject}: Could not read events CSV ({events_csv.name}): {e}")
            continue

        # find the matching event
        match = events.loc[events["event"] == s_event]
        if match.empty:
            print(f"[{idx}] {subject}: No event '{s_event}' found. Skipping…")
            continue

        onset = float(match.iloc[0]["onset"])

        try:
            hmd = load_hmd_json(hmd_json)
        except Exception as e:
            print(f"[{idx}] {subject}: Could not read HMD JSON ({hmd_json.name}): {e}")
            continue

        # cut window
        cut = slice_window(hmd, onset, pre_seconds, duration)

        if cut.empty:
            print(f"[{idx}] {subject}: Window around onset={onset:.3f}s produced no samples. Skipping…")
            continue

        # keep only desired columns (by index)
        # Ensure we have enough columns: timestamp + data columns
        cols = list(cut.columns)
        # Guarantee timestamp is first
        if cols[0] != "timestamp":
            cols = ["timestamp"] + [c for c in cols if c != "timestamp"]

        # Select by index safely
        # Map indices to column names (skip timestamp for indexing like MATLAB's data channels)
        data_cols = cols[1:]
        selected = []
        for ix in cols_to_keep_by_index:
            if 0 <= ix < len(data_cols):
                selected.append(data_cols[ix])
        out_df = cut[["timestamp"] + selected]

        out_path = out_dir / f"{subject}_start_stress.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[{idx}] {subject}: wrote {out_path}")

if __name__ == "__main__":
    main()
