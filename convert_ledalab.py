import numpy as np
from scipy.io import savemat
import json
from pathlib import Path
import pandas as pd

def fillmissing_linear(x: np.ndarray) -> np.ndarray:
    """
    MATLAB fillmissing(x,'linear') equivalent for 1D arrays (NaN = missing).
    - Fills interior NaNs by linear interpolation.
    - Extrapolates ends using nearest non-NaN (MATLAB behavior is similar if you don't specify EndValues).
    """
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    if n == 0:
        return x

    idx = np.arange(n)
    m = np.isfinite(x)
    if not m.any():
        return x  # all NaN -> unchanged

    # linear interp for all points, with endpoint extrapolation via left/right fill
    x_interp = np.interp(idx, idx[m], x[m])
    return x_interp


def make_gsr_mat(
    out_path: str,
    cut_data: np.ndarray,
    eda_times: np.ndarray,
    latency_s: int,
    latency_e: int,
    eda_srate: float,
    events_sub
):
    """
    Creates a .mat file with a MATLAB struct `data` containing:
      - conductance : fillmissing(cut_data,'linear')
      - time        : eda.times(latency_s-1*srate : latency_e+1*srate)/1000
      - timeoff     : 0
      - event       : struct array with fields time, nid, name, userdata

    Parameters
    ----------
    events_sub : iterable
        Each element should provide:
          - latency (samples)
          - value   (name string)
          - type    (userdata; string or numeric)
        Example element: {'latency': 12345, 'value': 'stim', 'type': 'trigger'}
        or an object with attributes .latency/.value/.type
    """
    # --- conductance (MATLAB typically stores as column vectors; choose shape as you prefer)
    gsr_full = fillmissing_linear(np.asarray(cut_data)).astype(np.float64)

    # --- time slice (indices are assumed to be integer sample indices)
    i0 = int(latency_s)
    i1 = int(latency_e)

    # clamp to valid range (optional but prevents IndexError)
    i0 = max(i0, 0)
    i1 = min(i1, len(eda_times) - 1)

    conductance = gsr_full[i0:i1+1]
    time = (np.asarray(eda_times, dtype=np.float64)[i0:i1 + 1])

    # --- event struct array
    # Build a 1xN MATLAB struct array using a structured numpy array with object fields
    ev_list = list(events_sub)
    n_ev = len(ev_list)

    ev_dtype = np.dtype([
        ("time",     "O"),
        ("nid",      "O"),
        ("name",     "O"),
        ("userdata", "O"),
    ])
    event = np.empty((1, n_ev), dtype=ev_dtype)

    def get_field(e, k):
        return e[k] if isinstance(e, dict) else getattr(e, k)

    for i, e in enumerate(ev_list, start=1):
        event[0, i - 1]["time"]     = float(get_field(e, "sample")) / float(eda_srate)
        event[0, i - 1]["nid"]      = int(i)
        event[0, i - 1]["name"]     = str(get_field(e, "value"))
        event[0, i - 1]["userdata"] = get_field(e, "type")  # keep as-is (string/number/etc.)

    # --- top-level `data` struct
    # A MATLAB struct is created by saving a dict under the key "data".
    data = {
        "conductance": conductance,
        "time":        time,
        "timeoff":     np.array([[0.0]]),  # scalar (1x1) like MATLAB
        "event":       event,
    }

    savemat(out_path, {"data": data}, do_compression=True)


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    # Root folder that contains one subfolder per subject and Output folder
    ROOT_DIR = Path(r"P:\BIOSTAT\processed\filtered_gsr")
    OUT_DIR = Path(r"P:\BIOSTAT\processed\Ledalab\uncorrected")

    # Filenames inside each subject folder (change if yours differ)
    MANIFEST_FILENAME = "manifest.json"
    EVENTS_FILENAME = "Markers-events.csv"
    EVENTS_META = "Markers-events.json"
    GSR_FILENAME = "GSR.csv"
    META_FILENAME = "GSR.json"

    # Events names
    FIRST_EVENT = "Tutorial  starting baseline"
    LAST_EVENT = "Station scene  user told operator why they chose this spot  showing waypoint to assessment point"


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
        # Check if the events exist in the file
        if not ((events_df["value"] == FIRST_EVENT).any() and
                (events_df["value"] == LAST_EVENT).any()):
            print(f"[WARN] Missing events in {subject_dir}, skipping.")
            continue
        meta_path = subject_dir / META_FILENAME
        with open(meta_path, "r") as m:
            meta_data = json.load(m)
        sr = meta_data["resampled_fs"]
        s_start = 1 * sr
        s_end = 20 * sr
        # time vector
        n_samples = len(gsr_df['CH1'])
        time_sec = np.arange(n_samples) / sr  # seconds
        if time_sec.size == 0:
            print(f"[WARN] No valid time points for {subject_dir}, skipping.")
            continue
        # Using the name of the events get the sample in which every event is found
        start_idx = events_df["sample"][events_df.index[events_df["value"] == FIRST_EVENT]]
        if len(start_idx) == 1:
            first_idx = start_idx.iloc[0]
        else:
            first_idx = start_idx.iloc[-1]
        end_idx = events_df["sample"][events_df.index[events_df["value"] == LAST_EVENT]]
        if len(end_idx) == 1:
            last_idx = end_idx.iloc[0]
        else:
            last_idx = end_idx.iloc[-1]
        start_sample = max(0, first_idx - s_start)
        end_sample = min(last_idx + s_end, len(gsr_df) - 1)  # inclusive index
        # Events to use
        between_mask = (events_df["sample"] >= first_idx) & (events_df["sample"] <= last_idx)
        events_between = events_df.loc[between_mask].copy()
        # Convert event sample positions to seconds (same reference as time_sec)
        events_between = events_between.reset_index(drop=True)
        # placeholders; replace with your real arrays/values
        cut_data   = gsr_df["CH1"].to_numpy()
        eda_times  = time_sec
        latency_s  = start_sample
        latency_e  = end_sample
        eda_srate  = sr

        events_sub = events_between

        make_gsr_mat(
            out_path="GSR_subject01.mat",
            cut_data=cut_data,
            eda_times=eda_times,
            latency_s=latency_s,
            latency_e=latency_e,
            eda_srate=eda_srate,
            events_sub=events_sub,
        )
