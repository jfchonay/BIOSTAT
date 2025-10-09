import re
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pyxdf


# ---------- helpers to mimic FieldTrip-ish structs ----------

def _first_timestamp(x):
    """Return the first time stamp (float seconds) or np.nan."""
    ts = x.get("time_stamps")
    return float(ts[0]) if ts is not None and len(ts) else np.nan


def stream_to_ft(xdf_stream):
    """
    Convert one pyxdf stream dict to a FieldTrip-style dict:
      - hdr: {Fs, nChans, label, FirstTimeStamp, ...}
      - trial: [np.ndarray shape (n_chans, n_samples)]
      - time:  [np.ndarray shape (n_samples,)]
    """
    info = xdf_stream.get("info", {})
    chn = info.get("desc", {}).get("channels", {}).get("channel", [])
    labels = [c.get("label", ["ch"])[0] for c in chn] if isinstance(chn, list) else []

    data = xdf_stream.get("time_series")
    ts = xdf_stream.get("time_stamps")

    # normalize shapes: FieldTrip is (n_chan, n_samp)
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # (1, n_samp)
    elif arr.shape[0] < arr.shape[1]:
        # most xdf streams already are (n_samp, n_chan) -> transpose
        # pyxdf usually returns (n_samples, n_channels)
        arr = arr.T  # (n_chans, n_samp)

    # sample rate: prefer nominal_srate; fall back to median diff
    Fs = None
    try:
        Fs = float(info.get("nominal_srate", 0.0))
    except Exception:
        Fs = 0.0
    if not Fs or Fs == 0.0:
        if ts is not None and len(ts) > 1:
            Fs = 1.0 / np.median(np.diff(ts))
        else:
            Fs = np.nan

    if not labels or len(labels) != arr.shape[0]:
        # build generic labels if missing or mismatched
        labels = [f"ch{ii+1}" for ii in range(arr.shape[0])]

    ft = {
        "hdr": {
            "Fs": Fs,
            "nChans": arr.shape[0],
            "label": labels,
            "FirstTimeStamp": _first_timestamp(xdf_stream),
            "name": info.get("name", [""])[0] if isinstance(info.get("name", []), list) else info.get("name"),
            "type": info.get("type", [""])[0] if isinstance(info.get("type", []), list) else info.get("type"),
        },
        # FieldTrip has a list of trials; here we mirror your single continuous trial
        "trial": [arr],
        "time": [np.asarray(ts, dtype=float) if ts is not None else np.arange(arr.shape[1]) / Fs],
    }
    return ft


def xdf_inspect(streams):
    """
    Return (streamNames, channelMetaData) similar to your MATLAB helper.
    - streamNames: list of names
    """
    names = []
    ch_meta = []
    for s in streams:
        info = s.get("info", {})
        name = info.get("name", [""])[0] if isinstance(info.get("name", []), list) else info.get("name", "")
        names.append(name)
    return names


def stream_to_events(event_stream, ref_time_stamps):
    """
    Convert an event stream with time_stamps to a structure with sample indices
    aligned to the reference time grid (ref_time_stamps).
    Returns a dict with 'sample' (np.array of ints) and 'time' (np.array of floats).
    """
    evt_ts = np.asarray(event_stream.get("time_stamps"), dtype=float)
    ref_ts = np.asarray(ref_time_stamps, dtype=float)

    if evt_ts.size == 0:
        return {"sample": np.array([], dtype=int), "time": np.array([], dtype=float)}

    # map each event timestamp to nearest sample index in the reference stream
    idx = np.searchsorted(ref_ts, evt_ts, side="left")
    idx = np.clip(idx, 0, ref_ts.size - 1)

    # if the closer neighbor is on the left, pick it
    left_is_closer = (idx > 0) & (
        np.abs(evt_ts - ref_ts[idx - 1]) <= np.abs(evt_ts - ref_ts[idx])
    )
    idx[left_is_closer] = idx[left_is_closer] - 1

    return {"sample": idx.astype(int), "time": ref_ts[idx]}


# ---------- configuration ----------

DATA_DIR = r"P:\BIOSTAT\raw_data"
EVENT_STREAM_NAME = "ExperimentMarkerStream"
POLAR_STREAM_NAME = "PolarBand"
SHIMMER_STREAM_NAME = "Shimmer_GSR"
MOTION_STREAM_NAME = "HMD"
VIVE_STREAM_NAME = "ViveProEye"
LOOKEDAT_STREAM_NAME = "LookedAtObject"

# Base acquisition date/time (matches your MATLAB vectors)
BASE_ACQ_DATETIME = datetime(2025, 5, 20, 12, 0, 0)  # YYYY,MM,DD,HH,MM,SS


# ---------- main ----------

def main():
    # list all files (non-dirs), sorted for deterministic indexing
    all_files = sorted(
        [p for p in Path(DATA_DIR).iterdir() if p.is_file()],
        key=lambda p: p.name.lower(),
    )

    for i_sub in all_files:

        xdf_path = Path(i_sub)

        # 1) load and inspect XDF
        streams, _ = pyxdf.load_xdf(xdf_path)
        stream_names = xdf_inspect(streams)

        # find streams by name (order can vary)
        def find_stream(name):
            for k, s in enumerate(streams):
                info = s.get("info", {})
                sname = info.get("name", [""])[0] if isinstance(info.get("name", []), list) else info.get("name", "")
                if sname == name:
                    return k
            return None

        idx_event = find_stream(EVENT_STREAM_NAME)
        idx_polar = find_stream(POLAR_STREAM_NAME)
        idx_shim = find_stream(SHIMMER_STREAM_NAME)
        idx_motion = find_stream(MOTION_STREAM_NAME)
        idx_vive = find_stream(VIVE_STREAM_NAME)
        idx_looked = find_stream(LOOKEDAT_STREAM_NAME)

        missing = [(EVENT_STREAM_NAME, idx_event),
                   (POLAR_STREAM_NAME, idx_polar),
                   (SHIMMER_STREAM_NAME, idx_shim),
                   (MOTION_STREAM_NAME, idx_motion),
                   (VIVE_STREAM_NAME, idx_vive)]
        missing_names = [name for name, idx in missing if idx is None]
        if missing_names:
            print(f"[warn] missing streams: {missing_names} (continuing with available ones)")

        # 2) convert to FieldTrip-like dicts
        polar_ft = stream_to_ft(streams[idx_polar]) if idx_polar is not None else None
        shimmer_ft = stream_to_ft(streams[idx_shim]) if idx_shim is not None else None
        motion_ft = stream_to_ft(streams[idx_motion]) if idx_motion is not None else None
        vive_ft = stream_to_ft(streams[idx_vive]) if idx_vive is not None else None

        # Event stream handling (rebuild cells, sanitize, align to Polar)
        event_struct = []
        if idx_event is not None and polar_ft is not None:
            evt_stream = streams[idx_event].copy()

            # The MATLAB code wraps each element into a cell, then uses regex to sanitize.
            # In XDF via LSL markers, time_series is often a list of lists/strings.
            raw_series = evt_stream.get("time_series", [])
            # Flatten one level so we end with a list of string-like entries.
            flat = []
            for e in raw_series:
                # typical shapes: ['MarkerName'] or e could already be a string
                if isinstance(e, (list, tuple)) and len(e) == 1:
                    flat.append(e[0])
                else:
                    flat.append(e)
            evt_stream["time_series"] = flat

            # align event timestamps to Polar samples
            out_events = stream_to_events(evt_stream, streams[idx_polar]["time_stamps"])

            # build eventStruct array of dicts
            pattern = re.compile(r"[^a-zA-Z0-9]")
            for Ei, val in enumerate(evt_stream["time_series"]):
                # convert to string and sanitize like regexprep(..., '[^a-zA-Z0-9]', ' ')
                sval = str(val)
                sval = pattern.sub(" ", sval)

                event_struct.append({
                    "type": "Marker",
                    "sample": int(out_events["sample"][Ei]),
                    "offset": 0,
                    "value": sval,
                    "duration": 0
                })

        # 3) Save time sync info (differences relative to Polar first timestamp)
        def onset_diff(ft_like):
            if ft_like is None or not np.isfinite(ft_like["hdr"]["FirstTimeStamp"]):
                return np.nan
            return float(ft_like["hdr"]["FirstTimeStamp"] - polar_ft["hdr"]["FirstTimeStamp"])

        shimmer_onset_diff = onset_diff(shimmer_ft)
        motion_onset_diff = onset_diff(motion_ft)
        vive_onset_diff = onset_diff(vive_ft)

        # RFC3339-like (no timezone) with milliseconds, reproducing MATLAB:
        #   motionOnset = [YYYY,MM,DD,HH,MM, MotionOnsetDiff];
        #   datestr(datenum(...), 'yyyy-mm-ddTHH:MM:SS.FFF')
        def to_rfc3339(base_dt, extra_seconds):
            if not np.isfinite(extra_seconds):
                return None
            dt = base_dt + timedelta(seconds=float(extra_seconds))
            # format: yyyy-mm-ddTHH:MM:SS.FFF (milliseconds)
            return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"

        eeg_acq_time = to_rfc3339(BASE_ACQ_DATETIME, 0.0)  # your MATLAB had 0 seconds
        motion_acq_time = to_rfc3339(BASE_ACQ_DATETIME, motion_onset_diff)
        shimmer_acq_time = to_rfc3339(BASE_ACQ_DATETIME, shimmer_onset_diff)
        vive_acq_time = to_rfc3339(BASE_ACQ_DATETIME, vive_onset_diff)

        # ----- example: print a brief summary (or save as needed) -----
        print("Streams found:", stream_names)
        print("Polar Fs:", None if polar_ft is None else polar_ft["hdr"]["Fs"])
        print("n events:", len(event_struct))
        if event_struct:
            print("first 3 events:", event_struct[:3])

        print("Acquisition times (RFC3339-like, no TZ):")
        print("  EEG   :", eeg_acq_time)
        print("  Motion:", motion_acq_time)
        print("  Shimm :", shimmer_acq_time)
        print("  Vive  :", vive_acq_time)

        # TODO: if you want to persist, you can np.savez or json.dump the structs here.
        # For example:
        # out_dir = Path(DATA_DIR) / "processed"
        # out_dir.mkdir(exist_ok=True)
        # np.savez(out_dir / (Path(xdf_path).stem + "_summary.npz"),
        #          polar_hdr=polar_ft["hdr"], events=event_struct, ...)


if __name__ == "__main__":
    main()