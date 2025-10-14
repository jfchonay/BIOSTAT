# hmd_synced_xdf_export.py
# Usage:
#   python hmd_synced_xdf_export.py --in "C:/path/to/xdf_dir" --out "./export"
#
# What it does:
#   * Finds the "HMD" stream and uses its time stamps as the global reference.
#   * Produces HMD-synced CSVs for numeric streams: columns are time (s since HMD start) + channels.
#   * Produces event CSVs for marker-like streams (Markers, LookedAtObject/LookedAtObjects):
#       columns: event_ts (original XDF ts), onset_hmd_s, hmd_sample, value
#   * Sidecar JSON per stream with metadata, stats, and channel info.
#   * A manifest.json summarizing the export and the reference timeline.

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import pyxdf


# -------------------------- helpers --------------------------

SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def sanitize(s: str, fallback: str = "unnamed"):
    if s is None:
        return fallback
    s2 = SANITIZE_RE.sub("_", str(s)).strip("_")
    return s2 or fallback


def first_scalar(v, fallback=None) -> Optional[str]:
    """Unwrap nested list scalars like ['ECG'] -> 'ECG'."""
    if isinstance(v, list) and v:
        return first_scalar(v[0], fallback)
    if isinstance(v, (str, int, float)):
        return str(v)
    return fallback


def stream_name(stream) -> str:
    info = stream.get("info", {})
    nm = info.get("name", [""])[0] if isinstance(info.get("name", []), list) else info.get("name", "")
    return str(nm or "")


def stream_type(stream) -> str:
    info = stream.get("info", {})
    tp = info.get("type", [""])[0] if isinstance(info.get("type", []), list) else info.get("type", "")
    return str(tp or "")


def is_event_stream(stream) -> bool:
    """Identify event-like streams (Markers and LookedAtObject[s])."""
    nm = stream_name(stream).lower()
    tp = stream_type(stream).lower()
    # Typical conventions: type 'Markers', name containing 'marker', 'lookedatobject(s)'
    if "marker" in nm or "marker" in tp:
        return True
    if "lookedatobject" in nm or "lookedatobject" in tp:
        return True
    if "lookedatobjects" in nm or "lookedatobjects" in tp:
        return True
    # Heuristic: time_series is list of strings or list-of-1 strings
    ts = stream.get("time_series", [])
    if isinstance(ts, list) and ts:
        s0 = ts[0]
        if isinstance(s0, str):
            return True
        if isinstance(s0, (list, tuple)) and len(s0) == 1 and isinstance(s0[0], (str, bytes)):
            return True
    return False


def channel_meta_from_stream(stream) -> Tuple[List[str], List[Optional[str]]]:
    """
    Return (names, units) lists for channels, robust to XDF/LSL nested list/dict shapes.
    Works when desc/channels/channel are dicts or lists-of-one dicts.
    """
    info = stream.get("info", {})
    desc = info.get("desc", {})

    # unwrap list-of-one
    if isinstance(desc, list) and len(desc) == 1:
        desc = desc[0]

    channels = desc.get("channels", {}) if isinstance(desc, dict) else {}
    if isinstance(channels, list) and len(channels) == 1:
        channels = channels[0]

    ch = channels.get("channel", []) if isinstance(channels, dict) else []
    if isinstance(ch, list) and len(ch) == 1 and isinstance(ch[0], list):
        ch = ch[0]

    names, units = [], []
    if isinstance(ch, list) and ch:
        for i, c in enumerate(ch):
            if not isinstance(c, dict):
                continue
            nm = first_scalar(c.get("name"), f"ch{i+1}")
            un = first_scalar(c.get("unit"), None)
            names.append(sanitize(nm, f"ch{i+1}"))
            units.append(un)
        return names, units

    # fallback if no metadata
    ts = np.asarray(stream.get("time_series"))
    n_chan = ts.shape[1] if ts.ndim == 2 else 1
    return [f"ch{i+1}" for i in range(n_chan)], [None] * n_chan


def stream_stats(stream):
    ts = np.asarray(stream.get("time_stamps", []), dtype=float)
    info = stream.get("info", {})
    try:
        nominal = float(info.get("nominal_srate", 0.0))
    except Exception:
        nominal = 0.0
    eff = None
    if ts.size > 1:
        diffs = np.diff(ts)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        eff = float(1.0 / np.median(diffs)) if diffs.size else None
    return {
        "n_samples": int(ts.size),
        "first_time_stamp": (float(ts[0]) if ts.size else None),
        "last_time_stamp": (float(ts[-1]) if ts.size else None),
        "duration_sec": (float(ts[-1] - ts[0]) if ts.size >= 2 else 0.0),
        "nominal_srate": (nominal if nominal else None),
        "estimated_srate": (eff if eff else None),
    }


def nearest_indices_with_mask(src_ts: np.ndarray, tgt_ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map each target timestamp to nearest index in src_ts.
    Returns (indices, valid_mask) where valid_mask is True only for tgt_ts within [src_ts[0], src_ts[-1]].
    """
    if src_ts.size == 0 or tgt_ts.size == 0:
        return np.zeros(tgt_ts.size, dtype=int), np.zeros(tgt_ts.size, dtype=bool)

    idx = np.searchsorted(src_ts, tgt_ts, side="left")
    idx = np.clip(idx, 0, src_ts.size - 1)
    left = (idx > 0) & (np.abs(tgt_ts - src_ts[idx - 1]) <= np.abs(tgt_ts - src_ts[idx]))
    idx[left] -= 1
    valid = (tgt_ts >= src_ts[0]) & (tgt_ts <= src_ts[-1])
    return idx.astype(int), valid


def to_rfc3339(dt: datetime) -> str:
    # yyyy-mm-ddTHH:MM:SS.FFF (no TZ), MATLAB-like 'FFF'
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"


# -------------------------- core exporting --------------------------

BASE_ACQ_DATETIME = datetime(2025, 5, 20, 12, 0, 0)  # HMD = time zero by convention


def export_xdf_hmd_synced(xdf_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    streams, fileheader = pyxdf.load_xdf(str(xdf_path))

    # ---- find HMD reference stream ----
    hmd_idx = None
    for i, s in enumerate(streams):
        if stream_name(s).strip().lower() == "hmd":
            hmd_idx = i
            break
    if hmd_idx is None:
        raise RuntimeError(f"No HMD stream found in {xdf_path.name} (by exact name 'HMD').")

    ref_stream = streams[hmd_idx]
    ref_name = stream_name(ref_stream)
    ref_ts = np.asarray(ref_stream.get("time_stamps", []), dtype=float)
    if ref_ts.size == 0:
        raise RuntimeError(f"HMD stream has no time stamps in {xdf_path.name}.")

    ref_first = float(ref_ts[0])
    hmd_time = ref_ts - ref_first  # time since HMD start (s); this is the global time vector

    # manifest skeleton
    manifest = {
        "source_file": str(xdf_path),
        "export_time_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "file_header": fileheader,
        "reference": {
            "stream_name": ref_name,
            "base_acq_time": to_rfc3339(BASE_ACQ_DATETIME),
            "duration_sec": float(hmd_time[-1]) if hmd_time.size else 0.0,
            "n_samples": int(hmd_time.size),
        },
        "streams": []
    }

    # For per-stream timing descriptions relative to HMD
    onset_diff_seconds = {}
    acquisition_times = {}

    # ---- iterate and export each stream ----
    for k, stream in enumerate(streams):
        name = stream_name(stream)
        stype = stream_type(stream)
        base = f"{k:02d}-{sanitize(name, f'stream{k}')}"
        is_event = is_event_stream(stream)

        # original stream time stamps & first
        ts = np.asarray(stream.get("time_stamps", []), dtype=float)
        s_first = float(ts[0]) if ts.size else None

        # onset diff vs HMD and RFC3339 acq time
        if s_first is not None:
            diff_sec = float(s_first - ref_first)
            onset_diff_seconds[name] = diff_sec
            acquisition_times[name] = to_rfc3339(BASE_ACQ_DATETIME + timedelta(seconds=diff_sec))
        else:
            onset_diff_seconds[name] = None
            acquisition_times[name] = None

        # ---- EVENT STREAMS (Markers / LookedAtObjects): build onsets vs HMD ----
        if is_event:
            # flatten values
            vals = []
            for e in stream.get("time_series", []):
                if isinstance(e, (list, tuple)) and len(e) == 1:
                    vals.append(e[0])
                else:
                    vals.append(e)

            # sanitize values to keep alnum + spaces (similar to MATLAB regexprep)
            pattern = re.compile(r"[^a-zA-Z0-9]")
            vals_clean = [pattern.sub(" ", str(v)) for v in vals]

            # map event timestamps to nearest HMD sample index
            idx, valid = nearest_indices_with_mask(ref_ts, ts)
            onset_hmd_s = np.full(ts.shape, np.nan, dtype=float)
            hmd_sample = np.full(ts.shape, -1, dtype=int)
            if hmd_time.size and ts.size:
                onset_hmd_s[valid] = hmd_time[idx[valid]]
                hmd_sample[valid] = idx[valid]

            df = pd.DataFrame({
                "event_ts": ts,            # original event timestamp in XDF clock
                "onset_hmd_s": onset_hmd_s,  # seconds since HMD start (what you’ll use)
                "hmd_sample": hmd_sample,    # sample index on HMD grid
                "value": pd.Series(vals_clean, dtype="string"),
            })

            csv_path = out_dir / f"{base}-events.csv"
            json_path = out_dir / f"{base}-events.json"
            df.to_csv(csv_path, index=False)

            # sidecar
            sidecar = {
                "name": name,
                "type": stype,
                "kind": "events",
                "stats": stream_stats(stream),
                "reference": {
                    "aligned_to": ref_name,
                    "timeline": "HMD",
                    "note": "onset_hmd_s and hmd_sample are aligned to HMD start at t=0s",
                },
                "columns": ["event_ts", "onset_hmd_s", "hmd_sample", "value"]
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, ensure_ascii=False, indent=2)

            manifest["streams"].append({
                "index": k,
                "name": name,
                "type": stype,
                "role": "events",
                "csv": str(csv_path),
                "json": str(json_path),
                "n_rows": int(len(df)),
                "columns": list(df.columns),
            })
            continue  # done with events

        # ---- NUMERIC STREAMS: resample/alignment to HMD timeline ----
        data = stream.get("time_series", [])
        arr = np.asarray(data)

        # Normalize to (n_samples, n_channels)
        if arr.ndim == 1:  # (n_samples,)
            arr = arr.reshape(-1, 1)
        elif arr.shape[0] < arr.shape[1]:
            arr = arr.T

        # channel names + units
        names, units = channel_meta_from_stream(stream)
        if len(names) != arr.shape[1]:
            names = [f"ch{i+1}" for i in range(arr.shape[1])]
            units = [None] * arr.shape[1]
        headers = [f"{n} ({u})" if u else n for n, u in zip(names, units)]

        # For each ref time, take nearest sample from this stream (inside coverage), else NaN
        idx, valid = nearest_indices_with_mask(ts, ref_ts)
        out = np.full((ref_ts.size, arr.shape[1]), np.nan, dtype=float)
        if ts.size and ref_ts.size and arr.shape[0] > 0:
            out[valid, :] = arr[idx[valid], :]

        df = pd.DataFrame(out, columns=headers)
        df.insert(0, "time", hmd_time)  # 0..HMD_duration (seconds)

        csv_path = out_dir / f"{base}-hmdsynced.csv"
        json_path = out_dir / f"{base}-hmdsynced.json"
        df.to_csv(csv_path, index=False)

        # sidecar
        sidecar = {
            "name": name,
            "type": stype,
            "kind": "signal_hmd_synced",
            "channels": [{"name": n, "unit": u} for n, u in zip(names, units)],
            "stats_original": stream_stats(stream),
            "reference": {
                "aligned_to": ref_name,
                "timeline": "HMD",
                "time_column": "time",  # seconds since HMD start
                "coverage_note": "Samples outside the stream’s original coverage are NaN after alignment.",
            },
            "columns": ["time"] + headers
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)

        manifest["streams"].append({
            "index": k,
            "name": name,
            "type": stype,
            "role": "signal",
            "csv": str(csv_path),
            "json": str(json_path),
            "n_rows": int(len(df)),
            "columns": list(df.columns),
        })

    # enrich manifest with timing maps
    manifest.update({
        "onset_diff_seconds_vs_HMD": onset_diff_seconds,   # {stream_name: seconds}
        "acquisition_time_rfc3339": acquisition_times      # {stream_name: 'YYYY-mm-ddTHH:MM:SS.FFF'}
    })

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# -------------------------- CLI --------------------------

def main():
    if __name__ == "__main__":
        in_dir = Path(r"/Users/josechonay/Documents/BIOSTAT_data")
        out_dir = Path(r"/Users/josechonay/Documents/BIOSTAT_derivatives")
        for xdf_file in in_dir.glob("*.xdf"):
            sub = out_dir / sanitize(xdf_file.stem)
            print(f"Exporting {xdf_file} -> {sub}")
            export_xdf_hmd_synced(xdf_file, sub)
        print("Done.")


if __name__ == "__main__":
    main()
