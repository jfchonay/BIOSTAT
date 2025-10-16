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

_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")

def _flatten_marker_values(time_series):
    """Turn nested marker like [['Start'], ['Stop']] into ['Start','Stop']."""
    flat = []
    for e in time_series:
        if isinstance(e, (list, tuple)) and len(e) == 1:
            flat.append(e[0])
        else:
            flat.append(e)
    return flat

def trim_to_hmd_window(stream_ts: np.ndarray,
                       stream_data: np.ndarray,
                       hmd_ts: np.ndarray,
                       clip_end: bool = True):
    """
    Trim a stream's native samples to the HMD window without resampling.

    stream_ts : (N,) native timestamps of the stream (float seconds)
    stream_data : (N,C) numeric data aligned to stream_ts
    hmd_ts : (M,) HMD timestamps (reference clock)
    clip_end : if True, drop samples after HMD end; if False, keep them

    Returns:
        t_rel : (K,) time in seconds since HMD start (0 at HMD first sample)
        y     : (K,C) trimmed data (no NaN padding)
        mask  : (N,) bool mask of kept samples in original stream
    """
    if hmd_ts.size == 0 or stream_ts.size == 0:
        return np.empty((0,), float), stream_data[:0], np.zeros(stream_ts.size, bool)

    start = hmd_ts[0]
    end   = hmd_ts[-1]
    if clip_end:
        keep = (stream_ts >= start) & (stream_ts <= end)
    else:
        keep = (stream_ts >= start)

    if stream_data.ndim == 1:
        stream_data = stream_data.reshape(-1, 1)

    t_rel = stream_ts[keep] - start
    y     = stream_data[keep, :]
    return t_rel, y, keep

def _ceil_indices(ref_ts: np.ndarray, evt_ts: np.ndarray) -> np.ndarray:
    """
    MATLAB's find(ref_ts >= t, 1, 'first') for each t:
      -> np.searchsorted(ref_ts, t, side='left')
    If t > ref_ts[-1], we clip to the last sample (MATLAB would return empty).
    """
    idx = np.searchsorted(ref_ts, evt_ts, side="left")
    idx = np.clip(idx, 0, max(0, ref_ts.size - 1))
    return idx.astype(int)

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
    # time_series is list of strings or list-of-1 strings
    ts = stream.get("time_series", [])
    if isinstance(ts, list) and ts:
        s0 = ts[0]
        if isinstance(s0, str):
            return True
        if isinstance(s0, (list, tuple)) and len(s0) == 1 and isinstance(s0[0], (str, bytes)):
            return True
    return False

def stream_to_events_hmd(in_streams, hmd_time_stamps: np.ndarray):
    """
    Accepts either a single stream dict or a list of stream dicts.
    Returns a sorted list of event dicts with fields:
      type, timestamp, sample, offset, duration, value
    """
    if not isinstance(in_streams, (list, tuple)):
        in_streams = [in_streams]

    out_events = []
    for s in in_streams:
        # original event times
        evt_ts = np.asarray(s.get("time_stamps", []), dtype=float)
        # ceil indices on HMD grid (first index with HMD_ts >= evt_ts)
        if hmd_time_stamps is None or hmd_time_stamps.size == 0:
            idx = np.zeros_like(evt_ts, dtype=int)
        else:
            idx = _ceil_indices(hmd_time_stamps, evt_ts)

        #'value'
        vals = _flatten_marker_values(s.get("time_series", []))
        # length safety
        if len(vals) != evt_ts.size:
            # trim/pad to match (rare; keeps behavior defined)
            n = min(len(vals), evt_ts.size)
            vals = vals[:n]
            evt_ts = evt_ts[:n]
            idx = idx[:n]

        # sanitize
        vals_clean = [_ALNUM_RE.sub(" ", str(v)) for v in vals]

        # event type from stream.info.type if present; default 'Marker'
        stype = s.get("info", {}).get("type", [""])[0] if isinstance(s.get("info", {}).get("type", []), list) \
                else s.get("info", {}).get("type", "") or "Marker"

        # build event dicts
        for t, i, v in zip(evt_ts, idx, vals_clean):
            out_events.append({
                "type": str(stype),
                "timestamp": float(t),   # original XDF timestamp
                "sample": int(i),        # HMD sample index (ceil)
                "offset": 0,
                "duration": 0,
                "value": v
            })

    # sort by original timestamp
    out_events.sort(key=lambda e: e["timestamp"])
    return out_events


def channel_meta_from_stream(stream) -> Tuple[List[str], List[Optional[str]]]:
    """
    Return (names, units) lists for channels, from XDF nested list/dict shapes.
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
            nm = first_scalar(c.get("label"), f"ch{i+1}")
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
        nominal = float(info.get("nominal_srate", 0.0)[0])
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
        "id": xdf_path.stem,
        "greenery": xdf_path.stem[5],
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
        base = f"{sanitize(stype, f'stream{k}')}"

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
        if is_event_stream(stream):
            event_struct = stream_to_events_hmd(stream, ref_ts)  # ref_ts is HMD timestamps
            # onset_hmd_s = hmd_time[sample]
            samples = np.array([e["sample"] for e in event_struct], dtype=int)
            # guard against empty
            onset_hmd_s = (hmd_time[samples] if samples.size else np.array([], dtype=float))

            df = pd.DataFrame({
                "onset": onset_hmd_s,  # seconds since HMD start
                "sample": samples,  # HMD sample index
                "type": [e["type"] for e in event_struct],
                "value": [e["value"] for e in event_struct],
                "offset": [e["offset"] for e in event_struct],
                "duration": [e["duration"] for e in event_struct],
            })
            df.to_csv(out_dir / f"{base}-events.csv", index=False)

            # sidecar
            sidecar_events = {
                "name": name,
                "type": stype,
                "kind": "events",
                "stats": stream_stats(stream),
                "reference": {
                    "aligned_to": ref_name,
                    "timeline": "HMD",
                    "note": "onset_hmd_s and hmd_sample are aligned to HMD start at t=0s",
                },
                "columns": ["onset", "sample", "value", "offset", "duration"]
            }
            # Optionally also save the MATLAB-like struct as JSON:
            with open(out_dir / f"{base}-events.json", "w", encoding="utf-8") as f:
                json.dump(sidecar_events, f, ensure_ascii=False, indent=2)

            manifest["streams"].append({
                "index": k,
                "name": name,
                "type": stype,
                "role": "events",
                "csv": str(out_dir / f"{base}-events.csv"),
                "json": str(out_dir / f"{base}-events.json"),
                "n_rows": int(len(df)),
                "columns": list(df.columns),
            })
            continue  # done with events

        data = np.asarray(stream.get("time_series", []))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.shape[0] < data.shape[1]:
            data = data.T  # (n_samples, n_channels)

        ts = np.asarray(stream.get("time_stamps", []), dtype=float)

        # Trim to HMD window; remove leading "empty" region automatically
        t_rel, y, keep = trim_to_hmd_window(ts, data, ref_ts, clip_end=True)

        # Build DataFrame with ONLY valid rows, time starts at 0 (HMD start)
        names, units = channel_meta_from_stream(stream)
        if len(names) != y.shape[1]:
            names = [f"ch{i + 1}" for i in range(y.shape[1])]
            units = [None] * y.shape[1]
        headers = [f"{n} ({u})" if u else n for n, u in zip(names, units)]

        df = pd.DataFrame(y, columns=headers)
        df.to_csv(out_dir / f"{base}.csv", index=False)
        json_path = out_dir / f"{base}.json"

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
                "coverage_note": "Samples outside the stream’s original coverage are NaN after alignment.",
            },
            "columns": headers
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)

        manifest["streams"].append({
            "index": k,
            "name": name,
            "type": stype,
            "role": "signal",
            "csv": str(out_dir / f"{base}.csv"),
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
        in_dir = Path(r"P:\BIOSTAT\lsl_full")
        out_dir = Path(r"P:\BIOSTAT\raw_data")
        for xdf_file in in_dir.glob("*.xdf"):
            sub = out_dir / f"sub-{sanitize(xdf_file.stem)[0:3]}"
            print(f"Exporting {xdf_file} -> {sub}")
            export_xdf_hmd_synced(xdf_file, sub)
        print("Done.")


if __name__ == "__main__":
    main()
