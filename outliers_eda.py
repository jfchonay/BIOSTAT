import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

def robust_z(x):
    """Robust z-score based on median and MAD."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def simple_peak_clean(x, z_thresh, pad, max_interpolate_samples=None):
    """
    x : 1D array-like (GSR signal)
    z_thresh : threshold on |robust_z| to define 'extreme peaks'
    pad : number of samples to extend around each detected peak
    max_interpolate_samples : if not None, only interpolate gaps with
                              length <= max_interpolate_samples

    Returns
    -------
    x_clean : np.ndarray
        Signal with extreme peaks interpolated.
    artifact_mask : np.ndarray[bool]
        True where samples were considered artifacts (before interpolation).
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # 1) robust z-score on amplitude
    z = robust_z(x)

    # 2) basic artifact mask: extreme amplitude or non-finite
    artifact = (np.abs(z) > z_thresh) | ~np.isfinite(x)

    # 3) expand mask by `pad` samples on each side (no external deps)
    if pad > 0 and artifact.any():
        idx = np.where(artifact)[0]
        for i in idx:
            start = max(0, i - pad)
            end = min(n, i + pad + 1)
            artifact[start:end] = True

    # 4) set artifacts to NaN
    x_clean = x.copy()
    x_clean[artifact] = np.nan

    # 5) interpolate
    s = pd.Series(x_clean)
    x_interp = s.interpolate(
        limit=max_interpolate_samples,
        limit_direction="both"
    ).to_numpy()

    return x_interp, artifact


ROOT_DIR = Path(r"P:\BIOSTAT\resampled\weird_channels")
GSR_FILENAME = "GSR.csv"
MANIFEST_FILENAME = "manifest.json"

for subject_dir in ROOT_DIR.iterdir():
    gsr_csv_path = subject_dir / GSR_FILENAME
    df = pd.read_csv(gsr_csv_path)
    df_clean = pd.DataFrame()
    x_raw = df["CH1"].values
    # clean using robust amplitude-based peak detector
    x_clean, artifact_mask = simple_peak_clean(
        x_raw,
        z_thresh=10.0,  # stricter or looser as needed
        pad=5,  # grow a bit around each spike
        max_interpolate_samples=None  # interpolate all gaps
    )
    df["CH1_clean"] = x_clean
    df["artifact"] = artifact_mask.astype(int)
