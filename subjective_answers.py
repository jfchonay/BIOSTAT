from pathlib import Path
import pandas as pd

# Base folder
DATA_DIR = Path(r"P:\BIOSTAT\subjective_answers")

for subj_dir in DATA_DIR.iterdir():
    if not subj_dir.is_dir():
        continue

    print(f"\nSubject: {subj_dir.name}")

    # Dictionary to hold loaded data for each target
    loaded = {}
