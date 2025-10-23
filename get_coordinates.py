from pathlib import Path
import pandas as pd
import json

# Base folder
DATA_DIR = Path("/Users/josechonay/Documents/BIOSTAT_data")

# Target file names (without extensions)
targets = ["Markers-events", "rigidBody"]

# Iterate over each subject folder
for subj_dir in DATA_DIR.iterdir():
    if not subj_dir.is_dir():
        continue

    print(f"\nSubject: {subj_dir.name}")

    # Dictionary to hold loaded data for each target
    loaded = {}

    for target in targets:
        csv_path = subj_dir / f"{target}.csv"
        json_path = subj_dir / f"{target}.json"

        if csv_path.exists() and json_path.exists():
            # Load CSV
            df = pd.read_csv(csv_path)

            # Load JSON
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            loaded[target] = {"csv": df, "json": meta}

            print(f"  Loaded {target}: {df.shape[0]} rows, {len(meta)} metadata keys")
        else:
            print(f"  Missing one of: {target}.csv / {target}.json")

    if "Markers-events" in loaded and "rigidBody" in loaded:
        events = loaded["Markers-events"]["csv"]
        coords = loaded["rigidBody"]["csv"]

        # Example operation: extract coordinates between two marker timestamps
        start_time = events.loc[events['value'] == 'Station scene  starting free choice phase', 'sample'].values
        stop_time = events.loc[events['value'] == 'Station scene  user told operator why they chose this spot  ' \
                                                  'showing waypoint to assessment point', 'sample'].values

