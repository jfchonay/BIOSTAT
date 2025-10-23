from pathlib import Path
from turtledemo.penrose import start

import pandas as pd
import json

# Base folder
DATA_DIR = Path(r"P:\BIOSTAT\raw_data")

# Target file names (without extensions)
targets = ["Markers-events", "rigidBody"]
start_list = []
end_list = []

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

        # We just want the x, y, z rigid columns
        coord_cols = meta['columns'][4:7]
        coords_xyz = coords[coord_cols]

        # Example operation: extract coordinates between two marker timestamps
        start_sample = events.loc[events['value'] == 'Station scene  starting free choice phase', 'sample'].values
        stop_sample = events.loc[events['value'] == 'Station scene  user told operator why they chose this spot  ' \
                                                  'showing waypoint to assessment point', 'sample'].values

        # Extract the coordinates for every event
        start_coord = coords_xyz.loc[start_sample]
        end_coord   = coords_xyz.loc[stop_sample]

        start_list.append(start_coord.values)
        end_list.append(end_coord.values)
# Deleting any elements that show up empty
clean_start = [i_s[0] for i_s in start_list if len(i_s) > 0]
clean_end = [i_e[0] for i_e in end_list if len(i_e) > 0]
out_dir = r'P:\BIOSTAT\nudging\coordinates'
start_df = pd.DataFrame(clean_start, columns=coord_cols)
end_df   = pd.DataFrame(clean_end, columns=coord_cols)

start_df.to_csv(out_dir + r"\start_coordinates.csv", index=False)
end_df.to_csv(out_dir + r"\end_coordinates.csv", index=False)

