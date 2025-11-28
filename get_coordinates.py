from pathlib import Path
import pandas as pd
import json

# Base folder
DATA_DIR = Path(r"P:\BIOSTAT\data_chunks")

# Target file names (without extensions)
targets = ["Markers-events", "rigidBody"]
coord_list = []

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
        coord_cols = [meta['columns'][4] , meta['columns'][6]]
        coords_xyz = coords[coord_cols]

        # Example operation: extract coordinates between two marker timestamps
        start_sample = events.loc[events['value'] == 'Station scene  starting free choice phase', 'sample'].values
        stop_sample = events.loc[events['value'] == 'Station scene  user told operator why they chose this spot  ' \
                                                  'showing waypoint to assessment point', 'sample'].values

        # Extract the coordinates for every event
        start_coord = coords_xyz.loc[start_sample]
        end_coord = coords_xyz.loc[stop_sample]

        coord_list.append([subj_dir.name, start_coord.values, end_coord.values])
# Deleting any elements that show up empty
cleaned = [
    item for item in coord_list
    if isinstance(item, list)
    and len(item) == 3
    and isinstance(item[0], str)
    and hasattr(item[1], "__len__") and len(item[1]) > 0
    and hasattr(item[2], "__len__") and len(item[2]) > 0
]

rows = []
for name, arr1, arr2 in cleaned:
    rows.append([name[0:7], *arr1[0], *arr2[0]])

coord_df = pd.DataFrame(rows, columns=['sub-ID', 'x_start', 'z_start', 'x_end', 'z_end'])

out_dir = r'P:\BIOSTAT\nudging\coordinates'
coord_df.to_csv(out_dir + r"\start_end_coordinates_chunks.csv", index=False)
