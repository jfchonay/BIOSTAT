from pathlib import Path
import pandas as pd
import re

# Base folder
folder = Path(r"/Users/josechonay/Documents/BIOSTAT_test_subs")

# Get all CSV files
csv_files = list(folder.glob("*.csv"))

# Group files by the first three digits
groups = {}
for f in csv_files:
    match = re.match(r"(\d{3})", f.stem)
    if match:
        prefix = match.group(1)
        groups.setdefault(prefix, []).append(f)

# Combine files within each group
output_folder = Path(r"/Users/josechonay/Documents/answers")
output_folder.mkdir(exist_ok=True)

for prefix, files in groups.items():
    dfs = [pd.read_csv(f) for f in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    output_file = output_folder / f"{prefix}_{files[0].name[5:-4]}.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined file: {output_file}")
