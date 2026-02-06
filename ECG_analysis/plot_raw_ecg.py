import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_ecg(time, ecg_signal, sub_dir, out_dir):
    plt.figure(figsize=(150, 4))
    plt.plot(time, ecg_signal, label='ECG Signal')
    plt.title(f'ECG Signal for {sub_dir.name}')
    plt.xlabel('Time (s)')
    plt.ylabel('ECG (mV)')
    plt.legend()
    plt.grid()
    plot_file = out_dir / f"{sub_dir.name}_ECG_plot.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved plot to {plot_file}")


def get_ecg_metadata(ecg_json_file):
    # Read ECG metadata from JSON file as in ECG_example.JSON and another example with missing data: ECG_example2.JSON
    with open(ecg_json_file, 'r') as f:
        ecg_metadata = json.load(f)
    nominal_srate = ecg_metadata.get("stats_original", {}).get("nominal_srate", None)
    estimated_srate = ecg_metadata.get("stats_original", {}).get("estimated_srate", None)
    n_samples = ecg_metadata.get("stats_original", {}).get("n_samples", None)
    duration_sec = ecg_metadata.get("stats_original", {}).get("duration_sec", None)
    return nominal_srate, estimated_srate, n_samples, duration_sec


def process_ecg(df, sub_dir, out_dir, ecg_metadata_df):
    # Check if required columns are present like in ECG_example.csv
    required_columns = ['time', 'ch1 (microvolts)']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column {col} in ECG data for {sub_dir}, skipping.")
            return

    # Get time and ECG signal
    time = df['time']
    ecg_signal = df['ch1 (microvolts)']

    if ecg_signal.empty:
        print(f"ECG signal is empty for {sub_dir}, skipping.")
        return

    # Get metadata from the ECG.JSON file if it exists
    ecg_json_file = sub_dir / "ECG.JSON"
    if ecg_json_file.exists():
        nominal_srate, estimated_srate, n_samples, duration_sec = get_ecg_metadata(ecg_json_file)
    else:
        nominal_srate = 0
        estimated_srate = 0
        n_samples = len(ecg_signal)
        duration_sec = time.iloc[-1] - time.iloc[0]

    # Update metadata dataframe
    ecg_metadata_df.loc[ecg_metadata_df['subject'] == sub_dir.name] = [sub_dir.name, "no", "complete", n_samples, duration_sec, nominal_srate, estimated_srate]

    # Plot ECG signal
    plot_ecg(time, ecg_signal, sub_dir, out_dir)


def main():
    if __name__ == "__main__":
        in_dir = Path(r"F:\BIOSTAT\1_raw_data_Jose")
        out_dir = Path(r"F:\BIOSTAT\Plots\raw_ECG")
        # Create a main dataframe to store metadata about missing or incomplete ECG files
        ecg_metadata_df = pd.DataFrame(columns=["subject", "biased", "status", "n_samples", "duration_sec", "nominal_srate", "estimated_srate"])

        # loop through the subject folders and take only the ECG .csv files
        for sub_dir in in_dir.glob("sub-*"):
            ecg_files = list(sub_dir.glob("ECG.csv"))
            if not ecg_files:
                print(f"No ECG files found in {sub_dir}, skipping.")
                # Add information about missing ECG file for this subject to the metadata dataframe
                ecg_metadata_df.loc[len(ecg_metadata_df)] = [sub_dir.name, "no", "missing", 0, 0, 0, 0]
                continue
            for ecg_file in ecg_files:
                print(f"Processing {ecg_file}")
                ecg_metadata_df.loc[len(ecg_metadata_df)] = [sub_dir.name, "no", "empty", 0, 0, 0, 0]
                try:
                    df = pd.read_csv(ecg_file)
                    process_ecg(df, sub_dir, out_dir, ecg_metadata_df)
                except ValueError:
                    print("File empty, skipping.")


        # Get file with list of biased subjects and update metadata dataframe accordingly
        biased_file = Path(r"F:\BIOSTAT\biased_subjects.txt")
        biased_subjects = set()
        if biased_file.exists():
            with open(biased_file, 'r') as f:
                for line in f:
                    biased_subjects.add(line.strip())
            for subject in biased_subjects:
                ecg_metadata_df.loc[ecg_metadata_df['subject'] == subject, 'biased'] = 'yes'
        else:
            print(f"Biased subjects file {biased_file} not found.")

        # Save the metadata dataframe to a CSV file
        metadata_dir = Path(r"F:\BIOSTAT")
        metadata_file = metadata_dir / "ECG_metadata_summary.csv"
        ecg_metadata_df.to_csv(metadata_file, index=False)
        print(f"Saved metadata summary to {metadata_file}")

        print("Done.")

if __name__ == "__main__":
    main()