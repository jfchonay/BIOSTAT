# Load an .xdf file using the pylsl library and print the stream information. There are 6 streams: EKG, EDA, Eyetracker, HMD, Events and Looked at Objects. 
# The order of streams may be different. A file may be non existent or there may be streams missing or empty. The code should handle these cases gracefully.
# The code should print the name of each stream, the number of channels, the sampling rate and the number of samples. 
# If a stream is missing or empty, it should print a message indicating that. Use a F:/BIOSTAT/0_source_data/033_p1d_0613.xdf file as an example.

from asyncio import streams
import os
import numpy as np
import pyxdf

# This function loads an XDF file and prints information about each stream. It handles cases where the file does not exist, streams are missing, or streams are empty.
# It returns all streams with time series data and time stamps for further processing if needed.
def load_xdf_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    # using big, more precise integer for the timestamps to avoid overflow issues with large precision timestamps.
    #streams, header = pyxdf.load_xdf(file_path, select_streams=None, synchronize_clocks=True, dejitter_timestamps=True, handle_clock_resets=True)

    try:
        streams, time_stamps = pyxdf.load_xdf(file_path, select_streams=None, synchronize_clocks=True, dejitter_timestamps=False, handle_clock_resets=True)
    except Exception as e:
        print(f"Error loading XDF file: {e}")
        return

    if not streams:
        print("No streams found in the XDF file.")
        return

    for stream in streams:
        name = stream['info']['name'][0]
        channels = stream['info']['channel_count'][0]
        sampling_rate = stream['info']['nominal_srate'][0]
        samples = len(stream['time_series'])

        if channels == 0 or samples == 0:
            print(f"Stream '{name}' is missing or empty.")
        else:
            print(f"Stream Name: {name}")
            print(f"Number of Channels: {channels}")
            print(f"Sampling Rate: {sampling_rate} Hz")
            print(f"Number of Samples: {samples}")
            print("-" * 30)

    # match streams and their timestamps by name to one of the streams: PolarBand, Shimmer_GSR, HMD, ViveProEye, LookedAtObject, ExperimentMarkerStream
    # each stream is a dictionary with keys 'info', 'time_series' and 'time_stamps'. The 'info' key contains a dictionary with the stream information, including the name of the stream. The 'time_series' key contains the actual data of the stream, and the 'time_stamps' key contains the timestamps for each sample in the stream. We can create a dictionary that maps the name of each stream to its corresponding time series data and timestamps for easier access later on.
    stream_dict = {}
    for stream in streams:
        name = stream['info']['name'][0]
        stream_dict[name] = {
            'time_series': stream['time_series'],
            'time_stamps': stream['time_stamps']
        }

    return stream_dict

def main():
    file_path = "F:/BIOSTAT/0_source_data/114_p0e_0726.xdf"
    stream_dict = load_xdf_file(file_path)
    # Print the first and the last timestamp of each stream as an example of the timestamps. This will also show if the timestamps are aligned across streams or not.
    for name, stream in stream_dict.items():
        if 'time_stamps' in stream and len(stream['time_stamps']) > 0:
            print(f"First timestamp of stream '{name}': {stream['time_stamps'][0]}")
            print(f"Last timestamp of stream '{name}': {stream['time_stamps'][-1]}")
        else:
            print(f"Stream '{name}' has no timestamps.")
    # # Take the smallest timestamp and use it as time 0 for all streams. This will allow us to align the streams based on their timestamps.
    # if stream_dict:
    #     min_time = min([min(stream['time_stamps']) for stream in stream_dict.values()])
    #     for stream in stream_dict.values():
    #         stream['time_stamps'] -= min_time
    # # Get the polar band stream and print the first 5 samples of the time series data and timestamps as an example
    # if 'PolarBand' in stream_dict:
    #     polar_band_stream = stream_dict['PolarBand']
    #     print("First 5 samples of PolarBand time series data:")
    #     print(polar_band_stream['time_series'][:5])
    #     print("First 5 timestamps of PolarBand stream:")
    #     print(polar_band_stream['time_stamps'][:5])
    


if __name__ == "__main__":
    main()