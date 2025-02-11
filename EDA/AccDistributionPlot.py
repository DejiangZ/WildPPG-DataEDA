import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def load_wildppg_participant(path):
    """
    Loads the data of a WildPPG participant and cleans it to receive nested dictionaries.
    """
    loaded_data = scipy.io.loadmat(path)

    # Clean id field
    loaded_data['id'] = loaded_data['id'][0]

    # Clean notes field
    if len(loaded_data['notes']) == 0:
        loaded_data['notes'] = ""
    else:
        loaded_data['notes'] = loaded_data['notes'][0]

    # Clean each body location's data
    for bodyloc in ['sternum', 'head', 'wrist', 'ankle']:
        bodyloc_data = dict()  # Store cleaned data
        if bodyloc in loaded_data:
            sensors = loaded_data[bodyloc][0].dtype.names  # Sensor names
            for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
                bodyloc_data[sensor_name] = dict()
                field_names = sensor_data[0][0].dtype.names  # Sensor field names
                for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                    bodyloc_data[sensor_name][sensor_field] = field_data[0]
                    if sensor_field == 'fs':  # Extract scalar for sampling rate
                        bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
            loaded_data[bodyloc] = bodyloc_data  # Replace with cleaned data
        else:
            loaded_data[bodyloc] = None  # If no data, set as None

    return loaded_data


def calculate_acceleration_magnitude(acc_x, acc_y, acc_z):
    """
    Calculate acceleration magnitude.
    """
    return np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))


def segment_signal(signal, window_size, stride):
    """
    Segment signal into overlapping windows with specified stride.

    Args:
        signal: Input signal array
        window_size: Number of samples in each window
        stride: Number of samples between window starts

    Returns:
        List of windowed signal segments
    """
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


# File paths
data_path = r'G:\My Drive\Dataset\WildPPG'
mat_path = 'WildPPG_Part_an0.mat'
file_path = os.path.join(data_path, mat_path)

# Load data
participant_data = load_wildppg_participant(file_path)

# Process wrist data with 8-second windows and 2-second stride
if participant_data['wrist']:
    acc_x = participant_data['wrist']['acc_x']['v']
    acc_y = participant_data['wrist']['acc_y']['v']
    acc_z = participant_data['wrist']['acc_z']['v']

    # Ensure signals have consistent lengths
    min_length = min(len(acc_x), len(acc_y), len(acc_z))
    acc_x, acc_y, acc_z = acc_x[:min_length], acc_y[:min_length], acc_z[:min_length]

    # Parameters for windowing
    sampling_rate = participant_data['wrist']['acc_x']['fs']  # Get sampling rate (e.g., 128 Hz)
    window_size = 8 * sampling_rate  # 8 seconds window
    stride = 2 * sampling_rate       # 2 seconds stride

    # Segment signals into windows
    acc_x_windows = segment_signal(acc_x, window_size, stride)
    acc_y_windows = segment_signal(acc_y, window_size, stride)
    acc_z_windows = segment_signal(acc_z, window_size, stride)

    # Compute acceleration magnitude for each window
    acc_magnitude_windows = calculate_acceleration_magnitude(acc_x_windows, acc_y_windows, acc_z_windows)

    # Compute statistics for each window
    mean_vals = np.mean(acc_magnitude_windows, axis=1)
    std_vals = np.std(acc_magnitude_windows, axis=1)
    max_vals = np.max(acc_magnitude_windows, axis=1)
    min_vals = np.min(acc_magnitude_windows, axis=1)

    # Print overall statistics
    print("Overall Acceleration Magnitude Statistics:")
    print(f"Mean (across all windows): {np.mean(mean_vals):.3f}")
    print(f"Standard Deviation (across all windows): {np.mean(std_vals):.3f}")
    print(f"Max (across all windows): {np.max(max_vals):.3f}")
    print(f"Min (across all windows): {np.min(min_vals):.3f}")
    print(f"Median (across all windows): {np.median(mean_vals):.3f}")

    # Plot histogram of mean acceleration across windows
    plt.figure(figsize=(12, 6))
    plt.hist(mean_vals, bins=50, alpha=0.75, label="Mean Acceleration per Window")
    plt.axvline(np.mean(mean_vals), color='red', linestyle='--', label="Overall Mean")
    plt.axvline(np.median(mean_vals), color='green', linestyle='--', label="Overall Median")
    plt.title("Mean Acceleration Distribution (8-second windows, wrist)")
    plt.xlabel("Mean Acceleration Magnitude (g)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot trend of mean acceleration across windows
    plt.figure(figsize=(14, 7))
    plt.plot(mean_vals, linestyle='-', label="Mean Acceleration per Window")
    plt.title("Acceleration Magnitude Trend (8-second windows, wrist)")
    plt.xlabel("Window Number")
    plt.ylabel("Mean Acceleration Magnitude (g)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Count number of windows falling into each 0.1 g bin
    bins = np.arange(0, np.ceil(np.max(mean_vals)) + 0.1, 0.1)
    bin_counts, _ = np.histogram(mean_vals, bins=bins)

    print("\nWindow Counts in Acceleration Magnitude Ranges:")
    for i in range(len(bin_counts)):
        print(f"{bins[i]:.1f}g - {bins[i+1]:.1f}g: {bin_counts[i]} windows")

else:
    print("Wrist data is not available.")
