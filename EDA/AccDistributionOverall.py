import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def load_wildppg_participant(path):
    """
    Loads the data of a WildPPG participant and cleans it to receive nested dictionaries.
    """
    loaded_data = scipy.io.loadmat(path)
    loaded_data['id'] = loaded_data['id'][0]
    if len(loaded_data['notes']) == 0:
        loaded_data['notes'] = ""
    else:
        loaded_data['notes'] = loaded_data['notes'][0]

    for bodyloc in ['sternum', 'head', 'wrist', 'ankle']:
        bodyloc_data = dict()
        if bodyloc in loaded_data:
            sensors = loaded_data[bodyloc][0].dtype.names
            for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
                bodyloc_data[sensor_name] = dict()
                field_names = sensor_data[0][0].dtype.names
                for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                    bodyloc_data[sensor_name][sensor_field] = field_data[0]
                    if sensor_field == 'fs':
                        bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
            loaded_data[bodyloc] = bodyloc_data
        else:
            loaded_data[bodyloc] = None

    return loaded_data


def calculate_acceleration_magnitude(acc_x, acc_y, acc_z):
    """
    Calculate acceleration magnitude.
    """
    return np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))


def segment_signal(signal, window_size, stride):
    """
    Segment signal into overlapping windows with specified stride.
    """
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


def process_participant_data(file_path: str, location: str = 'wrist') -> Dict:
    """
    Process acceleration data for a single participant at specified location.

    Args:
        file_path: Path to participant's .mat file
        location: Body location to analyze ('wrist', 'head', 'ankle', 'sternum')

    Returns:
        Dictionary containing acceleration statistics
    """
    participant_data = load_wildppg_participant(file_path)
    results = {
        'participant_id': participant_data['id'],
        'valid_data': False,
        'statistics': None
    }

    if participant_data[location]:
        acc_x = participant_data[location]['acc_x']['v']
        acc_y = participant_data[location]['acc_y']['v']
        acc_z = participant_data[location]['acc_z']['v']

        # Ensure signals have consistent lengths
        min_length = min(len(acc_x), len(acc_y), len(acc_z))
        acc_x, acc_y, acc_z = acc_x[:min_length], acc_y[:min_length], acc_z[:min_length]

        # Parameters for windowing
        sampling_rate = participant_data[location]['acc_x']['fs']
        window_size = 8 * sampling_rate  # 8 seconds window
        stride = 2 * sampling_rate  # 2 seconds stride

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

        results['valid_data'] = True
        results['statistics'] = {
            'mean_vals': mean_vals,
            'std_vals': std_vals,
            'max_vals': max_vals,
            'min_vals': min_vals,
            'overall_mean': np.mean(mean_vals),
            'overall_std': np.mean(std_vals),
            'overall_max': np.max(max_vals),
            'overall_min': np.min(min_vals),
            'overall_median': np.median(mean_vals)
        }

    return results



def analyze_dataset(data_path: str, locations: List[str] = None):
    """
    Analyze acceleration distribution for all participants in the dataset.

    Args:
        data_path: Path to WildPPG dataset
        locations: List of body locations to analyze. If None, analyzes all locations.
    """
    if locations is None:
        locations = ['wrist', 'head', 'ankle', 'sternum']

    # Get all .mat files in the directory
    mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]

    # Process each location
    for location in locations:
        print(f"\nAnalyzing {location.upper()} location:")
        print("-" * 50)

        all_means = []
        all_maxs = []
        valid_participants = 0

        # Process each participant
        for mat_file in mat_files:
            file_path = os.path.join(data_path, mat_file)
            results = process_participant_data(file_path, location)

            if results['valid_data']:
                valid_participants += 1
                stats = results['statistics']
                all_means.extend(stats['mean_vals'])
                all_maxs.extend(stats['max_vals'])

                print(f"\nParticipant {results['participant_id']}:")
                print(f"Mean acceleration: {stats['overall_mean']:.3f}g")
                print(f"Max acceleration: {np.max(stats['max_vals']):.3f}g")
                print(f"Number of windows: {len(stats['mean_vals'])}")

        if valid_participants > 0:
            # Plot overall distribution
            plt.figure(figsize=(12, 6))
            plt.hist(all_means, bins=50, alpha=0.75, label="Mean Acceleration per Window")
            plt.axvline(np.mean(all_means), color='red', linestyle='--', label="Overall Mean")
            plt.axvline(np.median(all_means), color='green', linestyle='--', label="Overall Median")
            plt.title(f"Mean Acceleration Distribution - {location.upper()}\n(All Participants, 8-second windows)")
            plt.xlabel("Mean Acceleration Magnitude (g)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Print overall statistics
            print(f"\nOverall Statistics for {location.upper()}:")
            print(f"Number of valid participants: {valid_participants}")
            print(f"Total windows analyzed: {len(all_means)}")
            print(f"Mean acceleration across all windows: {np.mean(all_means):.3f}g")
            print(f"Median acceleration across all windows: {np.median(all_means):.3f}g")
            print(f"Maximum acceleration observed: {np.max(all_maxs):.3f}g")

            # Detailed window distribution analysis
            print("\nDetailed Window Distribution Analysis:")
            print("-" * 40)

            # Create bins from 0 to maximum acceleration (rounded up to next 0.1)
            max_acc = np.ceil(np.max(all_means) * 10) / 10
            bins = np.arange(0, max_acc + 0.1, 0.1)

            # Calculate histogram
            hist, bin_edges = np.histogram(all_means, bins=bins)

            # Create distributions table
            print("\nAcceleration Range Distribution:")
            print("Range (g)      | Windows | Percentage")
            print("-" * 40)

            total_windows = len(all_means)
            cumulative_count = 0
            cumulative_percent = 0

            for i in range(len(hist)):
                count = hist[i]
                percentage = (count / total_windows) * 100
                cumulative_count += count
                cumulative_percent += percentage

                if count > 0:  # Only show ranges with data
                    print(f"{bin_edges[i]:4.1f} - {bin_edges[i + 1]:4.1f} | {count:7d} | {percentage:8.2f}%")

            # Plot cumulative distribution
            plt.figure(figsize=(12, 6))
            plt.plot(bin_edges[1:], np.cumsum(hist) / total_windows * 100,
                     'b-', linewidth=2, label='Cumulative Distribution')
            plt.grid(True)
            plt.xlabel('Acceleration Magnitude (g)')
            plt.ylabel('Cumulative Percentage (%)')
            plt.title(f'Cumulative Distribution of Window Acceleration - {location.upper()}')
            plt.legend()
            plt.show()

            # Print summary of ranges
            print("\nSummary of Key Ranges:")
            print(
                f"Windows below 1.0g: {np.sum(hist[bins[:-1] < 1.0]):7d} ({np.sum(hist[bins[:-1] < 1.0]) / total_windows * 100:5.1f}%)")
            print(
                f"Windows 1.0g-2.0g: {np.sum(hist[(bins[:-1] >= 1.0) & (bins[:-1] < 2.0)]):7d} ({np.sum(hist[(bins[:-1] >= 1.0) & (bins[:-1] < 2.0)]) / total_windows * 100:5.1f}%)")
            print(
                f"Windows above 2.0g: {np.sum(hist[bins[:-1] >= 2.0]):7d} ({np.sum(hist[bins[:-1] >= 2.0]) / total_windows * 100:5.1f}%)")

        else:
            print(f"No valid data found for {location}")

# Example usage
if __name__ == "__main__":
    data_path = r'G:\My Drive\Dataset\WildPPG'
    analyze_dataset(data_path)