import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import scipy.io
import os


class WildPPGProcessor:
    """
    Process WildPPG dataset with sliding window analysis:
    1. Segment signals into 8-second windows with 2-second stride
    2. Create DataFrame structure combining signals from all locations
    3. Calculate and classify acceleration levels
    4. Provide interface for quality assessment
    """

    def __init__(self, data_path: str):
        """
        Initialize processor with configuration parameters

        Args:
            data_path: Path to WildPPG dataset
        """
        self.data_path = data_path
        self.window_size = 8  # seconds
        self.stride = 2  # seconds
        self.sampling_rate = 128  # Hz
        self.samples_per_window = self.window_size * self.sampling_rate
        self.stride_samples = self.stride * self.sampling_rate

        # Body locations to analyze
        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']

        # Signal types to include
        self.ppg_signals = ['ppg_g', 'ppg_ir', 'ppg_r']
        self.acc_signals = ['acc_x', 'acc_y', 'acc_z']
        self.other_signals = ['altitude', 'temperature']

        # Acceleration thresholds (in g)
        self.acc_thresholds = {
            'stationary': 1.0,
            'light_motion': 1.5,
            'moderate_motion': 2.0,
            'intensive_motion': 4.0
        }

    def load_data(self, file_name: str) -> Dict:
        """Load and clean WildPPG data"""
        file_path = os.path.join(self.data_path, file_name)
        loaded_data = scipy.io.loadmat(file_path)

        # Clean ID and notes
        loaded_data['id'] = loaded_data['id'][0]
        loaded_data['notes'] = "" if len(loaded_data['notes']) == 0 else loaded_data['notes'][0]

        # Clean body location data
        for bodyloc in self.body_locations:
            bodyloc_data = dict()
            sensors = loaded_data[bodyloc][0].dtype.names
            for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
                bodyloc_data[sensor_name] = dict()
                field_names = sensor_data[0][0].dtype.names
                for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                    bodyloc_data[sensor_name][sensor_field] = field_data[0]
                    if sensor_field == 'fs':
                        bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
            loaded_data[bodyloc] = bodyloc_data

        return loaded_data

    def segment_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Segment signal into overlapping windows with specified stride

        Args:
            signal: Input signal array

        Returns:
            List of windowed signal segments
        """
        total_samples = len(signal)
        windows = []

        # Calculate number of windows
        n_windows = (total_samples - self.samples_per_window) // self.stride_samples + 1

        for i in range(n_windows):
            start_idx = i * self.stride_samples
            end_idx = start_idx + self.samples_per_window
            if end_idx <= total_samples:
                windows.append(signal[start_idx:end_idx])

        return windows

    def classify_motion(self, mean_acc: float) -> str:
        """
        Classify motion level based on mean acceleration.

        Args:
            mean_acc: Mean acceleration magnitude (g).

        Returns:
            Motion classification label.
        """
        if mean_acc <= 1.0:
            return 'stationary'
        elif mean_acc <= 1.1:
            return 'light_motion'
        elif mean_acc <= 1.5:
            return 'moderate_motion'
        else:
            return 'intensive_motion'

    def create_window_dataframe(self, file_name: str) -> pd.DataFrame:
        """
        Create DataFrame with windowed signals combining all locations

        Args:
            file_name: Name of the .mat file to process

        Returns:
            DataFrame containing windowed signals and classifications
        """
        # Load data
        data = self.load_data(file_name)
        windows_data = []

        # First check sternum ppg_g data
        sternum_ppg = data['sternum']['ppg_g']['v']
        if len(sternum_ppg) == 0:
            print(f"Warning: No data found in sternum ppg_g")
            return pd.DataFrame()

        n_windows = (len(sternum_ppg) - self.samples_per_window) // self.stride_samples + 1

        # Process each window
        for window_idx in range(n_windows):
            window_data = {
                'participant_id': data['id'],
                'window_index': window_idx,
                'start_sample': window_idx * self.stride_samples,
                'end_sample': window_idx * self.stride_samples + self.samples_per_window,
                'start_time': window_idx * self.stride,
                'end_time': window_idx * self.stride + self.window_size,
                'quality_score': None  # Reserved for quality assessment
            }

            # Add data from each location
            for location in self.body_locations:
                start_idx = window_idx * self.stride_samples
                end_idx = start_idx + self.samples_per_window

                # Add and normalize PPG signals
                for sig in self.ppg_signals:
                    # Check if signal exists and has data
                    if sig in data[location] and len(data[location][sig]['v']) > 0:
                        signal = data[location][sig]['v'][start_idx:end_idx]
                        if len(signal) > 0:  # Check if signal slice is not empty
                            if np.max(signal) != np.min(signal):  # Avoid division by zero
                                signal = ((signal - np.min(signal)) /
                                          (np.max(signal) - np.min(signal))) * 1023
                            window_data[f'{location}_{sig}'] = signal
                        else:
                            window_data[f'{location}_{sig}'] = np.zeros(self.samples_per_window)
                    else:
                        window_data[f'{location}_{sig}'] = np.zeros(self.samples_per_window)

                # Add accelerometer signals (no normalization needed)
                for sig in self.acc_signals:
                    if sig in data[location] and len(data[location][sig]['v']) > 0:
                        window_data[f'{location}_{sig}'] = data[location][sig]['v'][start_idx:end_idx]
                    else:
                        window_data[f'{location}_{sig}'] = np.zeros(self.samples_per_window)

                # Calculate acceleration magnitude
                acc_x = data[location]['acc_x']['v'][start_idx:end_idx] if 'acc_x' in data[location] else np.zeros(
                    self.samples_per_window)
                acc_y = data[location]['acc_y']['v'][start_idx:end_idx] if 'acc_y' in data[location] else np.zeros(
                    self.samples_per_window)
                acc_z = data[location]['acc_z']['v'][start_idx:end_idx] if 'acc_z' in data[location] else np.zeros(
                    self.samples_per_window)

                acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                window_data[f'{location}_acc_magnitude'] = acc_mag
                window_data[f'{location}_mean_acc'] = np.mean(acc_mag)
                window_data[f'{location}_motion_class'] = self.classify_motion(np.mean(acc_mag))

                # Add ECG if available (only for sternum)
                if location == 'sternum' and 'ecg' in data[location]:
                    if len(data[location]['ecg']['v']) > 0:
                        ecg_signal = data[location]['ecg']['v'][start_idx:end_idx]
                        if len(ecg_signal) > 0:
                            if np.max(ecg_signal) != np.min(ecg_signal):
                                ecg_signal = ((ecg_signal - np.min(ecg_signal)) /
                                              (np.max(ecg_signal) - np.min(ecg_signal))) * 1023
                            window_data['ecg'] = ecg_signal
                        else:
                            window_data['ecg'] = np.zeros(self.samples_per_window)

                # Add other signals with normalization
                for sig in self.other_signals:
                    if sig in data[location] and len(data[location][sig]['v']) > 0:
                        signal = data[location][sig]['v'][start_idx:end_idx]
                        if len(signal) > 0:
                            if np.max(signal) != np.min(signal):
                                signal = ((signal - np.min(signal)) /
                                          (np.max(signal) - np.min(signal))) * 1023
                            window_data[f'{location}_{sig}'] = signal
                        else:
                            window_data[f'{location}_{sig}'] = np.zeros(self.samples_per_window)
                    else:
                        window_data[f'{location}_{sig}'] = np.zeros(self.samples_per_window)

            windows_data.append(window_data)

        df = pd.DataFrame(windows_data)

        return df



from SQA import PPGQualityAssessor

if __name__ == "__main__":
    # Create processor instance
    processor = WildPPGProcessor("G:\\My Drive\\Dataset\\WildPPG")
    quality_assessor = PPGQualityAssessor()
    # Process single file
    file_name = "WildPPG_Part_an0.mat"
    df = processor.create_window_dataframe(file_name)

    for location in processor.body_locations:
        print(f"\n=== Quality Assessment Results for {location.upper()} ===")

        # Assess quality for each PPG channel
        for ppg_type in processor.ppg_signals:
            signal_col = f'{location}_{ppg_type}'
            df_assessed = quality_assessor.process_dataframe(df, location=location, signal_type=ppg_type)

            # Get motion class statistics
            print(f"\nSignal: {ppg_type}")
            print("\nQuality distribution by motion class:")

            # Group by motion class and quality
            stats = pd.crosstab(
                df_assessed[f'{location}_motion_class'],
                df_assessed[f'{signal_col}_quality'],
                normalize='index'
            ) * 100

            print(stats.round(2))

            # Overall quality statistics
            print("\nOverall quality distribution:")
            quality_counts = df_assessed[f'{signal_col}_quality'].value_counts()
            quality_percent = quality_counts / len(df_assessed) * 100
            for quality, percent in quality_percent.items():
                print(f"{quality}: {percent:.2f}%")

            # FOPC statistics by motion class
            print("\nFOPC statistics by motion class:")
            fopc_stats = df_assessed.groupby(f'{location}_motion_class')[f'{signal_col}_fopc'].agg(
                ['mean', 'std']).round(3)
            print(fopc_stats)

            print("\n" + "=" * 50)