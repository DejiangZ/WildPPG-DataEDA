import numpy as np
import pandas as pd
import scipy.io
import os
from typing import Dict, List
from pathlib import Path


class WildPPGtoCSV:
    """
    Convert WildPPG dataset to CSV files with sliding window analysis:
    1. Segment signals into 8-second windows with 2-second stride
    2. Create DataFrame structure combining signals from all locations
    3. Save segmented data to CSV files by participant
    """

    def __init__(self, data_path: str, output_path: str):
        """
        Initialize converter with paths and parameters

        Args:
            data_path: Path to WildPPG dataset
            output_path: Path to save CSV files
        """
        self.data_path = data_path
        self.output_path = output_path

        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Window parameters
        self.window_size = 8  # seconds
        self.stride = 2  # seconds
        self.sampling_rate = 128  # Hz
        self.samples_per_window = self.window_size * self.sampling_rate
        self.stride_samples = self.stride * self.sampling_rate

        # Signal configuration
        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']
        self.ppg_signals = ['ppg_g', 'ppg_ir', 'ppg_r']
        self.acc_signals = ['acc_x', 'acc_y', 'acc_z']
        self.other_signals = ['altitude', 'temperature']

    def load_mat_file(self, file_name: str) -> Dict:
        """
        Load and clean WildPPG .mat file data

        Args:
            file_name: Name of the .mat file

        Returns:
            Dictionary containing cleaned data
        """
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

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Standardize signal using Z-score normalization.

        Args:
            signal: Input signal array

        Returns:
            Z-score standardized signal array
        """
        if len(signal) == 0:
            return np.zeros(self.samples_per_window)

        mean = np.mean(signal)
        std = np.std(signal)

        if std == 0:  # Avoid division by zero
            return signal - mean

        return (signal - mean) / std

    def create_window_dataframe(self, data: Dict) -> pd.DataFrame:
        """
        Create DataFrame with windowed signals.

        Args:
            data: Loaded and cleaned WildPPG data

        Returns:
            DataFrame containing windowed signals
        """
        windows_data = []

        # Get reference signal for windowing
        ref_signal = data['sternum']['ppg_g']['v']
        n_windows = (len(ref_signal) - self.samples_per_window) // self.stride_samples + 1

        for window_idx in range(n_windows):
            start_idx = window_idx * self.stride_samples
            end_idx = start_idx + self.samples_per_window

            window_data = {
                'participant_id': data['id'],
                'window_index': window_idx,
                'start_time': window_idx * self.stride,
                'end_time': window_idx * self.stride + self.window_size,
            }

            # Process each location
            for location in self.body_locations:
                for sig in self.ppg_signals:
                    if sig in data[location]:
                        signal = data[location][sig]['v'][start_idx:end_idx]
                        window_data[f'{location}_{sig}'] = list(self.normalize_signal(signal))
                    else:
                        window_data[f'{location}_{sig}'] = [0] * self.samples_per_window

                for sig in self.acc_signals:
                    if sig in data[location]:
                        window_data[f'{location}_{sig}'] = list(data[location][sig]['v'][start_idx:end_idx])
                    else:
                        window_data[f'{location}_{sig}'] = [0] * self.samples_per_window

                for sig in self.other_signals:
                    if sig in data[location]:
                        signal = data[location][sig]['v'][start_idx:end_idx]
                        window_data[f'{location}_{sig}'] = list(self.normalize_signal(signal))
                    else:
                        window_data[f'{location}_{sig}'] = [0] * self.samples_per_window

                if location == 'sternum' and 'ecg' in data[location]:
                    signal = data[location]['ecg']['v'][start_idx:end_idx]
                    window_data['ecg'] = list(self.normalize_signal(signal))

            windows_data.append(window_data)

        return pd.DataFrame(windows_data)

    def process_and_save(self, file_name: str):
        """
        Process a single .mat file and save as CSV.

        Args:
            file_name: Name of the .mat file to process
        """
        try:
            # Load data
            data = self.load_mat_file(file_name)

            # Create DataFrame
            df = self.create_window_dataframe(data)

            # Expand signal arrays into multiple columns
            expanded_df = df.explode(
                [col for col in df.columns if col not in ['participant_id', 'window_index', 'start_time', 'end_time']])

            # Create output filename
            participant_id = data['id']
            output_file = os.path.join(self.output_path, f'participant_{participant_id}.csv')

            # Save to CSV
            expanded_df.to_csv(output_file, index=False)
            print(f"Successfully processed and saved data for participant {participant_id}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    def process_all_files(self):
        """Process all .mat files in the data directory"""
        mat_files = [f for f in os.listdir(self.data_path) if f.endswith('.mat')]
        total_files = len(mat_files)

        print(f"Found {total_files} .mat files to process")

        for i, file_name in enumerate(mat_files, 1):
            print(f"Processing file {i}/{total_files}: {file_name}")
            self.process_and_save(file_name)


# Example usage
if __name__ == "__main__":
    converter = WildPPGtoCSV(
        data_path="G:\\My Drive\\Dataset\\WildPPG",
        output_path="G:\\My Drive\\Dataset\\WildPPG_CSV"
    )
    converter.process_all_files()