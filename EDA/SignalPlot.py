import numpy as np
import scipy.io
from scipy import signal
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd

def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to signal

    Args:
        signal: Input signal array

    Returns:
        Normalized signal array
    """
    if np.std(signal) == 0:
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / np.std(signal)
class SignalVisualizer:
    """
    A class to visualize filtered PPG/ECG signals with window-based analysis
    Features:
    1. Signal filtering (PPG: 0.5-4Hz, ECG: 5-40Hz)
    2. Window-based analysis (8s windows, 2s stride)
    3. Signal visualization matching the reference style
    """

    def __init__(self, data_path: str):
        """
        Initialize visualizer with configuration parameters
        """
        self.data_path = data_path
        self.window_size = 8  # seconds
        self.stride = 2  # seconds
        self.sampling_rate = 128  # Hz
        self.samples_per_window = self.window_size * self.sampling_rate
        self.stride_samples = self.stride * self.sampling_rate

        # Filter parameters
        self.ppg_filter = {
            'lowcut': 0.5,
            'highcut': 4.0,
            'order': 4
        }
        self.ecg_filter = {
            'lowcut': 5.0,
            'highcut': 40.0,
            'order': 4
        }

        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']
        self.ppg_signals = ['ppg_g', 'ppg_ir', 'ppg_r']
        self.acc_signals = ['acc_x', 'acc_y', 'acc_z']

    def design_bandpass_filter(self, lowcut: float, highcut: float, order: int) -> Tuple:
        """Design Butterworth bandpass filter"""
        nyquist = self.sampling_rate * 0.5
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def apply_filter(self, data: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply appropriate bandpass filter"""
        if filter_type == 'ppg':
            b, a = self.design_bandpass_filter(**self.ppg_filter)
        else:  # ECG
            b, a = self.design_bandpass_filter(**self.ecg_filter)

        # Apply zero-phase filtering
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data

    def load_data(self, file_name: str) -> Dict:
        """Load and clean WildPPG data"""
        file_path = os.path.join(self.data_path, file_name)
        loaded_data = scipy.io.loadmat(file_path)

        loaded_data['id'] = loaded_data['id'][0]
        loaded_data['notes'] = "" if len(loaded_data['notes']) == 0 else loaded_data['notes'][0]

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



    # 在SignalVisualizer类中添加以下方法:
    def create_window_dataframe(self, file_name: str) -> pd.DataFrame:
        """Create DataFrame with filtered and normalized windowed signals"""
        data = self.load_data(file_name)
        windows_data = []

        sternum_ppg = data['sternum']['ppg_g']['v']
        if len(sternum_ppg) == 0:
            print(f"Warning: No data found in sternum ppg_g")
            return pd.DataFrame()

        n_windows = (len(sternum_ppg) - self.samples_per_window) // self.stride_samples + 1

        for window_idx in range(n_windows):
            window_data = {
                'participant_id': data['id'],
                'window_index': window_idx,
                'start_sample': window_idx * self.stride_samples,
                'end_sample': window_idx * self.stride_samples + self.samples_per_window
            }

            for location in self.body_locations:
                start_idx = window_idx * self.stride_samples
                end_idx = start_idx + self.samples_per_window

                # Process PPG signals
                for sig in self.ppg_signals:
                    if sig in data[location]:
                        raw_signal = data[location][sig]['v'][start_idx:end_idx]
                        # Filter then normalize
                        filtered_signal = self.apply_filter(raw_signal, 'ppg')
                        normalized_signal = zscore_normalize(filtered_signal)

                        window_data[f'{location}_{sig}_raw'] = raw_signal
                        window_data[f'{location}_{sig}_filtered'] = filtered_signal
                        window_data[f'{location}_{sig}_normalized'] = normalized_signal

                # Process accelerometer signals
                for sig in self.acc_signals:
                    if sig in data[location]:
                        acc_signal = data[location][sig]['v'][start_idx:end_idx]
                        window_data[f'{location}_{sig}'] = acc_signal
                        window_data[f'{location}_{sig}_normalized'] = zscore_normalize(acc_signal)

                # Calculate acceleration magnitude
                acc_x = data[location]['acc_x']['v'][start_idx:end_idx]
                acc_y = data[location]['acc_y']['v'][start_idx:end_idx]
                acc_z = data[location]['acc_z']['v'][start_idx:end_idx]
                acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                window_data[f'{location}_acc_magnitude'] = acc_mag
                window_data[f'{location}_acc_magnitude_normalized'] = zscore_normalize(acc_mag)
                window_data[f'{location}_mean_acc'] = np.mean(acc_mag)

                # Process ECG if available
                if location == 'sternum' and 'ecg' in data[location]:
                    raw_ecg = data[location]['ecg']['v'][start_idx:end_idx]
                    filtered_ecg = self.apply_filter(raw_ecg, 'ecg')
                    normalized_ecg = zscore_normalize(filtered_ecg)

                    window_data['ecg_raw'] = raw_ecg
                    window_data['ecg_filtered'] = filtered_ecg
                    window_data['ecg_normalized'] = normalized_ecg

            windows_data.append(window_data)

        return pd.DataFrame(windows_data)

    def plot_window_signals(self, df: pd.DataFrame, window_idx: int, location: str,
                            use_normalized: bool = True):
        """
        Plot filtered and optionally normalized signals for specified window

        Args:
            df: DataFrame containing windowed signals
            window_idx: Index of window to plot
            location: Body location to plot
            use_normalized: Whether to plot normalized signals (default: True)
        """
        window_data = df[df['window_index'] == window_idx].iloc[0]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        samples = np.arange(window_data['start_sample'], window_data['end_sample'])

        # Plot PPG
        if use_normalized:
            ppg_signal = window_data[f'{location}_ppg_g_normalized']
            ylabel_ppg = 'PPG (z-score)'
        else:
            ppg_signal = window_data[f'{location}_ppg_g_filtered']
            ylabel_ppg = 'PPG (a.u.)'

        ax1.plot(samples, ppg_signal, 'b-', label='PPG', linewidth=1)
        ax1.set_ylabel(ylabel_ppg)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Plot ECG
        if 'ecg_normalized' in window_data:
            if use_normalized:
                ecg_signal = window_data['ecg_normalized']
                ylabel_ecg = 'ECG (z-score)'
            else:
                ecg_signal = window_data['ecg_filtered']
                ylabel_ecg = 'ECG (mV)'

            ax2.plot(samples, ecg_signal, 'b-', label='ECG', linewidth=1)
            ax2.set_ylabel(ylabel_ecg)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')

        # Plot acceleration
        if use_normalized:
            acc_signal = window_data[f'{location}_acc_magnitude_normalized']
            ylabel_acc = 'Acc (z-score)'
        else:
            acc_signal = window_data[f'{location}_acc_magnitude']
            ylabel_acc = 'Acc (g)'

        ax3.plot(samples, acc_signal, 'g-', label='Acceleration', linewidth=1)
        ax3.set_ylabel(ylabel_acc)
        ax3.set_xlabel('Sample Index')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        norm_status = "Normalized" if use_normalized else "Original"
        plt.suptitle(f'Signal Analysis at Sample {window_data["start_sample"]}\n'
                     f'{self.window_size} seconds Window ({norm_status})', y=1.02)

        plt.tight_layout()
        plt.show()

    def find_high_acc_windows(self, df: pd.DataFrame, location: str, threshold: float) -> List[int]:
        """Find window indices where acceleration exceeds threshold"""
        mean_acc = df[f'{location}_mean_acc']
        high_acc_windows = df[mean_acc > threshold]['window_index'].tolist()

        print(f"\nWindows exceeding {threshold}g at {location}:")
        for idx in high_acc_windows:
            print(f"Window {idx}: Mean acc = {mean_acc[idx]:.2f}g")

        return high_acc_windows


# Example usage
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = SignalVisualizer("G:\\My Drive\\Dataset\\WildPPG")

    # Process file and create window DataFrame
    file_name = "WildPPG_Part_an0.mat"
    df = visualizer.create_window_dataframe(file_name)

    # Find windows with high acceleration
    high_acc_windows = visualizer.find_high_acc_windows(df, location='wrist', threshold=1.4)

    # Plot signals for a specific window
    if len(high_acc_windows) > 0:
        window_to_plot = high_acc_windows[0]  # Plot first high acceleration window
        visualizer.plot_window_signals(df, window_idx=5, location='wrist', use_normalized=True)