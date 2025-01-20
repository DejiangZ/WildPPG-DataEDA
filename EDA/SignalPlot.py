import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import scipy.io
import os


class WildPPGSignalAnalyzer:
    """
    A class to analyze and visualize PPG/ECG signals with motion artifacts
    Features:
    1. Select body location and PPG channel
    2. Filter signals based on accelerometer magnitude threshold
    3. Apply appropriate bandpass filters to PPG and ECG
    4. Visualize filtered signals with acceleration data
    """

    def __init__(self, data_path: str):
        """
        Initialize the analyzer with dataset path and parameters

        Args:
            data_path: Path to WildPPG dataset
        """
        self.data_path = data_path
        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']
        self.ppg_channels = ['ppg_g', 'ppg_ir', 'ppg_r']
        self.fs = 128  # Sampling frequency in Hz

        # Filter parameters
        self.ppg_filter = {
            'lowcut': 0.5,  # Hz
            'highcut': 4.0,  # Hz
            'order': 4
        }
        self.ecg_filter = {
            'lowcut': 5.0,  # Hz
            'highcut': 40.0,  # Hz
            'order': 4
        }

    def load_data(self, file_name: str) -> Dict:
        """
        Load and clean WildPPG data

        Args:
            file_name: Name of the .mat file

        Returns:
            Cleaned data dictionary
        """
        file_path = os.path.join(self.data_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = scipy.io.loadmat(file_path)

        # Clean ID and notes
        data['id'] = data['id'][0]
        data['notes'] = "" if len(data['notes']) == 0 else data['notes'][0]

        # Clean body location data
        for bodyloc in self.body_locations:
            bodyloc_data = dict()
            if bodyloc in data:
                sensors = data[bodyloc][0].dtype.names
                for sensor_name, sensor_data in zip(sensors, data[bodyloc][0][0]):
                    bodyloc_data[sensor_name] = dict()
                    field_names = sensor_data[0][0].dtype.names
                    for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                        bodyloc_data[sensor_name][sensor_field] = field_data[0]
                        if sensor_field == 'fs':
                            bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
            data[bodyloc] = bodyloc_data

        return data

    def get_acc_magnitude(self, data: Dict, location: str) -> np.ndarray:
        """
        Calculate acceleration magnitude for a given location

        Args:
            data: Data dictionary
            location: Body location

        Returns:
            Acceleration magnitude array
        """
        try:
            acc_x = data[location]['acc_x']['v']
            acc_y = data[location]['acc_y']['v']
            acc_z = data[location]['acc_z']['v']
            return np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        except KeyError as e:
            raise KeyError(f"Missing accelerometer data for {location}: {str(e)}")

    def apply_bandpass_filter(self, data: np.ndarray, is_ppg: bool = True) -> np.ndarray:
        """
        Apply appropriate bandpass filter based on signal type

        Args:
            data: Signal data array
            is_ppg: True for PPG signals, False for ECG

        Returns:
            Filtered signal array
        """
        filter_params = self.ppg_filter if is_ppg else self.ecg_filter
        nyquist = self.fs * 0.5
        low = filter_params['lowcut'] / nyquist
        high = filter_params['highcut'] / nyquist

        b, a = signal.butter(filter_params['order'], [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def find_motion_segments(self, acc_mag: np.ndarray, threshold: float) -> List[int]:
        """
        Find central indices of segments where acceleration exceeds threshold

        Args:
            acc_mag: Acceleration magnitude array
            threshold: Acceleration threshold in g

        Returns:
            List of central indices for high acceleration segments
        """
        high_acc_indices = np.where(acc_mag > threshold)[0]
        if len(high_acc_indices) == 0:
            return []

        # Group consecutive indices into segments
        segments = np.split(high_acc_indices, np.where(np.diff(high_acc_indices) > 1)[0] + 1)

        # Get center index of each segment
        centers = [segment[len(segment) // 2] for segment in segments]
        return centers

    def extract_window(self, signal: np.ndarray, center_idx: int,
                       window_seconds: float = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract signal window around center index

        Args:
            signal: Input signal array
            center_idx: Center index for the window
            window_seconds: Window size in seconds

        Returns:
            Tuple of (time array, signal window)
        """
        half_window = int(window_seconds * self.fs / 2)
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(signal), center_idx + half_window)

        signal_window = signal[start_idx:end_idx]
        time = np.arange(len(signal_window)) / self.fs - window_seconds / 2

        return time, signal_window

    def plot_signals(self, ppg_signal: np.ndarray, ecg_signal: np.ndarray,
                     acc_mag: np.ndarray, center_idx: int, ppg_location: str,
                     window_seconds: float = 30) -> None:
        """
        Create visualization of PPG, ECG and acceleration signals with professional styling

        Args:
            ppg_signal: Filtered PPG signal
            ecg_signal: Filtered ECG signal
            acc_mag: Acceleration magnitude
            center_idx: Center index for the window
            ppg_location: Location of PPG sensor
            window_seconds: Window size in seconds
        """
        # Extract signal windows and calculate indices
        half_window = int(window_seconds * self.fs / 2)
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(ppg_signal), center_idx + half_window)

        # Get signal windows
        ppg_window = ppg_signal[start_idx:end_idx]
        ecg_window = ecg_signal[start_idx:end_idx]
        acc_window = acc_mag[start_idx:end_idx]

        # Create sample indices for x-axis
        indices = np.arange(start_idx, end_idx)

        # Set professional color scheme
        ppg_color = '#2F5597'  # 深蓝色
        ecg_color = '#2F5597'  # 深红色
        acc_color = '#548235'  # 深绿色

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Set style parameters
        plt.style.use('seaborn-white')
        line_width = 1.2

        # Plot signals with sample indices
        ax1.plot(indices, ppg_window, color=ppg_color, linewidth=line_width,
                 label=f'PPG ({ppg_location})')
        ax2.plot(indices, ecg_window, color=ecg_color, linewidth=line_width,
                 label='ECG (sternum)')
        ax3.plot(indices, acc_window, color=acc_color, linewidth=line_width,
                 label='Acceleration')

        # Configure axes
        ax1.set_ylabel('PPG (a.u.)', fontsize=10)
        ax2.set_ylabel('ECG (mV)', fontsize=10)
        ax3.set_ylabel('Acc (g)', fontsize=10)
        ax3.set_xlabel('Sample Index', fontsize=10)

        # Add second x-axis for time reference
        ax1_t = ax1.twiny()
        time_ticks = np.linspace(start_idx, end_idx, 5)
        time_labels = [f'{(x - center_idx) / self.fs:.1f}' for x in time_ticks]
        ax1_t.set_xticks(time_ticks)
        ax1_t.set_xticklabels(time_labels)
        ax1_t.set_xlabel('Time (s)', fontsize=10)

        # Configure grid and legends
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3, color='gray', linestyle='--')
            ax.legend(loc='upper right', frameon=True, fontsize=9)
            ax.tick_params(labelsize=9)

        # Set title
        plt.suptitle(f'Signal Analysis at Sample {center_idx}', fontsize=12)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        plt.show()

    def analyze_signals(self, file_name: str, ppg_location: str = 'wrist',
                        ppg_channel: str = 'ppg_g', acc_threshold: float = 4.0,
                        segment_index: int = 0) -> None:
        """
        Analyze and visualize PPG and ECG signals with motion artifacts

        Args:
            file_name: Name of the data file
            ppg_location: Body location for PPG sensor
            ppg_channel: PPG channel to analyze
            acc_threshold: Acceleration threshold in g
            segment_index: Index of the segment to analyze
        """
        # Input validation
        if ppg_location not in self.body_locations:
            raise ValueError(f"Invalid location. Must be one of {self.body_locations}")
        if ppg_channel not in self.ppg_channels:
            raise ValueError(f"Invalid PPG channel. Must be one of {self.ppg_channels}")

        # Load and prepare data
        data = self.load_data(file_name)

        # Get PPG and acceleration data
        try:
            ppg_data = data[ppg_location][ppg_channel]['v']
            acc_mag = self.get_acc_magnitude(data, ppg_location)
            ecg_data = data['sternum']['ecg']['v']
        except KeyError as e:
            raise KeyError(f"Missing required data: {str(e)}")

        # Find motion segments
        segments = self.find_motion_segments(acc_mag, acc_threshold)
        if not segments:
            print(f"No segments found with acceleration > {acc_threshold}g")
            return

        print(f"Found {len(segments)} segments with acceleration > {acc_threshold}g")

        # Select segment
        if segment_index >= len(segments):
            print(f"Segment index {segment_index} out of range. Using first segment.")
            segment_index = 0

        center_idx = segments[segment_index]

        # Filter signals
        filtered_ppg = self.apply_bandpass_filter(ppg_data, is_ppg=True)
        filtered_ecg = self.apply_bandpass_filter(ecg_data, is_ppg=False)

        # Plot signals
        self.plot_signals(filtered_ppg, filtered_ecg, acc_mag, center_idx, ppg_location)


# Example usage
if __name__ == "__main__":
    analyzer = WildPPGSignalAnalyzer("G:\\My Drive\\Dataset\\WildPPG")

    try:
        analyzer.analyze_signals(
            file_name="WildPPG_Part_an0.mat",
            ppg_location="wrist",
            ppg_channel="ppg_g",
            acc_threshold=3.0,
            segment_index=79
        )
    except Exception as e:
        print(f"Error analyzing signals: {str(e)}")