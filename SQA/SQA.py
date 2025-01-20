import numpy as np
from scipy import signal
import os
from typing import Dict, List, Tuple
import scipy.io


class PPGQualityChecker:
    """
    PPG signal quality assessment system for WildPPG dataset
    Features:
    1. Acceleration-based motion artifact detection
    2. Signal saturation detection
    3. Predictor coefficient based quality assessment
    """

    def __init__(self, data_path: str):
        """
        Initialize the checker

        Args:
            data_path: Path to WildPPG dataset
        """
        self.data_path = data_path
        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']

        # Thresholds
        self.acc_threshold = 7.0  # Acceleration threshold in g
        self.pc_threshold = 0.93  # Predictor coefficient threshold
        self.fs = 128  # Sampling frequency

    def load_data(self, file_name: str) -> Dict:
        """
        Load and clean WildPPG participant data

        Args:
            file_name: Name of the .mat file

        Returns:
            Cleaned data dictionary
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

    def check_saturation(self, signal: np.ndarray) -> bool:
        """
        Check if signal is saturated

        Args:
            signal: PPG signal to check

        Returns:
            True if signal is saturated, False otherwise
        """
        # Check for constant values that indicate saturation
        diff = np.diff(signal)
        zero_runs = np.where(diff == 0)[0]

        if len(zero_runs) > 0:
            # Check for runs of constant values
            run_lengths = np.diff(np.where(np.abs(np.diff(zero_runs)) > 1)[0])
            if np.any(run_lengths > self.fs * 0.1):  # Runs longer than 100ms
                return True
        return False

    def get_acc_magnitude(self, data: Dict, location: str) -> np.ndarray:
        """
        Calculate acceleration magnitude for given location

        Args:
            data: Data dictionary
            location: Body location

        Returns:
            Acceleration magnitude array
        """
        acc_x = data[location]['acc_x']['v']
        acc_y = data[location]['acc_y']['v']
        acc_z = data[location]['acc_z']['v']

        return np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

    def compute_predictor_coefficient(self, signal: np.ndarray) -> float:
        """
        Compute first order predictor coefficient

        Args:
            signal: Input signal

        Returns:
            Predictor coefficient value
        """
        # Add random noise to improve feature discrimination
        noise_level = 0.1
        noise = np.random.normal(0, noise_level, len(signal))
        signal = signal + noise

        # Compute first difference
        diff_signal = np.diff(signal)

        # Normalize
        diff_signal = diff_signal / np.max(np.abs(diff_signal))

        # Compute autocorrelation coefficients
        r = np.correlate(diff_signal, diff_signal, mode='full')
        r = r[len(r) // 2:][:2]

        # Compute predictor coefficient
        if r[0] != 0:
            pc = r[1] / r[0]
        else:
            pc = 0

        return pc

    def assess_quality(self, data: Dict, location: str) -> Dict:
        """
        Assess PPG signal quality for given location

        Args:
            data: Data dictionary
            location: Body location to assess

        Returns:
            Quality assessment results
        """
        results = {
            'location': location,
            'quality': 'unknown',
            'issues': []
        }

        # Get PPG and acceleration data
        ppg_signal = data[location]['ppg_g']['v']
        acc_mag = self.get_acc_magnitude(data, location)

        # Check for saturation
        if self.check_saturation(ppg_signal):
            results['quality'] = 'poor'
            results['issues'].append('signal_saturation')
            return results

        # Check acceleration
        if np.any(acc_mag > self.acc_threshold):
            results['quality'] = 'poor'
            results['issues'].append('motion_artifact')

        # Compute predictor coefficient
        pc = self.compute_predictor_coefficient(ppg_signal)

        # Classify signal quality based on PC
        if pc > self.pc_threshold:
            if 'motion_artifact' not in results['issues']:
                results['quality'] = 'good'
        elif pc > -0.5:
            results['quality'] = 'poor'
            if 'motion_artifact' not in results['issues']:
                results['issues'].append('noisy_signal')
        else:
            results['quality'] = 'poor'
            results['issues'].append('no_pulse')

        return results

# 创建质量检查器实例
checker = PPGQualityChecker("G:\\My Drive\\Dataset\\WildPPG")

# 加载一个参与者的数据
data = checker.load_data("WildPPG_Part_an0.mat")

for location in checker.body_locations:
    results = checker.assess_quality(data, location)
    print(f"\nLocation: {results['location']}")
    print(f"Quality: {results['quality']}")
    if results['issues']:
        print(f"Issues: {', '.join(results['issues'])}")