import numpy as np
import scipy.io
import os
from typing import Dict, List, Tuple
from pythonProject.EDA.MissingRateCheck import checker

class WildPPGSignalQualityChecker:
    """
    A class to check the signal quality of WildPPG dataset.
    Focuses on:
    1. Accelerometer signal range check (> 10g)
    2. Weak signal detection for PPG and ECG
    """


    def __init__(self, data_path: str):
        """
        Initialize the checker

        Args:
            data_path: Path to the WildPPG dataset
        """
        self.data_path = data_path
        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']

        # Define sensor groups
        self.acc_sensors = ['acc_x', 'acc_y', 'acc_z']
        self.ppg_sensors = ['ppg_g', 'ppg_ir', 'ppg_r']
        self.ecg_sensors = ['ecg']

        # Constants
        self.ACC_THRESHOLD = 10.0  # 10g threshold
        self.WEAK_SIGNAL_FACTOR = 0.1  # Factor for weak signal detection

    def check_acc_range(self, data: Dict) -> Dict:
        """
        Check if accelerometer data exceeds 10g threshold

        Args:
            data: Loaded WildPPG data dictionary

        Returns:
            Dictionary containing accelerometer range statistics
        """
        results = {}

        for location in self.body_locations:
            location_results = {
                'total_points': 0,
                'exceeded_points': 0,
                'exceed_ratio': 0.0,
                'max_value': 0.0
            }

            # Skip if location data is not available
            if not data[location]:
                results[location] = None
                continue

            # Calculate magnitude of acceleration
            acc_data = []
            for sensor in self.acc_sensors:
                if sensor in data[location]:
                    acc_values = data[location][sensor]['v']
                    acc_data.append(acc_values)

            if len(acc_data) == 3:  # Only process if we have all three axes
                acc_data = np.array(acc_data)
                # Calculate magnitude (in g's)
                acc_magnitude = np.sqrt(np.sum(np.square(acc_data), axis=0))

                # Calculate statistics
                valid_data = acc_magnitude[~np.isnan(acc_magnitude)]
                location_results['total_points'] = len(valid_data)
                location_results['exceeded_points'] = np.sum(valid_data > self.ACC_THRESHOLD)
                location_results['exceed_ratio'] = (location_results['exceeded_points'] /
                                                    location_results['total_points']
                                                    if location_results['total_points'] > 0 else 0)
                location_results['max_value'] = np.max(valid_data)

            results[location] = location_results

        return results

    def check_weak_signals(self, data: Dict) -> Dict:
        """
        Check for weak signals in PPG and ECG data

        Args:
            data: Loaded WildPPG data dictionary

        Returns:
            Dictionary containing weak signal statistics
        """
        results = {}

        for location in self.body_locations:
            location_results = {
                'ppg': {},
                'ecg': {} if location == 'sternum' else None  # ECG only available at sternum
            }

            # Skip if location data is not available
            if not data[location]:
                results[location] = None
                continue

            # Check PPG signals
            for sensor in self.ppg_sensors:
                if sensor in data[location]:
                    signal_data = data[location][sensor]['v']
                    valid_data = signal_data[~np.isnan(signal_data)]

                    if len(valid_data) > 0:
                        signal_range = np.ptp(valid_data)
                        std_dev = np.std(valid_data)
                        is_weak = signal_range < std_dev * self.WEAK_SIGNAL_FACTOR

                        location_results['ppg'][sensor] = {
                            'signal_range': signal_range,
                            'std_dev': std_dev,
                            'is_weak': is_weak
                        }

            # Check ECG signal (only for sternum)
            if location == 'sternum' and 'ecg' in data[location]:
                signal_data = data[location]['ecg']['v']
                valid_data = signal_data[~np.isnan(signal_data)]

                if len(valid_data) > 0:
                    signal_range = np.ptp(valid_data)
                    std_dev = np.std(valid_data)
                    is_weak = signal_range < std_dev * self.WEAK_SIGNAL_FACTOR

                    location_results['ecg']['ecg'] = {
                        'signal_range': signal_range,
                        'std_dev': std_dev,
                        'is_weak': is_weak
                    }

            results[location] = location_results

        return results

    def print_report(self, acc_results: Dict, weak_signal_results: Dict):
        """
        Print a formatted report of the signal quality checks
        """
        print("\n=== Signal Quality Report ===")

        # Accelerometer Report
        print("\n--- Accelerometer Range Check ---")
        for location, results in acc_results.items():
            print(f"\nLocation: {location}")
            if results:
                print(f"Total points: {results['total_points']}")
                print(f"Points exceeding 10g: {results['exceeded_points']}")
                print(f"Exceed ratio: {results['exceed_ratio'] * 100:.2f}%")
                print(f"Maximum acceleration: {results['max_value']:.2f}g")
            else:
                print("No accelerometer data available")

        # Weak Signal Report
        print("\n--- Weak Signal Check ---")
        for location, results in weak_signal_results.items():
            print(f"\nLocation: {location}")
            if results:
                if results['ppg']:
                    print("\nPPG Signals:")
                    for sensor, stats in results['ppg'].items():
                        print(f"  {sensor}:")
                        print(f"    Signal Range: {stats['signal_range']:.2f}")
                        print(f"    Standard Deviation: {stats['std_dev']:.2f}")
                        print(f"    Weak Signal: {'Yes' if stats['is_weak'] else 'No'}")

                if results['ecg']:
                    print("\nECG Signal:")
                    stats = results['ecg']['ecg']
                    print(f"    Signal Range: {stats['signal_range']:.2f}")
                    print(f"    Standard Deviation: {stats['std_dev']:.2f}")
                    print(f"    Weak Signal: {'Yes' if stats['is_weak'] else 'No'}")
            else:
                print("No signal data available")

def load_wildppg_participant(path: str) -> Dict:
        """
        Load and clean WildPPG participant data

        Args:
            path: Path to the .mat file

        Returns:
            Cleaned data dictionary
        """
        loaded_data = scipy.io.loadmat(path)
        loaded_data['id'] = loaded_data['id'][0]
        loaded_data['notes'] = "" if len(loaded_data['notes']) == 0 else loaded_data['notes'][0]

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


# Example usage
if __name__ == "__main__":
    # Create checker instance
    checker = WildPPGSignalQualityChecker("G:\\My Drive\\Dataset\\WildPPG")

    # Load data
    data = load_wildppg_participant("G:\\My Drive\\Dataset\\WildPPG\\WildPPG_Part_an0.mat")

    # Perform checks
    acc_results = checker.check_acc_range(data)
    weak_signal_results = checker.check_weak_signals(data)

    # Print report
    checker.print_report(acc_results, weak_signal_results)