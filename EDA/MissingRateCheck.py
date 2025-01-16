import numpy as np
import scipy.io
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class WildPPGMissingChecker:
    """
    A class to check the quality of WildPPG dataset
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']
        # Define all expected sensors
        self.sensors = {
            'ppg': ['ppg_g', 'ppg_ir', 'ppg_r'],
            'acc': ['acc_x', 'acc_y', 'acc_z'],
            'others': ['altitude', 'temperature']
        }
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

    def print_report(self, report: Dict):
        """
        Print the report in a readable format
        """
        print(f"\n=== Data Quality Report ===")
        print(f"File: {report['file_name']}")
        print(f"Participant ID: {report['participant_id']}")

        print("\n=== Missing Values Analysis ===")
        for location, sensors in report['missing_values'].items():
            print(f"\nLocation: {location}")
            has_missing = False
            for sensor, stats in sensors.items():
                if stats['total_points'] > 0:  # Only show sensors that have data
                    has_missing = True
                    print(f"  {sensor:<10}: Total Points {stats['total_points']:>8}, "
                          f"Missing Points {stats['missing_points']:>6}, "
                          f"Missing Rate {stats['missing_ratio'] * 100:>6.2f}%")
            if not has_missing:
                print("  No missing value analysis available")


    def check_missing_values(self, data: Dict) -> Dict:
        """
        Check for missing values in the dataset
        """
        missing_stats = {}

        for location in self.body_locations:
            missing_stats[location] = {}
            for sensor_type, sensors in self.sensors.items():
                for sensor in sensors:
                    try:
                        if sensor in data[location]:
                            signal = data[location][sensor]['v']
                            total_points = len(signal)
                            missing_points = np.sum(np.isnan(signal))
                            missing_ratio = missing_points / total_points if total_points > 0 else 1.0

                            missing_stats[location][sensor] = {
                                'total_points': total_points,
                                'missing_points': missing_points,
                                'missing_ratio': missing_ratio
                            }
                    except (KeyError, TypeError) as e:
                        print(f"Warning: Could not process {sensor} in {location}: {str(e)}")
                        continue

        return missing_stats

    def load_data(self, file_name: str) -> Dict:
        """
        Load a single participant's data
        Args:
            file_name: Name of the .mat file to load
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

    def generate_report(self, file_name: str) -> Dict:
        """
        Generate a comprehensive quality report for a single file
        """
        print(f"Loading file: {file_name}")
        data = self.load_data(file_name)

        report = {
            'file_name': file_name,
            'participant_id': data['id'],
            'missing_values': self.check_missing_values(data),
        }

        return report

# Create quality checker instance
checker = WildPPGMissingChecker(data_path='G:\\My Drive\\Dataset\\WildPPG')

# Check a single file
report = checker.generate_report('WildPPG_Part_an0.mat')

# Print report
checker.print_report(report)