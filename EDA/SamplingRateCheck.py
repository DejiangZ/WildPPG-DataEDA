import numpy as np
import scipy.io
import os
from typing import Dict, List, Tuple


class WildPPGSamplingChecker:
    """
    A class to check the sampling consistency of WildPPG dataset.
    Checks both within-location and between-location sampling consistency.
    """

    def __init__(self, data_path: str):
        """
        Initialize the checker with data path and sensor definitions
        """
        self.data_path = data_path
        self.body_locations = ['sternum', 'head', 'ankle', 'wrist']

        # Define sensor groups for each location
        self.location_sensors = {
            'sternum': {
                'ppg': ['ppg_g', 'ppg_ir', 'ppg_r'],
                'acc': ['acc_x', 'acc_y', 'acc_z'],
                'ecg': ['ecg'],
                'env': ['altitude', 'temperature']
            },
            'wrist': {
                'ppg': ['ppg_g', 'ppg_ir', 'ppg_r'],
                'acc': ['acc_x', 'acc_y', 'acc_z'],
                'env': ['altitude', 'temperature']
            },
            'head': {
                'ppg': ['ppg_g', 'ppg_ir', 'ppg_r'],
                'acc': ['acc_x', 'acc_y', 'acc_z'],
                'env': ['altitude', 'temperature']
            },
            'ankle': {
                'ppg': ['ppg_g', 'ppg_ir', 'ppg_r'],
                'acc': ['acc_x', 'acc_y', 'acc_z'],
                'env': ['altitude', 'temperature']
            }
        }

        # Expected sampling rates
        self.expected_fs = {
            'ppg': 128,
            'acc': 128,
            'ecg': 128,
            'env': 0.5
        }

    def check_within_location(self, data: Dict) -> Dict:
        """
        Check sampling consistency within each body location

        Args:
            data: Loaded and cleaned WildPPG data dictionary

        Returns:
            Dictionary containing sampling consistency results for each location
        """
        results = {}

        for location in self.body_locations:
            location_results = {
                'sampling_rates': {},
                'data_lengths': {},
                'consistency': True,
                'issues': []
            }

            # Check each sensor group
            for group, sensors in self.location_sensors[location].items():
                group_lengths = []
                group_rates = []

                for sensor in sensors:
                    try:
                        if sensor in data[location]:
                            signal_length = len(data[location][sensor]['v'])
                            sampling_rate = data[location][sensor]['fs']

                            group_lengths.append((sensor, signal_length))
                            group_rates.append((sensor, sampling_rate))

                            # Check against expected sampling rate
                            if abs(sampling_rate - self.expected_fs[group]) > 0.1:  # Allow small numerical errors
                                location_results['issues'].append(
                                    f"{sensor} has fs={sampling_rate}Hz (expected {self.expected_fs[group]}Hz)"
                                )
                                location_results['consistency'] = False
                    except (KeyError, TypeError) as e:
                        location_results['issues'].append(f"Error processing {sensor}: {str(e)}")
                        continue

                # Store results for this group
                location_results['sampling_rates'][group] = group_rates
                location_results['data_lengths'][group] = group_lengths

                # Check length consistency within group
                if group_lengths:
                    base_length = group_lengths[0][1]
                    for sensor, length in group_lengths[1:]:
                        if length != base_length:
                            location_results['issues'].append(
                                f"Length mismatch in {group}: {sensor}({length}) != {group_lengths[0][0]}({base_length})"
                            )
                            location_results['consistency'] = False

            results[location] = location_results

        return results

    def check_between_locations(self, data: Dict) -> Dict:
        """
        Check sampling consistency between different body locations

        Args:
            data: Loaded and cleaned WildPPG data dictionary

        Returns:
            Dictionary containing between-location sampling consistency results
        """
        results = {
            'sampling_rates': {},
            'data_lengths': {},
            'consistency': True,
            'issues': []
        }

        # Check PPG and ACC across locations
        for sensor_group in ['ppg', 'acc']:
            group_rates = {}
            group_lengths = {}

            for location in self.body_locations:
                sensors = self.location_sensors[location][sensor_group]

                # Use first sensor as representative for the group
                if sensors and sensors[0] in data[location]:
                    sensor = sensors[0]
                    rate = data[location][sensor]['fs']
                    length = len(data[location][sensor]['v'])

                    group_rates[location] = rate
                    group_lengths[location] = length

            # Check sampling rate consistency
            if group_rates:
                base_rate = list(group_rates.values())[0]
                for loc, rate in group_rates.items():
                    if abs(rate - base_rate) > 0.1:  # Allow small numerical errors
                        results['issues'].append(
                            f"{sensor_group} sampling rate mismatch: {loc}({rate}Hz) != {base_rate}Hz"
                        )
                        results['consistency'] = False

            # Check length consistency
            if group_lengths:
                base_length = list(group_lengths.values())[0]
                for loc, length in group_lengths.items():
                    if length != base_length:
                        results['issues'].append(
                            f"{sensor_group} length mismatch: {loc}({length}) != {base_length}"
                        )
                        results['consistency'] = False

            results['sampling_rates'][sensor_group] = group_rates
            results['data_lengths'][sensor_group] = group_lengths

        return results

    def print_report(self, within_results: Dict, between_results: Dict):
        """
        Print a formatted report of the sampling consistency checks
        """
        print("\n=== Within-Location Sampling Consistency Report ===")
        for location, results in within_results.items():
            print(f"\nLocation: {location}")
            print(f"Consistency: {'✓' if results['consistency'] else '✗'}")

            if results['issues']:
                print("Issues found:")
                for issue in results['issues']:
                    print(f"  - {issue}")
            else:
                print("No issues found")

            print("\nData Lengths:")
            for group, lengths in results['data_lengths'].items():
                if lengths:
                    print(f"  {group}: " + ", ".join([f"{sensor}({length})" for sensor, length in lengths]))

        print("\n=== Between-Location Sampling Consistency Report ===")
        print(f"Overall Consistency: {'✓' if between_results['consistency'] else '✗'}")

        if between_results['issues']:
            print("\nIssues found:")
            for issue in between_results['issues']:
                print(f"  - {issue}")
        else:
            print("\nNo issues found")

        print("\nCross-location comparison:")
        for sensor_group in ['ppg', 'acc']:
            print(f"\n{sensor_group.upper()} Sensors:")
            if sensor_group in between_results['sampling_rates']:
                rates = between_results['sampling_rates'][sensor_group]
                lengths = between_results['data_lengths'][sensor_group]
                for location in self.body_locations:
                    if location in rates:
                        print(f"  {location}: {rates[location]}Hz, length={lengths[location]}")

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

checker = WildPPGSamplingChecker("G:\\My Drive\\Dataset\\WildPPG")

data = load_wildppg_participant("G:\\My Drive\\Dataset\\WildPPG\\WildPPG_Part_an0.mat")

within_results = checker.check_within_location(data)
between_results = checker.check_between_locations(data)

checker.print_report(within_results, between_results)