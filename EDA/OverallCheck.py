import os
import json
from datetime import datetime
import numpy as np
import scipy
from AbnormalSignalCheck import WildPPGSignalQualityChecker
from AbnormalSignalCheck import load_wildppg_participant
from MissingRateCheck import WildPPGMissingChecker
from SamplingRateCheck import WildPPGSamplingChecker
import sys
from io import StringIO
import contextlib

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.bool_)):  # 添加np.bool_到整数类型检查中
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class WildPPGQualityCheck:
    """
    Comprehensive quality check for WildPPG dataset
    Combines signal quality, missing rate, and sampling rate checks
    """

    def __init__(self, data_path: str):
        """
        Initialize quality checkers

        Args:
            data_path: Path to the WildPPG dataset
        """
        self.data_path = data_path
        self.signal_checker = WildPPGSignalQualityChecker(data_path)
        self.missing_checker = WildPPGMissingChecker(data_path)
        self.sampling_checker = WildPPGSamplingChecker(data_path)

    @contextlib.contextmanager
    def capture_output(self):
        """Capture stdout to a string"""
        stdout = StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout
        try:
            yield stdout
        finally:
            sys.stdout = old_stdout

    def check_single_file(self, filename: str) -> dict:
        """
        Perform all quality checks on a single file

        Args:
            filename: Name of the .mat file

        Returns:
            Dictionary containing all check results
        """
        file_path = os.path.join(self.data_path, filename)
        data = load_wildppg_participant(file_path)

        with self.capture_output() as output:
            acc_results = self.signal_checker.check_acc_range(data)
            weak_signal_results = self.signal_checker.check_weak_signals(data)
            self.signal_checker.print_report(acc_results, weak_signal_results)

            missing_report = self.missing_checker.generate_report(filename)
            self.missing_checker.print_report(missing_report)

            within_results = self.sampling_checker.check_within_location(data)
            between_results = self.sampling_checker.check_between_locations(data)
            self.sampling_checker.print_report(within_results, between_results)

        results = {
            'filename': filename,
            'participant_id': data['id'],
            'timestamp': datetime.now().isoformat(),
            'signal_quality': {
                'accelerometer': acc_results,
                'weak_signals': weak_signal_results
            },
            'missing_rate': missing_report,
            'sampling_rate': {
                'within_location': within_results,
                'between_location': between_results
            },
            'detailed_report': output.getvalue()
        }

        return results

    def check_all_files(self) -> dict:
        """
        Check all .mat files in the dataset

        Returns:
            Dictionary containing results for all files
        """
        all_results = {}
        mat_files = [f for f in os.listdir(self.data_path)
                     if f.startswith('WildPPG_Part_') and f.endswith('.mat')]

        total_files = len(mat_files)
        for i, filename in enumerate(mat_files, 1):
            print(f"\nProcessing file {i}/{total_files}: {filename}")
            try:
                results = self.check_single_file(filename)
                all_results[filename] = results
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                all_results[filename] = {'error': str(e)}

        return all_results

    def save_results(self, results: dict, output_dir: str = 'quality_check_results'):
        """
        Save results to files

        Args:
            results: Dictionary containing all check results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON results using custom encoder
        json_path = os.path.join(output_dir, f'quality_check_results_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        # Save detailed text report
        report_path = os.path.join(output_dir, f'quality_check_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            for filename, result in results.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"File: {filename}\n")
                f.write(f"Participant ID: {result.get('participant_id', 'N/A')}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(result.get('detailed_report', 'No detailed report available'))
                f.write('\n')

        print(f"\nResults saved to:")
        print(f"JSON results: {json_path}")
        print(f"Detailed report: {report_path}")


# Example usage
if __name__ == "__main__":
    data_path = r'G:\My Drive\Dataset\WildPPG'
    checker = WildPPGQualityCheck(data_path)
    results = checker.check_all_files()
    checker.save_results(results)