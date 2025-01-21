import numpy as np
from scipy import signal
from typing import Dict, Union, Tuple


class PPGQualityAssessment:
    """
    A class to assess PPG signal quality based on the paper:
    "On-Device Integrated PPG Quality Assessment and Sensor Disconnection/Saturation Detection System"

    The class implements three main checks:
    1. Nearly Zero Amplitude (NZA) detection
    2. Signal Saturation (SS) detection
    3. First Order Predictor Coefficient (FOPC) based quality assessment
    """

    def __init__(self, sampling_rate: int = 128):
        """
        Initialize the PPG quality assessment class.

        Args:
            sampling_rate: Sampling frequency of the PPG signal (default: 128Hz)
        """
        self.fs = sampling_rate

        # Thresholds for quality assessment
        self.nza_threshold = 10  # Nearly zero amplitude threshold
        self.saturation_threshold = 1020  # Signal saturation threshold
        self.fopc_threshold_high = 0.93  # FOPC threshold for good quality
        self.fopc_threshold_low = -0.5  # FOPC threshold for noisy signal
        self.noise_level = 0.1  # Random noise level (10% of signal range)

    def check_signal_quality(self, signal: np.ndarray) -> Dict:
        """
        Main method to check PPG signal quality.

        Args:
            signal: Input PPG signal array

        Returns:
            Dictionary containing quality assessment results
        """
        results = {
            'is_nza': False,
            'is_saturated': False,
            'quality_level': None,  # 'good', 'motion_artifact', or 'noisy'
            'fopc_value': None
        }

        # Check for NZA
        if self._is_nza(signal):
            results['is_nza'] = True
            results['quality_level'] = 'noisy'
            return results

        # Check for saturation
        if self._is_saturated(signal):
            results['is_saturated'] = True
            results['quality_level'] = 'noisy'
            return results

        # Compute FOPC and determine quality
        fopc = self._compute_fopc(signal)
        results['fopc_value'] = fopc

        # Determine quality level based on FOPC
        if fopc > self.fopc_threshold_high:
            results['quality_level'] = 'good'
        elif fopc > self.fopc_threshold_low:
            results['quality_level'] = 'motion_artifact'
        else:
            results['quality_level'] = 'noisy'

        return results

    def _is_nza(self, signal: np.ndarray) -> bool:
        """
        Check if signal has nearly zero amplitude.

        Args:
            signal: Input PPG signal array

        Returns:
            Boolean indicating if signal has nearly zero amplitude
        """
        return np.max(np.abs(signal)) < self.nza_threshold

    def _is_saturated(self, signal: np.ndarray) -> bool:
        """
        Check if signal is saturated.

        Args:
            signal: Input PPG signal array

        Returns:
            Boolean indicating if signal is saturated
        """
        # Check for consecutive samples near max/min values
        is_max_saturated = np.any(signal > self.saturation_threshold)
        is_min_saturated = np.any(signal < 3)  # As per paper

        return is_max_saturated or is_min_saturated

    def _compute_fopc(self, signal: np.ndarray) -> float:
        """
        Compute First Order Predictor Coefficient (FOPC) for signal quality assessment.

        Args:
            signal: Input PPG signal array

        Returns:
            FOPC value
        """
        # Compute differential signal
        diff_signal = np.diff(signal)

        # Normalize signal
        if np.max(np.abs(diff_signal)) > 0:
            norm_signal = diff_signal / np.max(np.abs(diff_signal))
        else:
            return -1.0  # Return low quality indicator if signal is constant

        # Add random noise
        noise = np.random.normal(0, self.noise_level, len(norm_signal))
        noisy_signal = norm_signal + noise

        # Compute autocorrelation for lag 0 and 1
        auto_corr = np.correlate(noisy_signal, noisy_signal, mode='full')
        center = len(auto_corr) // 2
        r0 = auto_corr[center]
        r1 = auto_corr[center + 1]

        # Compute FOPC
        if r0 > 0:
            fopc = r1 / r0
        else:
            fopc = -1.0

        return fopc

    def process_ppg_segment(self, signal: np.ndarray,
                            window_size: int = 640) -> Dict[str, list]:
        """
        Process PPG signal in segments and return quality assessment for each segment.

        Args:
            signal: Input PPG signal array
            window_size: Size of each segment in samples (default: 5 seconds at 128Hz)

        Returns:
            Dictionary containing lists of quality assessments for each segment
        """
        results = {
            'is_nza': [],
            'is_saturated': [],
            'quality_level': [],
            'fopc_value': []
        }

        # Process signal in segments
        for i in range(0, len(signal), window_size):
            segment = signal[i:i + window_size]
            if len(segment) == window_size:  # Only process complete segments
                segment_result = self.check_signal_quality(segment)
                for key in results.keys():
                    results[key].append(segment_result[key])

        return results