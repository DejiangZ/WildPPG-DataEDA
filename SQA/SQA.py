import numpy as np
from typing import Dict, List, Union
import pandas as pd


class PPGQualityAssessor:
    """
    PPG signal quality assessment based on first-order predictor coefficient (FOPC)
    method from the paper "On-Device Integrated PPG Quality Assessment".

    Features:
    1. Differenced sensor signal computation
    2. Random noise addition
    3. FOPC calculation
    4. Hierarchical decision rules for quality assessment
    """

    def __init__(self):
        """Initialize the PPG quality assessor with default parameters"""
        # Thresholds for quality assessment
        self.noise_amplitude = 0.1  # Random noise amplitude (10% of signal)
        self.fopc_thresholds = {
            'noise_free': 0.93,  # Threshold for noise-free PPG
            'corrupted': -0.5  # Threshold for corrupted signal
        }
        self.amp_threshold = 1e-6   # Amplitude threshold for NZA detection
        self.saturation_count = 2  # Minimum consecutive samples for saturation

    def _compute_differenced_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute differenced sensor signal

        Args:
            signal: Input PPG signal

        Returns:
            Differenced signal
        """
        return np.diff(signal, prepend=signal[0])

    def _add_random_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add random noise to normalized signal

        Args:
            signal: Input signal

        Returns:
            Signal with added noise
        """
        # Normalize signal
        if np.max(np.abs(signal)) > 0:
            normalized = signal / np.max(np.abs(signal))
        else:
            normalized = signal

        # Add random noise
        noise = np.random.normal(0, self.noise_amplitude, size=len(signal))
        return normalized + noise

    def _compute_fopc(self, signal: np.ndarray) -> float:
        """
        Compute first-order predictor coefficient

        Args:
            signal: Input signal

        Returns:
            First-order predictor coefficient
        """
        if len(signal) < 2:
            return 0

        # Compute autocorrelation at lags 0 and 1
        r0 = np.sum(signal * signal)
        r1 = np.sum(signal[:-1] * signal[1:])

        if r0 == 0:
            return 0

        return r1 / r0

    def _check_saturation(self, signal: np.ndarray,
                          quantization_bits: int = 10) -> bool:
        """
        Check for signal saturation
        """
        # 增加信号范围检查的容忍度
        max_value = 2 ** quantization_bits - 1

        # 调整饱和阈值
        high_threshold = max_value * 0.95  # 95%最大值
        low_threshold = max_value * 0.05  # 5%最大值

        saturated_high = np.where(signal >= high_threshold)[0]
        saturated_low = np.where(signal <= low_threshold)[0]

        # 要求更多的连续样本
        self.saturation_count = 10  # 增加连续样本数要求

        for sat_indices in [saturated_high, saturated_low]:
            if len(sat_indices) >= self.saturation_count:
                for i in range(len(sat_indices) - self.saturation_count + 1):
                    if np.all(np.diff(sat_indices[i:i + self.saturation_count]) == 1):
                        return True
        return False

    def _check_nza(self, signal: np.ndarray) -> bool:
        """
        Check for nearly-zero amplitude signal

        Args:
            signal: Input signal

        Returns:
            True if signal has nearly-zero amplitude
        """
        return np.max(np.abs(signal)) < self.amp_threshold

    def assess_quality(self, signal: np.ndarray) -> Dict:
        """
        Assess PPG signal quality using hierarchical decision rules

        Args:
            signal: Input PPG signal

        Returns:
            Dictionary containing quality assessment results
        """

        results = {
            'is_nza': False,
            'is_saturated': False,
            'quality_class': None,
            'fopc_value': None
        }

        # Rule 1: Check for nearly-zero amplitude
        if self._check_nza(signal):
            results['is_nza'] = True
            results['quality_class'] = 'nza'
            return results

        # Rule 2: Check for saturation
        if self._check_saturation(signal):
            results['is_saturated'] = True
            results['quality_class'] = 'saturated'
            return results

        # Compute FOPC for quality assessment
        diff_signal = self._compute_differenced_signal(signal)
        noisy_signal = self._add_random_noise(diff_signal)
        fopc = self._compute_fopc(noisy_signal)
        results['fopc_value'] = fopc

        # Rules 3-5: Classify based on FOPC value
        if fopc > self.fopc_thresholds['noise_free']:
            results['quality_class'] = 'noise_free'
        elif fopc > self.fopc_thresholds['corrupted']:
            results['quality_class'] = 'motion_corrupted'
        else:
            results['quality_class'] = 'pulse_free_noise'

        return results

    def process_dataframe(self, df: pd.DataFrame, location: str = 'wrist',
                          signal_type: str = 'ppg_g') -> pd.DataFrame:
        """
        Process DataFrame containing PPG signals

        Args:
            df: Input DataFrame with PPG signals
            location: Body location for PPG signal
            signal_type: Type of PPG signal to assess

        Returns:
            DataFrame with quality assessment results added
        """
        # Create copy of input DataFrame
        df_assessed = df.copy()

        # Column name for the PPG signal
        signal_col = f'{location}_{signal_type}'

        # Assess quality for each window
        quality_results = []
        for _, row in df.iterrows():
            if signal_col in row:
                signal = row[signal_col]
                results = self.assess_quality(signal)
                quality_results.append(results)
            else:
                quality_results.append({
                    'is_nza': None,
                    'is_saturated': None,
                    'quality_class': 'missing_signal',
                    'fopc_value': None
                })

        # Add results to DataFrame
        df_assessed[f'{signal_col}_is_nza'] = [r['is_nza'] for r in quality_results]
        df_assessed[f'{signal_col}_is_saturated'] = [r['is_saturated'] for r in quality_results]
        df_assessed[f'{signal_col}_quality'] = [r['quality_class'] for r in quality_results]
        df_assessed[f'{signal_col}_fopc'] = [r['fopc_value'] for r in quality_results]

        return df_assessed


# Example usage with SQAbasingACC:
if __name__ == "__main__":
    # Create quality assessor
    quality_assessor = PPGQualityAssessor()

    # Example with random signal
    signal = np.random.randn(1000)
    results = quality_assessor.assess_quality(signal)
    print("Quality assessment results:", results)