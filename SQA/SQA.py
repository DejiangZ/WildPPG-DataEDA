import numpy as np
from typing import Dict, List, Union
import pandas as pd


class PPGQualityAssessor:
    """
    PPG signal quality assessment based on first-order predictor coefficient (FOPC).
    Implementation following the paper methodology:
    1. Amplitude normalization for signal preprocessing
    2. Differenced sensor signal computation
    3. Random noise addition
    4. FOPC calculation using Levinson-Durbin recursion
    """

    def __init__(self):
        """Initialize the PPG quality assessor with default parameters"""
        # Thresholds for quality assessment based on FOPC
        self.noise_amplitude = 0.1  # Random noise amplitude
        self.fopc_thresholds = {
            'noise_free': 0.93,  # Threshold for noise-free PPG
            'corrupted': -0.5  # Threshold for corrupted signal
        }

    def _compute_differenced_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute differenced sensor signal using first-order backward difference

        Args:
            signal: Input PPG signal

        Returns:
            Differenced signal
        """
        return np.diff(signal, prepend=signal[0])

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply amplitude normalization to the signal
        y[n] = signal[n]/max(|signal|)

        Args:
            signal: Input signal array

        Returns:
            Amplitude normalized signal
        """
        max_abs = np.max(np.abs(signal))
        if max_abs == 0:
            return np.zeros_like(signal)
        return signal / max_abs

    def _add_random_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add uniform random noise to normalized signal, with noise amplitude
        dynamically adjusted based on signal characteristics.

        Args:
            signal: Input signal

        Returns:
            Signal with added noise
        """
        # Calculate the dynamic noise amplitude based on signal's standard deviation
        signal_std = np.std(signal)

        # Ensure noise amplitude is at least a small value to avoid adding no noise
        dynamic_noise_amplitude = max(self.noise_amplitude * signal_std, 0.01 * np.max(np.abs(signal)))

        # Generate uniform random noise in the range [-dynamic_noise_amplitude, dynamic_noise_amplitude]
        noise = np.random.uniform(-dynamic_noise_amplitude, dynamic_noise_amplitude, size=len(signal))

        return signal + noise

    def _levinson_durbin(self, r: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Implement Levinson-Durbin recursion to compute prediction coefficients

        Args:
            r: Autocorrelation sequence
            order: Prediction order (default: 1 for FOPC)

        Returns:
            Prediction coefficients
        """
        if r[0] == 0:
            return np.zeros(order)

        # Initialize arrays
        a = np.zeros(order + 1)
        k = np.zeros(order)
        e = np.zeros(order + 1)

        # Initial values
        a[0] = 1.0
        e[0] = r[0]

        # Recursion
        for i in range(order):
            # Compute reflection coefficient
            sum_val = sum(a[j] * r[i - j] for j in range(i + 1))
            k[i] = -(r[i + 1] + sum_val) / e[i]

            # Update prediction coefficients
            a_prev = a.copy()
            for j in range(i + 1):
                a[j] = a_prev[j] + k[i] * a_prev[i - j]
            a[i + 1] = k[i]

            # Update prediction error
            e[i + 1] = e[i] * (1 - k[i] ** 2)

        return a[1:]  # Return coefficients excluding a[0]

    def _compute_fopc(self, signal: np.ndarray) -> float:
        """
        Compute first-order predictor coefficient using Levinson-Durbin algorithm

        Args:
            signal: Input signal

        Returns:
            First-order predictor coefficient
        """
        if len(signal) < 2:
            return 0
        # Compute autocorrelation for lags 0 and 1
        r = np.correlate(signal, signal, mode='full')
        # Extract relevant autocorrelation values
        r = r[len(signal) - 1:len(signal) + 1]

        # Compute FOPC using Levinson-Durbin
        fopc = self._levinson_durbin(r, order=1)
        return float(fopc[0])

    def assess_quality(self, signal: np.ndarray) -> Dict:
        """
        Assess PPG signal quality using FOPC-based classification

        Args:
            signal: Input PPG signal

        Returns:
            Dictionary containing quality assessment results
        """
        results = {
            'quality_class': None,
            'fopc_value': None
        }

        # Compute differenced signal
        diff_signal = self._compute_differenced_signal(signal)

        # Apply amplitude normalization
        normalized_signal = self._normalize_signal(diff_signal)

        # Add random noise
        noisy_signal = self._add_random_noise(normalized_signal)
        # Compute FOPC using Levinson-Durbin
        fopc = self._compute_fopc(noisy_signal)
        results['fopc_value'] = fopc

        # Classify based on FOPC value
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
                    'quality_class': 'missing_signal',
                    'fopc_value': None
                })

        # Add results to DataFrame
        df_assessed[f'{signal_col}_quality'] = [r['quality_class'] for r in quality_results]
        df_assessed[f'{signal_col}_fopc'] = [r['fopc_value'] for r in quality_results]

        return df_assessed