#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Preprocessing Module

Provides ECG signal preprocessing functions including filtering, denoising, and baseline wander removal.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, filtfilt, sosfilt
import pywt
import neurokit2 as nk

class SignalPreprocessor:
    """ECG Signal Preprocessing Class"""
    
    def __init__(self, sampling_rate=500, wavelet_type='db4'):
        """Initialize preprocessor
        
        Args:
            sampling_rate: Signal sampling rate (Hz)
            wavelet_type: Wavelet transform type
        """
        self.sampling_rate = sampling_rate
        self.wavelet_type = wavelet_type
    
    def filter_signal(self, data, lowcut=0.5, highcut=40.0, order=4):
        """Bandpass filter
        
        Args:
            data: Raw signal data
            lowcut: Low frequency cutoff
            highcut: High frequency cutoff
            order: Filter order
            
        Returns:
            Filtered signal
        """
        try:
            # Check data length, reduce filter order if too short
            data_len = len(data)
            if data_len < 50:
                print(f"Warning: Data length too short ({data_len} points), cannot perform effective filtering")
                return data
            
            # Reduce filter order for short sequences
            effective_order = min(order, max(1, data_len // 20))
            
            nyq = 0.5 * self.sampling_rate
            low = lowcut / nyq
            high = highcut / nyq
            
            # Ensure cutoff frequencies are within valid range
            if low >= 1.0 or high >= 1.0:
                print("Warning: Cutoff frequency must be less than half of Nyquist frequency")
                if low >= 1.0:
                    low = 0.95
                if high >= 1.0:
                    high = 0.95
            
            # Use butter from scipy.signal, use sos format for improved numerical stability
            sos = butter(effective_order, [low, high], btype='band', output='sos')
            # Ensure sos is not None
            if sos is not None:
                filtered_data = sosfilt(sos, data)
                return filtered_data
            else:
                print("Warning: Filter parameters error, returning original data")
                return data
        except Exception as e:
            print(f"Signal filtering failed: {str(e)}")
            return data
    
    def remove_outliers(self, data, threshold=3.0):
        """Remove outliers
        
        Args:
            data: Signal data
            threshold: Outlier threshold (multiple of standard deviation)
            
        Returns:
            Signal with outliers removed
        """
        if len(data) == 0:
            return data
            
        # Calculate mean and standard deviation
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return data
            
        # Set upper and lower bounds
        lower_bound = mean_val - threshold * std_val
        upper_bound = mean_val + threshold * std_val
        
        # Clip outliers
        clipped_data = np.clip(data, lower_bound, upper_bound)
        
        return clipped_data
    
    def normalize_signal(self, data):
        """Signal normalization
        
        Args:
            data: Signal data
            
        Returns:
            Normalized signal
        """
        if len(data) == 0:
            return data
            
        # Calculate max and min values
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Avoid division by zero
        if max_val == min_val:
            return np.zeros_like(data)
            
        # Normalize to [-1, 1] range
        normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
        
        return normalized_data
    
    def remove_baseline_wander(self, data, cutoff=0.5):
        """Remove baseline wander
        
        Args:
            data: Signal data
            cutoff: Cutoff frequency
            
        Returns:
            Signal with baseline wander removed
        """
        try:
            # Check data length, skip processing if too short
            data_len = len(data)
            if data_len < 50:
                print(f"Warning: Data length too short ({data_len} points), cannot perform baseline wander removal")
                return data
            
            nyq = 0.5 * self.sampling_rate
            cutoff_norm = cutoff / nyq
            
            # Ensure cutoff frequency is within valid range
            if cutoff_norm >= 1.0:
                print("Warning: Cutoff frequency must be less than half of Nyquist frequency")
                cutoff_norm = 0.95
            
            # Use butter from scipy.signal, use sos format for improved numerical stability
            sos = butter(1, cutoff_norm, btype='high', output='sos')
            # Ensure sos is not None
            if sos is not None:
                filtered_data = sosfilt(sos, data)
                return filtered_data
            else:
                print("Warning: Filter parameters error, returning original data")
                return data
        except Exception as e:
            print(f"Baseline wander removal failed: {str(e)}")
            return data
    
    def denoise_wavelet(self, data, wavelet=None, level=4):
        """Wavelet denoising
        
        Args:
            data: Signal data
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Denoised signal
        """
        if wavelet is None:
            wavelet = self.wavelet_type
            
        # Reduce decomposition level if signal is too short
        if len(data) < 2**(level + 1):
            level = int(np.log2(len(data))) - 1
            level = max(1, level)  # At least 1
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # Threshold processing
        threshold = np.sqrt(2 * np.log(len(data)))
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # Signal reconstruction
        return pywt.waverec(coeffs, wavelet)
    
    def detect_r_peaks(self, signal_data):
        """Detect R peaks
        
        Args:
            signal_data: ECG signal data
            
        Returns:
            List of R peak positions
        """
        try:
            # Check data length
            if len(signal_data) < 200:  # Too short signal cannot reliably detect R peaks
                print(f"Warning: Signal length too short ({len(signal_data)} points), R peak detection may be unreliable")
                if len(signal_data) < 50:  # Very short signal, return empty list directly
                    return []
            
            # Signal preprocessing - use simplified processing for short sequences
            if len(signal_data) < 500:
                # Simple denoising
                from scipy.signal import medfilt
                filtered_signal = medfilt(signal_data, kernel_size=3)
            else:
                # Full preprocessing for longer signals
                filtered_signal = self.filter_signal(signal_data, lowcut=5, highcut=15)  # Focus on R wave frequency range
            
            # Use neurokit2 to detect R peaks with lower detection threshold
            try:
                _, info = nk.ecg_peaks(filtered_signal, sampling_rate=self.sampling_rate, method='neurokit')
                r_peaks = info['ECG_R_Peaks']
                
                # Validate R peak count
                if len(r_peaks) > 2:
                    return r_peaks
            except Exception as e:
                print(f"neurokit2 R peak detection failed: {str(e)}")
                # Continue trying backup method
            
            # Backup method: use simple peak detection, lower threshold to improve detection rate
            try:
                from scipy.signal import find_peaks
                
                # Calculate signal variance to determine threshold, lower threshold to increase detection rate
                signal_std = np.std(filtered_signal)
                # Lower threshold from 0.5 to 0.3
                height = 0.3 * signal_std if signal_std > 0 else 0.05
                
                # Peak detection, reduce minimum distance requirement
                peaks, _ = find_peaks(filtered_signal, height=height, 
                                     distance=int(0.15 * self.sampling_rate))
                
                if len(peaks) > 2:
                    return peaks
                else:
                    # If still cannot detect enough peaks, further lower threshold
                    height = 0.2 * signal_std if signal_std > 0 else 0.01
                    peaks, _ = find_peaks(filtered_signal, height=height, 
                                         distance=int(0.1 * self.sampling_rate))
                    return peaks if len(peaks) > 0 else []
            except Exception as e:
                print(f"Backup R peak detection failed: {str(e)}")
                return []
        except Exception as e:
            print(f"R peak detection failed: {str(e)}")
            return []
    
    def segment_beats(self, signal_data, r_peaks, before_r=0.2, after_r=0.4):
        """Segment beats based on R peaks
        
        Args:
            signal_data: ECG signal data
            r_peaks: List of R peak positions
            before_r: Time window before R peak (seconds)
            after_r: Time window after R peak (seconds)
            
        Returns:
            List of segmented beats
        """
        beats = []
        before_samples = int(before_r * self.sampling_rate)
        after_samples = int(after_r * self.sampling_rate)
        
        for r_peak in r_peaks:
            # Ensure window is within signal range
            start = max(0, r_peak - before_samples)
            end = min(len(signal_data), r_peak + after_samples)
            
            # Extract beat segment
            beat = signal_data[start:end]
            beats.append(beat)
        
        return beats
    
    def preprocess_signal(self, signal_data):
        """Complete signal preprocessing pipeline
        
        Args:
            signal_data: Raw ECG signal data
            
        Returns:
            Preprocessed signal
        """
        try:
            # 1. Baseline wander removal
            signal_no_baseline = self.remove_baseline_wander(signal_data)
            
            # 2. Bandpass filtering
            signal_filtered = self.filter_signal(signal_no_baseline)
            
            # 3. Wavelet denoising
            signal_clean = self.denoise_wavelet(signal_filtered)
            
            return signal_clean
        except Exception as e:
            print(f"Signal preprocessing failed: {str(e)}")
            # If processing fails, return original signal
            return signal_data
    
    def prepare_multi_lead_data(self, data):
        """Process multi-lead data
        
        Args:
            data: Multi-lead ECG data (DataFrame)
            
        Returns:
            Preprocessed multi-lead data
        """
        # Identify lead columns
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        present_leads = [lead for lead in leads if lead in data.columns]
        
        if not present_leads:
            print("Warning: Standard lead names not found, trying to find other possible lead columns")
            # Try to find numeric columns as leads
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                present_leads = numeric_cols
        
        processed_data = data.copy()
        
        # Preprocess each lead
        for lead in present_leads:
            try:
                lead_data = data[lead].values
                if len(lead_data) > 10:  # Ensure enough data points
                    processed_data[lead] = self.preprocess_signal(lead_data)
            except Exception as e:
                print(f"Failed to process lead {lead}: {str(e)}")
        
        return processed_data
