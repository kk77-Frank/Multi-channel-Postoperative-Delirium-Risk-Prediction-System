#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Extraction Module

Provides ECG signal feature extraction functions including time-domain, frequency-domain, and time-frequency domain features.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
import antropy
import neurokit2 as nk
import scipy.stats
import warnings
import time
import concurrent.futures

# Import GPU management module
try:
    from ecg_modules.gpu_manager import (
        get_gpu_manager, is_gpu_available, execute_on_gpu, 
        get_memory_usage, reset_gpu, print_gpu_status
    )
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    print("Warning: GPU management module import failed, will use basic GPU management")
    GPU_MANAGER_AVAILABLE = False

# Define module-level CONFIG
CONFIG = {
    'SAMPLING_RATE': 500,       # Sampling rate
    'USE_GPU': True,            # Whether to use GPU
    'MEMORY_EFFICIENT': True,   # Memory efficient mode
    'CACHE_ENABLED': True,      # Enable caching
}

# Try to import global configuration
try:
    import importlib.util
    spec = importlib.util.find_spec('config')
    
    if spec is not None and spec.loader is not None:
        try:
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            if hasattr(config_module, 'CONFIG'):
                GLOBAL_CONFIG = config_module.CONFIG
                CONFIG.update(GLOBAL_CONFIG)
                print(f"Loaded global config from config.py: {GLOBAL_CONFIG}")
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
except ImportError:
    print("config.py not found, using default configuration")
    pass

# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global variables for tracking CUDA initialization status
CUDA_INIT_ATTEMPTED = False
CUDA_INIT_SUCCESS = False
CUPY_MODULE = None

# Check if CUDA is available and set up GPU acceleration
def initialize_gpu():
    """Initialize GPU and check if available
    
    Returns:
        Boolean indicating if GPU is available
    """
    global CUDA_INIT_ATTEMPTED, CUDA_INIT_SUCCESS, CUPY_MODULE
    
    # If already attempted initialization, return result directly
    if CUDA_INIT_ATTEMPTED:
        return CUDA_INIT_SUCCESS
    
    CUDA_INIT_ATTEMPTED = True
    
    # If environment variable NO_CUPY=1 is set, force disable GPU
    if os.environ.get('NO_CUPY', '0') == '1':
        print("Environment variable NO_CUPY=1, forcing GPU acceleration disabled")
        CUDA_INIT_SUCCESS = False
        return False
    
    # If config disables GPU, return False directly
    if not CONFIG['USE_GPU']:
        print("Config disables GPU acceleration")
        CUDA_INIT_SUCCESS = False
        return False
    
    # Use GPU management module with priority
    if GPU_MANAGER_AVAILABLE:
        gpu_available = is_gpu_available()
        if gpu_available:
            # Get GPU manager and print status
            gpu_manager = get_gpu_manager(
                memory_fraction=CONFIG.get('GPU_MEMORY_FRACTION', 0.8),
                max_retries=3
            )
            
            # Get cupy module
            if gpu_manager.cupy_module is not None:
                CUPY_MODULE = gpu_manager.cupy_module
                print(f"Initialize GPU using GPU management module: {gpu_manager.device_names[gpu_manager.active_device]}")
            
            CUDA_INIT_SUCCESS = True
            return True
        else:
            print("GPU management module reports GPU unavailable")
            CUDA_INIT_SUCCESS = False
            return False
    
    # If GPU management module unavailable, use traditional method
    try:
        # First try using PyTorch to detect CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"Detected available GPU using PyTorch: {torch.cuda.get_device_name(0)}")
                # Reset CUDA cache
                torch.cuda.empty_cache()
                CUDA_INIT_SUCCESS = True
                return True
        except ImportError:
            pass
        
        # If PyTorch unavailable, try using cupy
        try:
            import cupy as cp
            CUPY_MODULE = cp
            
            # Wait briefly before creating CUDA context
            time.sleep(0.5)
            
            # Explicitly create CUDA context
            device = cp.cuda.Device(0)
            context = device.use()
            
            # Try a simple operation to verify cupy is working
            x = cp.array([1, 2, 3])
            y = cp.sum(x)
            
            # Force synchronize device to ensure operation completes
            cp.cuda.Stream.null.synchronize()
            
            print("Enabled GPU acceleration using CuPy")
            CUDA_INIT_SUCCESS = True
            return True
        except Exception as e:
            print(f"Unable to initialize CuPy: {str(e)}")
    except Exception as e:
        print(f"GPU detection failed: {str(e)}")
    
    print("Will use CPU for computation")
    CUDA_INIT_SUCCESS = False
    return False

# Initialize GPU
GPU_AVAILABLE = initialize_gpu()

class FeatureExtractor:
    """ECG Feature Extraction Class"""
    
    def __init__(self, sampling_rate=500):
        """Initialize feature extractor
        
        Args:
            sampling_rate: Signal sampling rate (Hz)
        """
        self.sampling_rate = sampling_rate
    
        # Available feature types
        self.available_features = [
            'time_domain',
            'frequency_domain',
            'nonlinear',
            'wavelet',
            'heart_rate_variability',
            'morphological'
        ]
        
        # Initialize GPU status
        self.use_gpu = GPU_AVAILABLE
        
        # Batch processing parameters - increase batch size to improve GPU utilization
        self.batch_size = 2048  # Increase default batch size to 2048, improve GPU parallelism
        
        # Initialize feature cache
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize GPU reset threshold
        self.gpu_reset_threshold = 100  # Reset GPU every 100 operations
        
        # If GPU available, warm up GPU to avoid first call delay
        if self.use_gpu:
            try:
                self._warm_up_gpu()
            except Exception as e:
                print(f"GPU warm-up failed: {str(e)}")
                self.use_gpu = False
        
        # Performance statistics
        self.gpu_operations = 0
        self.cpu_operations = 0
        self.gpu_failures = 0
        
        # Optimization parameters
        self.enable_feature_caching = True
        self.most_valuable_features = ['time_domain', 'frequency_domain', 'heart_rate_variability'] 
        self.cache_compression = True  # Whether to compress cache data
    
    def _warm_up_gpu(self):
        """Warm up GPU, perform some simple operations to initialize CUDA environment"""
        if not self.use_gpu:
            return
            
        try:
            if GPU_MANAGER_AVAILABLE:
                # Warm up using GPU management module
                def warm_up_func():
                    # Ensure CUPY_MODULE is initialized
                    if CUPY_MODULE is None:
                        raise ImportError("CUPY module not initialized")
                        
                    cp = CUPY_MODULE
                    # Use larger array for warm-up, simulate actual workload
                    test_array = cp.random.rand(5000)
                    result1 = cp.fft.fft(test_array)
                    result2 = cp.sum(cp.abs(result1))
                    return result2
                
                result = execute_on_gpu(warm_up_func)
                if result is not None:
                    print("GPU warm-up complete (using GPU management module)")
                else:
                    raise Exception("GPU warm-up failed")
            else:
                # Traditional warm-up method
                if CUPY_MODULE is not None:
                    cp = CUPY_MODULE
                    # Use larger array for warm-up, simulate actual workload
                    test_array = cp.random.rand(5000)
                    result1 = cp.fft.fft(test_array)
                    result2 = cp.sum(cp.abs(result1))
                    cp.cuda.Stream.null.synchronize()  # Ensure operation completes
                    print("GPU warm-up complete")
                else:
                    raise ImportError("CuPy module not properly initialized")
        except Exception as e:
            print(f"GPU warm-up failed, will disable GPU acceleration: {str(e)}")
            self.use_gpu = False
            
    def _safe_gpu_compute(self, func, *args, **kwargs):
        """Safely execute GPU computation, handle possible CUDA errors
        
        Args:
            func: GPU computation function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            
        Returns:
            Function computation result, or empty dict if error occurs
        """
        if not self.use_gpu:
            return {}
            
        try:
            # Increment GPU operation count
            self.gpu_operations += 1
            
            # Check if need to reset GPU status
            if self.gpu_operations > self.gpu_reset_threshold:
                if GPU_MANAGER_AVAILABLE:
                    reset_gpu()
                self.gpu_operations = 0
                
            # Execute GPU computation function
            result = func(*args, **kwargs)
            
            # If result is None, return empty dict instead of None
            if result is None:
                return {}
                
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if CUDA out of memory error
            if any(msg in error_msg for msg in ['out of memory', 'cudaerror', 'memory allocation']):
                print(f"GPU out of memory error: {str(e)}")
                
                # Try to free GPU memory and retry once
                if GPU_MANAGER_AVAILABLE:
                    reset_gpu()
                    
                # If possible, reduce batch size and retry
                try:
                    original_batch_size = self.batch_size
                    self.batch_size = max(1024, self.batch_size // 2)
                    print(f"Reduce batch size to {self.batch_size} and retry")
                    
                    # Retry computation
                    result = func(*args, **kwargs)
                    
                    # Restore original batch size
                    self.batch_size = original_batch_size
                    
                    # If result is None, return empty dict
                    if result is None:
                        return {}
                        
                    return result
                    
                except Exception as retry_error:
                    print(f"Retry failed: {str(retry_error)}")
                    # Return empty dict
                    return {}
            else:
                print(f"GPU computation error: {str(e)}")
                # Return empty dict
                return {}
                
        finally:
            # Periodic memory cleanup
            if self.gpu_operations % 5 == 0 and GPU_MANAGER_AVAILABLE:
                try:
                    if CUPY_MODULE is not None and hasattr(CUPY_MODULE, 'cuda') and hasattr(CUPY_MODULE.cuda, 'memory'):
                        CUPY_MODULE.cuda.memory.empty_cache()
                    elif GPU_MANAGER_AVAILABLE:
                        reset_gpu()
                except Exception:
                    pass
                
        # If all attempts fail, return empty dict
        return {}
    
    def extract_all_features(self, signal_data, r_peaks=None):
        """Extract all types of features
        
        Args:
            signal_data: ECG signal data
            r_peaks: List of R peak positions (optional)
            
        Returns:
            Dictionary containing all features
        """
        features = {}
        
        # Check signal data quality
        signal_length = len(signal_data)
        signal_std = np.std(signal_data)
        signal_quality_ok = signal_length > 1000 and signal_std > 0.01
        
        # Extract basic time-domain features (low computational cost)
        time_features = self.extract_time_domain_features(signal_data)
        features.update(time_features)
        
        # If R peaks provided, prioritize HRV feature extraction (high clinical value)
        if r_peaks is not None and len(r_peaks) > 1:
            hrv_features = self.extract_hrv_features(signal_data, r_peaks)
            morph_features = self.extract_morphological_features(signal_data, r_peaks)
            features.update(hrv_features)
            features.update(morph_features)
        
        # Only compute complex features when signal quality is good
        if signal_quality_ok:
            # Extract frequency domain features (high clinical significance)
            freq_features = self.extract_frequency_domain_features(signal_data)
            features.update(freq_features)
            
            # Extract simplified nonlinear features (retain most clinically valuable)
            nonlinear_features = self.extract_reduced_nonlinear_features(signal_data)
            features.update(nonlinear_features)
            
            # Only extract wavelet features when signal quality is very good (high computational cost)
            if signal_length > 2000 and signal_std > 0.05:
                wavelet_features = self.extract_wavelet_features(signal_data)
                if wavelet_features is not None:
                    features.update(wavelet_features)
        else:
            # Poor signal quality, record placeholder features to maintain feature vector consistency
            freq_keys = ['freq_total_power', 'freq_vlf_power', 'freq_lf_power', 'freq_hf_power', 
                         'freq_lf_hf_ratio', 'freq_spec_entropy', 'freq_peak_freq']
            nonlinear_keys = ['nonlinear_sample_entropy', 'nonlinear_app_entropy', 
                             'nonlinear_hjorth_mobility', 'nonlinear_hjorth_complexity']
                             
            # Fill with null values
            for key in freq_keys + nonlinear_keys:
                features[key] = np.nan
        
        # Clean up GPU memory after completing large computations
        if self.use_gpu and GPU_MANAGER_AVAILABLE and (self.gpu_operations % 10 == 0):
            reset_gpu()
        
        return features
    
    def extract_time_domain_features(self, signal_data):
        """Extract time-domain features
        
        Args:
            signal_data: ECG signal data
            
        Returns:
            Time-domain feature dictionary
        """
        features = {}
        
        try:
            # Basic statistical features
            features['mean'] = np.mean(signal_data)
            features['std'] = np.std(signal_data)
            features['var'] = np.var(signal_data)
            features['rms'] = np.sqrt(np.mean(np.square(signal_data)))
            features['kurt'] = stats.kurtosis(signal_data)
            features['skew'] = stats.skew(signal_data)
            
            # Maximum, minimum, and range
            features['max'] = np.max(signal_data)
            features['min'] = np.min(signal_data)
            features['range'] = features['max'] - features['min']
            features['max_min_ratio'] = features['max'] / (abs(features['min']) + 1e-10)
            
            # Percentiles
            features['p25'] = np.percentile(signal_data, 25)
            features['p50'] = np.percentile(signal_data, 50)  # Median
            features['p75'] = np.percentile(signal_data, 75)
            features['iqr'] = features['p75'] - features['p25']  # Interquartile range
            
            # Crest and peak factors
            features['crest_factor'] = features['max'] / (features['rms'] + 1e-10)
            features['peak_factor'] = features['max'] / (features['mean'] + 1e-10)
            features['impulse_factor'] = features['max'] / (np.mean(np.abs(signal_data)) + 1e-10)
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
            features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
            
            # Mean crossing rate
            features['mean_crossing_rate'] = np.sum(np.diff((signal_data - features['mean']) > 0) != 0) / len(signal_data)
            
        except Exception as e:
            print(f"Time-domain feature extraction failed: {str(e)}")
            # Return null values
            for feature in ['mean', 'std', 'var', 'rms', 'kurt', 'skew', 'max', 'min', 
                           'range', 'max_min_ratio', 'p25', 'p50', 'p75', 'iqr', 
                           'crest_factor', 'peak_factor', 'impulse_factor', 
                           'zero_crossing_rate', 'mean_crossing_rate']:
                features[feature] = np.nan
        
        return {f'time_{k}': v for k, v in features.items()}
    
    def extract_frequency_domain_features(self, signal_data):
        """Extract frequency domain features
        
        Args:
            signal_data: Input signal, 1D numpy array
            
        Returns:
            Dictionary containing frequency domain features
        """
        features = {}
        
        # Check signal validity
        if signal_data is None or len(signal_data) < 10:
            return {f'freq_{k}': np.nan for k in ['total_power', 'vlf_power', 'lf_power', 'hf_power', 
                                                'lf_hf_ratio', 'spec_entropy', 'peak_freq']}
        
        # Set frequency calculation parameters
        fs = self.sampling_rate
        
        # Define GPU computation function
        if self.use_gpu:
            try:
                def gpu_fft_compute(signal_data):
                    # Ensure CUPY_MODULE is initialized
                    if CUPY_MODULE is None:
                        raise ImportError("CUPY module not initialized")
                        
                    cp = CUPY_MODULE
                    
                    # Batch processing to improve efficiency
                    batch_size = self.batch_size
                    signal_len = len(signal_data)
                    
                    # For long signals, use batch processing
                    if signal_len > batch_size:
                        # Split into multiple batches
                        num_batches = (signal_len + batch_size - 1) // batch_size
                        batches = []
                        
                        for i in range(num_batches):
                            start = i * batch_size
                            end = min(start + batch_size, signal_len)
                            batch = signal_data[start:end]
                            # Transfer to GPU
                            gpu_batch = cp.array(batch)
                            # Compute spectrum
                            freq_batch = cp.abs(cp.fft.rfft(gpu_batch))
                            # Transfer back to CPU
                            batches.append(cp.asnumpy(freq_batch))
                        
                        # Merge results
                        freq = np.concatenate(batches)
                    else:
                        # Short signal, process directly
                        gpu_signal = cp.array(signal_data)
                        freq = cp.asnumpy(cp.abs(cp.fft.rfft(gpu_signal)))
                    
                    # Compute frequency axis
                    freqs = np.fft.rfftfreq(len(signal_data), d=1/fs)
                    
                    # Compute power spectral density
                    psd = freq ** 2 / len(signal_data)
                    
                    # Compute energy in each frequency band
                    # VLF: 0-0.04Hz, LF: 0.04-0.15Hz, HF: 0.15-0.4Hz
                    vlf_mask = freqs < 0.04
                    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                    
                    # Compute total energy in each frequency band
                    vlf_power = np.sum(psd[vlf_mask]) if np.any(vlf_mask) else 0
                    lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
                    hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
                    total_power = np.sum(psd)
                    
                    # Compute frequency domain features
                    features = {
                        'vlf_power': vlf_power,
                        'lf_power': lf_power,
                        'hf_power': hf_power,
                        'total_power': total_power,
                        'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
                        'vlf_percent': vlf_power / total_power * 100 if total_power > 0 else 0,
                        'lf_percent': lf_power / total_power * 100 if total_power > 0 else 0,
                        'hf_percent': hf_power / total_power * 100 if total_power > 0 else 0,
                        'peak_freq': freqs[np.argmax(psd)] if len(psd) > 0 else 0,
                        'spec_entropy': scipy.stats.entropy(psd + 1e-10)
                    }
                    
                    return features
                
                # Use safe GPU computation
                freq_features = self._safe_gpu_compute(gpu_fft_compute, signal_data)
                if freq_features is not None:
                    features.update(freq_features)
                    return {f'freq_{k}': v for k, v in features.items()}
                # If GPU computation returns None, fall back to CPU computation
                
            except Exception as e:
                print(f"Failed to compute frequency domain features using GPU: {str(e)}, falling back to CPU computation")
        
        # CPU computation logic
        try:
            # Use scipy to compute spectrum
            freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(512, len(signal_data)))
            
            # Compute energy in each frequency band
            vlf_mask = freqs < 0.04
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            
            vlf_power = np.sum(psd[vlf_mask]) if np.any(vlf_mask) else 0
            lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
            hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
            total_power = np.sum(psd)
            
            # Compute frequency domain features
            features['total_power'] = total_power
            features['vlf_power'] = vlf_power
            features['lf_power'] = lf_power
            features['hf_power'] = hf_power
            features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
            features['peak_freq'] = freqs[np.argmax(psd)] if len(psd) > 0 else 0
            features['spec_entropy'] = scipy.stats.entropy(psd + 1e-10)
        except Exception as e:
            print(f"Frequency domain feature extraction failed: {str(e)}")
            # Return null values
            for feature in ['total_power', 'vlf_power', 'lf_power', 'hf_power', 
                           'lf_hf_ratio', 'spec_entropy', 'peak_freq']:
                features[feature] = np.nan
        
        return {f'freq_{k}': v for k, v in features.items()}
    
    def extract_reduced_nonlinear_features(self, signal_data):
        """Extract reduced nonlinear features, focusing on computing most clinically meaningful features
        
        Args:
            signal_data: ECG signal data
            
        Returns:
            Nonlinear feature dictionary (reduced)
        """
        features = {}
        
        try:
            # Sample entropy and approximate entropy
            features['sample_entropy'] = antropy.sample_entropy(signal_data)
            features['app_entropy'] = antropy.app_entropy(signal_data)
            
            # Hjorth parameters (low computational cost but high value)
            features['hjorth_mobility'], features['hjorth_complexity'] = antropy.hjorth_params(signal_data)
            
            # Permutation entropy (stable to noise, clinically meaningful)
            features['perm_entropy'] = antropy.perm_entropy(signal_data, normalize=True)
            
            # Selectively compute some high-cost features
            signal_length = len(signal_data)
            if signal_length > 1500:  # Only compute for longer signals
                # Spectral entropy
                features['spectral_entropy'] = antropy.spectral_entropy(signal_data, sf=self.sampling_rate, method='welch', normalize=True)
            
                # Petrosian fractal dimension (moderate computational cost)
                features['petrosian_fd'] = antropy.petrosian_fd(signal_data)
            else:
                features['spectral_entropy'] = np.nan
                features['petrosian_fd'] = np.nan
                
        except Exception as e:
            print(f"Reduced nonlinear feature extraction failed: {str(e)}")
            # Return null values
            for feature in ['sample_entropy', 'app_entropy', 'perm_entropy', 
                           'hjorth_mobility', 'hjorth_complexity', 
                           'spectral_entropy', 'petrosian_fd']:
                features[feature] = np.nan
        
        return {f'nonlinear_{k}': v for k, v in features.items()}
    
    def extract_wavelet_features(self, signal_data, wavelet='db4', level=4):
        """Extract wavelet features
        
        Args:
            signal_data: Input signal, 1D numpy array
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary containing wavelet features
        """
        features = {}
        
        # Check signal validity
        if signal_data is None or len(signal_data) < 10:
            # Return empty dict instead of None
            return {f'wavelet_{k}': np.nan for k in ['entropy', 'energy', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4']}
        
        # Set wavelet computation parameters
        wavelet_type = wavelet
        decomp_level = min(level, int(np.log2(len(signal_data))) - 2)
        decomp_level = max(decomp_level, 1)  # At least one level of decomposition
        
        # Define GPU computation function
        if self.use_gpu:
            try:
                def gpu_wavelet_compute(signal_data):
                    # Ensure CUPY_MODULE is initialized
                    if CUPY_MODULE is None:
                        raise ImportError("CUPY module not initialized")
                        
                    cp = CUPY_MODULE
                    
                    # Create GPU array from signal data
                    signal_gpu = cp.array(signal_data)
                    
                    try:
                        # Try to use CuPy for wavelet transform
                        # If CuPy doesn't have wavelet functions, fall back to CPU computation
                        if hasattr(cp, 'wavelets') and hasattr(cp.wavelets, 'wavedec'):
                            coeffs = cp.wavelets.wavedec(signal_gpu, wavelet_type, level=decomp_level)
                            # Extract features
                            features = {}
                            
                            # Compute total energy and band energy
                            total_energy = 0
                            band_energy = []
                            
                            for i, coeff in enumerate(coeffs):
                                energy = cp.sum(coeff**2)
                                band_energy.append(float(energy))
                                total_energy += float(energy)
                            
                            # Compute band energy ratio
                            if total_energy > 0:
                                for i, energy in enumerate(band_energy):
                                    features[f'wavelet_ratio_{i+1}'] = float(energy / total_energy)
                            
                            # Compute Shannon entropy
                            cA = coeffs[0]  # Approximation coefficients
                            cD = coeffs[1:]  # Detail coefficients
                            
                            # Compute entropy of approximation coefficients
                            p_values = cp.abs(cA)**2 / cp.sum(cp.abs(cA)**2)
                            entropy_cA = -cp.sum(p_values * cp.log2(p_values + 1e-10))
                            features['wavelet_entropy'] = float(entropy_cA)
                            
                            # Compute energy
                            features['wavelet_energy'] = float(total_energy)
                            
                            return features
                        else:
                            # If CuPy doesn't have wavelet functions, return empty dict and let CPU computation handle it
                            return {}
                    except Exception as e:
                        print(f"GPU wavelet computation failed: {str(e)}, falling back to CPU")
                        return {}
                        
                # Safely execute GPU computation
                gpu_features = self._safe_gpu_compute(gpu_wavelet_compute, signal_data)
                
                if gpu_features:
                    return gpu_features
                else:
                    print("GPU wavelet computation failed or not executed, using CPU computation")
            except Exception as e:
                print(f"GPU wavelet feature extraction failed: {str(e)}, falling back to CPU")
                # Continue with CPU code
        
        # CPU computation - only execute when GPU computation fails or is not enabled
        try:
            # Use PyWavelets library
            import pywt
            
            # Execute wavelet transform
            coeffs = pywt.wavedec(signal_data, wavelet_type, level=decomp_level)
            
            # Compute total energy and band energy
            total_energy = 0
            band_energy = []
            
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff**2)
                band_energy.append(energy)
                total_energy += energy
            
            # Compute band energy ratio
            if total_energy > 0:
                for i, energy in enumerate(band_energy):
                    features[f'wavelet_ratio_{i+1}'] = energy / total_energy
            else:
                # If total energy is zero, set to zero or NaN
                for i in range(len(band_energy)):
                    features[f'wavelet_ratio_{i+1}'] = 0.0
            
            # Compute Shannon entropy
            cA = coeffs[0]  # Approximation coefficients
            
            # Compute entropy of approximation coefficients
            p_values = np.abs(cA)**2 / np.sum(np.abs(cA)**2)
            entropy_cA = -np.sum(p_values * np.log2(p_values + 1e-10))
            features['wavelet_entropy'] = entropy_cA
            
            # Compute energy
            features['wavelet_energy'] = total_energy
            
        except Exception as e:
            print(f"Wavelet feature extraction error: {str(e)}")
            
            # Return default values
            default_features = ['entropy', 'energy', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4']
            features = {f'wavelet_{k}': np.nan for k in default_features}
        
        return features
    
    def extract_complete_hrv_features(self, signal_data, r_peaks):
        """Extract complete advanced HRV features (using neurokit2's full analysis)
        
        Args:
            signal_data: ECG signal data
            r_peaks: List of R peak positions
            
        Returns:
            Complete HRV feature dictionary
        """
        features = {}
        
        try:
            # Check if enough R peaks
            if r_peaks is None or len(r_peaks) < 10:
                return {}
            
            # Use neurokit2's complete HRV analysis
            # This extracts all advanced features: CSI, MFDFA, MSEn, ShanEn, RCMSEn, etc.
            try:
                hrv_indices = nk.hrv(r_peaks, sampling_rate=self.sampling_rate, show=False)
                
                # Extract all HRV features
                if hrv_indices is not None and len(hrv_indices) > 0:
                    for col in hrv_indices.columns:
                        # Feature naming format: hrv_HRV_XXX (consistent with training)
                        features[f'hrv_{col}'] = hrv_indices.iloc[0][col]
                    
                    # Add lowercase aliases for compatibility with old version naming
                    # Training used lowercase naming: mean_rr, sdnn, rmssd, pnn50, triangular_index
                    if 'HRV_MeanNN' in hrv_indices.columns:
                        features['hrv_mean_rr'] = hrv_indices.iloc[0]['HRV_MeanNN']
                    if 'HRV_SDNN' in hrv_indices.columns:
                        features['hrv_sdnn'] = hrv_indices.iloc[0]['HRV_SDNN']
                    if 'HRV_RMSSD' in hrv_indices.columns:
                        features['hrv_rmssd'] = hrv_indices.iloc[0]['HRV_RMSSD']
                    if 'HRV_pNN50' in hrv_indices.columns:
                        features['hrv_pnn50'] = hrv_indices.iloc[0]['HRV_pNN50']
                    if 'HRV_HTI' in hrv_indices.columns:
                        # HTI (HRV Triangular Index)
                        features['hrv_triangular_index'] = hrv_indices.iloc[0]['HRV_HTI']
                
            except Exception as e:
                print(f"Complete HRV analysis failed: {str(e)}")
                # Fall back to basic HRV features on failure
                pass
                
        except Exception as e:
            print(f"Advanced HRV feature extraction failed: {str(e)}")
        
        return features
    
    def extract_hrv_features(self, signal_data, r_peaks):
        """Extract heart rate variability features (retain original function for compatibility, but prioritize complete HRV)
        
        Args:
            signal_data: ECG signal data
            r_peaks: List of R peak positions
            
        Returns:
            HRV feature dictionary
        """
        # First try to extract complete advanced HRV features
        advanced_features = self.extract_complete_hrv_features(signal_data, r_peaks)
        if advanced_features:
            return advanced_features
        
        # If complete analysis fails, use basic HRV features
        features = {}
        
        try:
            # Check if enough R peaks
            if r_peaks is None or len(r_peaks) < 2:
                # Insufficient R peaks, return default values
                features['mean_rr'] = np.nan
                features['sdnn'] = np.nan
                features['rmssd'] = np.nan
                features['pnn50'] = np.nan
                features['triangular_index'] = np.nan
                return {f'hrv_{k}': v for k, v in features.items()}
                
            # Compute RR intervals (in milliseconds)
            rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
            
            # Filter outliers
            if len(rr_intervals) > 0:
                # Filter extreme RR intervals (e.g., less than 300ms or greater than 2000ms)
                valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
                
                # If not enough valid RR intervals, use original RR intervals
                if len(valid_rr) < 2:
                    valid_rr = rr_intervals
            
                # Time-domain HRV features
                features['mean_rr'] = np.mean(valid_rr)
                features['sdnn'] = np.std(valid_rr)
                
                # For features requiring at least 2 RR interval differences, check if enough data
                if len(valid_rr) >= 2:
                    features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(valid_rr))))
                    features['pnn50'] = np.sum(np.abs(np.diff(valid_rr)) > 50) / len(valid_rr) * 100
                else:
                    features['rmssd'] = np.nan
                    features['pnn50'] = np.nan
                
                # Geometric HRV features require at least 5 RR intervals
                if len(valid_rr) >= 5:
                    try:
                        hist_bins = range(int(min(valid_rr)), int(max(valid_rr)) + 1)
                        if len(hist_bins) > 1:  # Ensure enough bins
                            hist_values = np.histogram(valid_rr, bins=hist_bins)[0]
                            if np.max(hist_values) > 0:
                                features['triangular_index'] = len(valid_rr) / (np.max(hist_values) + 1e-10)
                            else:
                                features['triangular_index'] = np.nan
                        else:
                            features['triangular_index'] = np.nan
                    except Exception:
                        features['triangular_index'] = np.nan
                else:
                    features['triangular_index'] = np.nan
                
                # Frequency domain and nonlinear HRV features require more data points
                # For short time series, skip these complex features to avoid errors
                if len(valid_rr) >= 10:  # Increase minimum RR interval count requirement
                    try:
                        # Convert RR intervals to continuous peak indices
                        peaks = np.zeros(len(valid_rr) + 1)
                        for i in range(1, len(peaks)):
                            peaks[i] = peaks[i-1] + valid_rr[i-1]
                            
                        # Convert time unit from milliseconds to seconds
                        peaks = peaks / 1000.0
                        
                        # Use neurokit2 to compute frequency domain HRV features
                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                        
                        # Extract frequency domain features
                        for col in hrv_freq.columns:
                            features[col] = hrv_freq.iloc[0][col]
                            
                            # Only compute nonlinear features when enough data points
                            if len(valid_rr) >= 20:
                                # Use same peak indices as frequency domain
                                hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
                                
                                # Extract nonlinear features
                                for col in hrv_nonlinear.columns:
                                    features[col] = hrv_nonlinear.iloc[0][col]
                    except Exception as e:
                        print(f"Advanced HRV feature computation failed: {str(e)}")
            
        except Exception as e:
            print(f"HRV feature extraction failed: {str(e)}")
            # Return basic null values
            for feature in ['mean_rr', 'sdnn', 'rmssd', 'pnn50', 'triangular_index']:
                features[feature] = np.nan
            
        return {f'hrv_{k}': v for k, v in features.items()}
    
    def extract_morphological_features(self, signal_data, r_peaks):
        """Extract morphological features
        
        Args:
            signal_data: ECG signal data
            r_peaks: List of R peak positions
            
        Returns:
            Morphological feature dictionary
        """
        features = {}
        
        try:
            # Check R peak count
            if r_peaks is None or len(r_peaks) < 5:
                return {}
            
            # Use neurokit2 for waveform segmentation (more robust method)
            try:
                _, waves_peak = nk.ecg_delineate(signal_data, r_peaks, sampling_rate=self.sampling_rate, method="peaks")
            except:
                waves_peak = {}
            
            try:
                _, waves_dwt = nk.ecg_delineate(signal_data, r_peaks, sampling_rate=self.sampling_rate, method="dwt")
            except:
                waves_dwt = {}
            
            # Merge results from both methods
            waves = {}
            for key in list(set(list(waves_peak.keys()) + list(waves_dwt.keys()))):
                if key in waves_peak and key in waves_dwt:
                    # If both methods have results, choose the one with more non-None values
                    peak_valid = sum(1 for x in waves_peak[key] if x is not None and not np.isnan(x))
                    dwt_valid = sum(1 for x in waves_dwt[key] if x is not None and not np.isnan(x))
                    
                    if peak_valid >= dwt_valid:
                        waves[key] = waves_peak[key]
                    else:
                        waves[key] = waves_dwt[key]
                elif key in waves_peak:
                    waves[key] = waves_peak[key]
                elif key in waves_dwt:
                    waves[key] = waves_dwt[key]
            
            # Filter None values
            for key in list(waves.keys()):
                waves[key] = [x for x in waves[key] if x is not None and not np.isnan(x)]
            
            # If waveform segmentation fails, use simple ST segment estimation
            if not waves or len(waves.get('ECG_S_Offsets', [])) == 0:
                # Use simple method to estimate ST segment
                st_levels = self._estimate_st_segment_simple(signal_data, r_peaks)
                if st_levels:
                    features['st_level_mean'] = np.mean(st_levels)
                    features['st_level_std'] = np.std(st_levels)
                else:
                    features['st_level_mean'] = np.nan
                    features['st_level_std'] = np.nan
            
            # Compute amplitude and duration of each wave
            wave_types = ['P', 'Q', 'R', 'S', 'T']
            
            for wave in wave_types:
                # Check if wave positions exist
                onsets_key = f'ECG_{wave}_Onsets'
                offsets_key = f'ECG_{wave}_Offsets'
                peaks_key = f'ECG_{wave}_Peaks'
                
                if onsets_key in waves and offsets_key in waves and peaks_key in waves:
                    # Ensure enough data points
                    if (len(waves[onsets_key]) > 0 and len(waves[offsets_key]) > 0 and
                        len(waves[peaks_key]) > 0 and len(waves[onsets_key]) == len(waves[offsets_key])):
                        
                        # Compute duration (in milliseconds)
                        durations = []
                        for onset, offset in zip(waves[onsets_key], waves[offsets_key]):
                            if onset is not None and offset is not None and onset < offset:
                                duration_ms = (offset - onset) / self.sampling_rate * 1000
                                durations.append(duration_ms)
                        
                        # Compute amplitude
                        amplitudes = []
                        for peak in waves[peaks_key]:
                            if peak is not None and 0 <= peak < len(signal_data):
                                amplitude = signal_data[peak]
                                amplitudes.append(amplitude)
                        
                        # Compute statistics
                        if durations:
                            features[f'{wave}_duration_mean'] = np.mean(durations)
                            features[f'{wave}_duration_std'] = np.std(durations)
                        else:
                            features[f'{wave}_duration_mean'] = np.nan
                            features[f'{wave}_duration_std'] = np.nan
                        
                        if amplitudes:
                            features[f'{wave}_amplitude_mean'] = np.mean(amplitudes)
                            features[f'{wave}_amplitude_std'] = np.std(amplitudes)
                        else:
                            features[f'{wave}_amplitude_mean'] = np.nan
                            features[f'{wave}_amplitude_std'] = np.nan
                
            # Compute specific intervals
            self._calculate_intervals(features, waves, signal_data)
            
        except Exception as e:
            print(f"Morphological feature extraction failed: {str(e)}")
            # Return some basic null values
            features['morphological_extraction_failed'] = 1
        
        return {f'morph_{k}': v for k, v in features.items()}
    
    def _estimate_st_segment_simple(self, signal_data, r_peaks):
        """Simple estimation of ST segment level (used when waveform segmentation fails)
        
        Args:
            signal_data: ECG signal data
            r_peaks: R peak positions
            
        Returns:
            List of ST segment levels
        """
        try:
            st_levels = []
            
            for i in range(len(r_peaks) - 1):
                r_peak = r_peaks[i]
                next_r = r_peaks[i + 1]
                
                # ST segment approximately 80-120ms after R peak (40-60 samples @ 500Hz)
                st_start = int(r_peak + 0.08 * self.sampling_rate)  # 80ms after R peak
                st_end = int(r_peak + 0.12 * self.sampling_rate)    # 120ms after R peak
                
                # Ensure within signal range and doesn't exceed next R peak
                if st_start < len(signal_data) and st_end < min(len(signal_data), next_r):
                    # Take average of ST segment
                    st_segment = signal_data[st_start:st_end]
                    if len(st_segment) > 0:
                        st_levels.append(np.mean(st_segment))
            
            return st_levels if st_levels else None
            
        except Exception as e:
            return None
    
    def _calculate_intervals(self, features, waves, signal_data):
        """Compute waveform intervals
        
        Args:
            features: Feature dictionary, will be modified
            waves: Waveform dictionary
            signal_data: Signal data
        """
        try:
            # Compute PR interval
            if ('ECG_P_Peaks' in waves and 'ECG_R_Peaks' in waves and
                len(waves['ECG_P_Peaks']) > 0 and len(waves['ECG_R_Peaks']) > 0):
                
                # Find nearest P peak before each R peak
                pr_intervals = []
                for r_peak in waves['ECG_R_Peaks']:
                    if r_peak is None:
                        continue
                    
                    # Find last P peak before this R peak
                    p_peaks_before = [p for p in waves['ECG_P_Peaks'] if p is not None and p < r_peak]
                    
                    if p_peaks_before:
                        last_p = max(p_peaks_before)
                        pr_interval_ms = (r_peak - last_p) / self.sampling_rate * 1000
                        if 50 < pr_interval_ms < 300:  # Reasonable PR interval range
                            pr_intervals.append(pr_interval_ms)
                
                if pr_intervals:
                    features['pr_interval_mean'] = np.mean(pr_intervals)
                    features['pr_interval_std'] = np.std(pr_intervals)
                else:
                    features['pr_interval_mean'] = np.nan
                    features['pr_interval_std'] = np.nan
            
            # Compute QRS width
            if ('ECG_Q_Onsets' in waves and 'ECG_S_Offsets' in waves and
                len(waves['ECG_Q_Onsets']) > 0 and len(waves['ECG_S_Offsets']) > 0):
                
                # Ensure length matching
                min_len = min(len(waves['ECG_Q_Onsets']), len(waves['ECG_S_Offsets']))
                
                qrs_widths = []
                for i in range(min_len):
                    q_onset = waves['ECG_Q_Onsets'][i]
                    s_offset = waves['ECG_S_Offsets'][i]
                    
                    if q_onset is not None and s_offset is not None and q_onset < s_offset:
                        qrs_width_ms = (s_offset - q_onset) / self.sampling_rate * 1000
                        if 60 < qrs_width_ms < 180:  # Reasonable QRS width range
                            qrs_widths.append(qrs_width_ms)
                
                if qrs_widths:
                    features['qrs_width_mean'] = np.mean(qrs_widths)
                    features['qrs_width_std'] = np.std(qrs_widths)
                else:
                    features['qrs_width_mean'] = np.nan
                    features['qrs_width_std'] = np.nan
            
            # Compute QT interval
            if ('ECG_Q_Onsets' in waves and 'ECG_T_Offsets' in waves and
                len(waves['ECG_Q_Onsets']) > 0 and len(waves['ECG_T_Offsets']) > 0):
                
                # Ensure length matching
                min_len = min(len(waves['ECG_Q_Onsets']), len(waves['ECG_T_Offsets']))
                
                qt_intervals = []
                for i in range(min_len):
                    q_onset = waves['ECG_Q_Onsets'][i]
                    t_offset = waves['ECG_T_Offsets'][i]
                    
                    if q_onset is not None and t_offset is not None and q_onset < t_offset:
                        qt_interval_ms = (t_offset - q_onset) / self.sampling_rate * 1000
                        if 300 < qt_interval_ms < 500:  # Reasonable QT interval range
                            qt_intervals.append(qt_interval_ms)
                
                if qt_intervals:
                    features['qt_interval_mean'] = np.mean(qt_intervals)
                    features['qt_interval_std'] = np.std(qt_intervals)
                else:
                    features['qt_interval_mean'] = np.nan
                    features['qt_interval_std'] = np.nan
            
            # Compute ST segment level
            if ('ECG_S_Offsets' in waves and 'ECG_T_Onsets' in waves and
                len(waves['ECG_S_Offsets']) > 0 and len(waves['ECG_T_Onsets']) > 0):
                
                # Ensure length matching
                min_len = min(len(waves['ECG_S_Offsets']), len(waves['ECG_T_Onsets']))
                
                st_levels = []
                for i in range(min_len):
                    s_offset = waves['ECG_S_Offsets'][i]
                    t_onset = waves['ECG_T_Onsets'][i]
                    
                    if (s_offset is not None and t_onset is not None and 
                        s_offset < t_onset and s_offset < len(signal_data) and t_onset < len(signal_data)):
                        # Select ST segment midpoint
                        st_point = (s_offset + t_onset) // 2
                        if 0 <= st_point < len(signal_data):
                            st_level = signal_data[st_point]
                            st_levels.append(st_level)
                
                if st_levels:
                    features['st_level_mean'] = np.mean(st_levels)
                    features['st_level_std'] = np.std(st_levels)
            else:
                    features['st_level_mean'] = np.nan
                    features['st_level_std'] = np.nan
                
        except Exception as e:
            print(f"Interval computation failed: {str(e)}")
    
    def extract_features(self, signal_data, r_peaks=None):
        """Extract all features
        
        Args:
            signal_data: ECG signal data
            r_peaks: R peak positions, will be auto-detected if None
            
        Returns:
            Dictionary containing all extracted features
        """
        # Check signal validity
        if signal_data is None or len(signal_data) < 10:
            print("Signal data invalid or insufficient length")
            return {}
        
        # If caching enabled, try to get result from cache
        if self.enable_feature_caching:
            # Use hash of signal data as cache key
            try:
                cache_key = hash(str(signal_data.tobytes()) + str(r_peaks))
                
                if cache_key in self.feature_cache:
                    self.cache_hits += 1
                    return self.feature_cache[cache_key].copy()
                else:
                    self.cache_misses += 1
            except Exception:
                # If hashing fails, don't use cache
                pass
        
        # Extract features based on configured priority feature types
        features = {}
        
        # Time-domain features (basic ECG features)
        try:
            time_features = self.extract_time_domain_features(signal_data)
            features.update(time_features)
        except Exception as e:
            print(f"Time-domain feature extraction failed: {str(e)}")
        
        # Frequency domain features
        try:
            freq_features = self.extract_frequency_domain_features(signal_data)
            features.update(freq_features)
        except Exception as e:
            print(f"Frequency domain feature extraction failed: {str(e)}")
        
        # Check if R peak information available
        if r_peaks is None or len(r_peaks) < 3:
            # If no R peak information provided, try auto-detection
            try:
                from ecg_modules.preprocessing import SignalPreprocessor
                preprocessor = SignalPreprocessor(self.sampling_rate)
                r_peaks = preprocessor.detect_r_peaks(signal_data)
            except Exception as e:
                print(f"R peak auto-detection failed: {str(e)}")
        
        # Only extract HRV features when R peak detection successful
        if r_peaks is not None and len(r_peaks) >= 3:
            # Heart rate variability features
            try:
                hrv_features = self.extract_hrv_features(signal_data, r_peaks)
                features.update(hrv_features)
            except Exception as e:
                print(f"Heart rate variability feature extraction failed: {str(e)}")
                
            # Morphological features
            try:
                morph_features = self.extract_morphological_features(signal_data, r_peaks)
                features.update(morph_features)
            except Exception as e:
                print(f"Morphological feature extraction failed: {str(e)}")
        
        # Reduced nonlinear feature set (high computational cost but low importance)
        if 'nonlinear' in self.most_valuable_features:
            try:
                nonlin_features = self.extract_reduced_nonlinear_features(signal_data)
                features.update(nonlin_features)
            except Exception as e:
                print(f"Nonlinear feature extraction failed: {str(e)}")
        
        # Wavelet features (high computational cost)
        if 'wavelet' in self.most_valuable_features:
            try:
                wavelet_features = self.extract_wavelet_features(signal_data)
                features.update(wavelet_features)
            except Exception as e:
                print(f"Wavelet feature extraction failed: {str(e)}")
        
        # Cache results
        if self.enable_feature_caching:
            try:
                # Check current cache size, clean old entries if exceeds limit
                if len(self.feature_cache) > 500:  # Cache maximum 500 results
                    # Clear half of cache
                    keys_to_remove = list(self.feature_cache.keys())[:250]
                    for key in keys_to_remove:
                        del self.feature_cache[key]
                        
                self.feature_cache[cache_key] = features.copy()
            except Exception:
                # If caching fails, ignore error
                pass
        
        return features
    
    def calculate_inter_lead_features(self, ecg_data_dict, r_peaks_dict):
        """Calculate inter-lead features
        
        Args:
            ecg_data_dict: Dictionary containing multiple lead data
            r_peaks_dict: Dictionary containing R peak positions for multiple leads
            
        Returns:
            Inter-lead feature dictionary
        """
        features = {}
        
        try:
            # Check if enough lead data
            if len(ecg_data_dict) < 2:
                return features
            
            # Compute inter-lead correlation
            leads = list(ecg_data_dict.keys())
            for i in range(len(leads)):
                for j in range(i+1, len(leads)):
                    lead1 = leads[i]
                    lead2 = leads[j]
                    
                    # Get data for two leads
                    data1 = ecg_data_dict[lead1]
                    data2 = ecg_data_dict[lead2]
                    
                    # Ensure same length
                    min_len = min(len(data1), len(data2))
                    data1 = data1[:min_len]
                    data2 = data2[:min_len]
                    
                    # Compute correlation coefficient
                    try:
                        corr = np.corrcoef(data1, data2)[0, 1]
                        features[f'corr_{lead1}_{lead2}'] = corr
                    except Exception:
                        features[f'corr_{lead1}_{lead2}'] = np.nan
                    
                    # Compute mutual information
                    try:
                        from sklearn.feature_selection import mutual_info_regression
                        mutual_info = mutual_info_regression(data1.reshape(-1, 1), data2, random_state=0)[0]
                        features[f'mutual_info_{lead1}_{lead2}'] = mutual_info
                    except Exception:
                        features[f'mutual_info_{lead1}_{lead2}'] = np.nan
            
            # If R peak data available, calculate R peak synchronicity
            if r_peaks_dict and len(r_peaks_dict) >= 2:
                r_peak_leads = list(r_peaks_dict.keys())
                for i in range(len(r_peak_leads)):
                    for j in range(i+1, len(r_peak_leads)):
                        lead1 = r_peak_leads[i]
                        lead2 = r_peak_leads[j]
                        
                        r_peaks1 = r_peaks_dict[lead1]
                        r_peaks2 = r_peaks_dict[lead2]
                        
                        if len(r_peaks1) > 0 and len(r_peaks2) > 0:
                            # Calculate R peak time differences
                            try:
                                # Find nearest R peak pair
                                r_peak_diffs = []
                                for r1 in r_peaks1:
                                    diffs = np.abs(np.array(r_peaks2) - r1)
                                    min_diff_idx = np.argmin(diffs)
                                    if diffs[min_diff_idx] < self.sampling_rate * 0.2:  # R peaks within 200ms
                                        r_peak_diffs.append(diffs[min_diff_idx] / self.sampling_rate * 1000)  # Convert to milliseconds
                                
                                if r_peak_diffs:
                                    features[f'r_peak_delay_{lead1}_{lead2}_mean'] = np.mean(r_peak_diffs)
                                    features[f'r_peak_delay_{lead1}_{lead2}_std'] = np.std(r_peak_diffs)
                            except Exception:
                                features[f'r_peak_delay_{lead1}_{lead2}_mean'] = np.nan
                                features[f'r_peak_delay_{lead1}_{lead2}_std'] = np.nan
                        
        except Exception as e:
            print(f"Inter-lead feature computation failed: {str(e)}")
        
        return features
    
    def extract_vectorcardiogram_features(self, ecg_data_dict):
        """Extract vectorcardiography features
        
        Args:
            ecg_data_dict: Dictionary containing multiple lead data
            
        Returns:
            Vectorcardiography feature dictionary
        """
        features = {}
        
        try:
            # Check if necessary leads exist
            required_leads = ['I', 'II']
            if not all(lead in ecg_data_dict for lead in required_leads):
                return features
                
            # 获取导联数据
            lead_I = ecg_data_dict['I']
            lead_II = ecg_data_dict['II']
            
            # 计算导联III（如果不存在）
            if 'III' not in ecg_data_dict:
                lead_III = lead_II - lead_I
            else:
                lead_III = ecg_data_dict['III']
            
            # Ensure same length
            min_len = min(len(lead_I), len(lead_II), len(lead_III))
            lead_I = lead_I[:min_len]
            lead_II = lead_II[:min_len]
            lead_III = lead_III[:min_len]
            
            # 计算向量心电图的三个平面
            # 前额面 (frontal plane): I vs aVF
            if 'aVF' in ecg_data_dict:
                lead_aVF = ecg_data_dict['aVF'][:min_len]
            else:
                # 计算aVF = (II + III) / 2
                lead_aVF = (lead_II + lead_III) / 2
                
            # 水平面 (horizontal plane): I vs V2
            if 'V2' in ecg_data_dict:
                lead_V2 = ecg_data_dict['V2'][:min_len]
                
                # 计算水平面面积
                try:
                    # 计算轨迹面积
                    horizontal_area = np.abs(np.trapz(lead_V2, lead_I))
                    features['horizontal_plane_area'] = horizontal_area
                except Exception:
                    features['horizontal_plane_area'] = np.nan
                    
                # 计算水平面最大向量
                try:
                    horizontal_magnitude = np.sqrt(lead_I**2 + lead_V2**2)
                    features['horizontal_max_magnitude'] = np.max(horizontal_magnitude)
                    features['horizontal_mean_magnitude'] = np.mean(horizontal_magnitude)
                except Exception:
                    features['horizontal_max_magnitude'] = np.nan
                    features['horizontal_mean_magnitude'] = np.nan
            
            # 计算前额面特征
            try:
                # 计算轨迹面积
                frontal_area = np.abs(np.trapz(lead_aVF, lead_I))
                features['frontal_plane_area'] = frontal_area
                
                # 计算最大向量
                frontal_magnitude = np.sqrt(lead_I**2 + lead_aVF**2)
                features['frontal_max_magnitude'] = np.max(frontal_magnitude)
                features['frontal_mean_magnitude'] = np.mean(frontal_magnitude)
            except Exception:
                features['frontal_plane_area'] = np.nan
                features['frontal_max_magnitude'] = np.nan
                features['frontal_mean_magnitude'] = np.nan
                
            # 计算空间向量特征
            if 'V2' in ecg_data_dict:
                try:
                    lead_V2 = ecg_data_dict['V2'][:min_len]
                    spatial_magnitude = np.sqrt(lead_I**2 + lead_aVF**2 + lead_V2**2)
                    features['spatial_max_magnitude'] = np.max(spatial_magnitude)
                    features['spatial_mean_magnitude'] = np.mean(spatial_magnitude)
                except Exception:
                    features['spatial_max_magnitude'] = np.nan
                    features['spatial_mean_magnitude'] = np.nan
                    
        except Exception as e:
            print(f"Vectorcardiography feature computation failed: {str(e)}")
        
        # 添加心脏电轴特征
        try:
            axis_features = self.calculate_cardiac_axis(ecg_data_dict)
            features.update(axis_features)
        except Exception as e:
            print(f"Cardiac axis feature addition failed: {str(e)}")
                
        return features
    
    def calculate_cardiac_axis(self, ecg_data_dict):
        """计算心脏电轴
        
        Args:
            ecg_data_dict: 包含多个导联数据的字典
            
        Returns:
            心脏电轴特征字典
        """
        features = {}
        
        try:
            # Check if necessary leads exist
            required_leads = ['I', 'II']
            if not all(lead in ecg_data_dict for lead in required_leads):
                return features
                
            # 获取导联数据
            lead_I = ecg_data_dict['I']
            lead_II = ecg_data_dict['II']
            
            # 计算导联III（如果不存在）
            if 'III' not in ecg_data_dict:
                lead_III = lead_II - lead_I
            else:
                lead_III = ecg_data_dict['III']
            
            # 计算aVF（如果不存在）
            if 'aVF' not in ecg_data_dict:
                lead_aVF = (lead_II + lead_III) / 2
            else:
                lead_aVF = ecg_data_dict['aVF']
                
            # 计算aVL（如果不存在）
            if 'aVL' not in ecg_data_dict:
                lead_aVL = (lead_I - lead_III) / 2
            else:
                lead_aVL = ecg_data_dict['aVL']
            
            # 计算QRS波的平均振幅
            try:
                # 检测R峰
                from ecg_modules.preprocessing import SignalPreprocessor
                preprocessor = SignalPreprocessor(self.sampling_rate)
                
                r_peaks_I = preprocessor.detect_r_peaks(lead_I)
                r_peaks_aVF = preprocessor.detect_r_peaks(lead_aVF)
                
                # 确保R峰是数值类型
                if r_peaks_I is not None:
                    r_peaks_I = np.array(r_peaks_I, dtype=np.int32)
                else:
                    r_peaks_I = np.array([], dtype=np.int32)
                    
                if r_peaks_aVF is not None:
                    r_peaks_aVF = np.array(r_peaks_aVF, dtype=np.int32)
                else:
                    r_peaks_aVF = np.array([], dtype=np.int32)
                
                # If R peaks detected, calculate QRS amplitude
                if len(r_peaks_I) > 0 and len(r_peaks_aVF) > 0:
                    # 取R峰周围的QRS波段
                    qrs_window = int(0.05 * self.sampling_rate)  # 50ms窗口
                    
                    # 计算导联I的QRS振幅
                    I_amplitudes = []
                    for r_peak in r_peaks_I:
                        r_peak_int = int(r_peak)  # 确保是整数
                        if r_peak_int >= qrs_window and r_peak_int + qrs_window < len(lead_I):
                            qrs_segment = lead_I[r_peak_int-qrs_window:r_peak_int+qrs_window]
                            I_amplitude = np.max(qrs_segment) - np.min(qrs_segment)
                            I_amplitudes.append(I_amplitude)
                    
                    # 计算导联aVF的QRS振幅
                    aVF_amplitudes = []
                    for r_peak in r_peaks_aVF:
                        r_peak_int = int(r_peak)  # 确保是整数
                        if r_peak_int >= qrs_window and r_peak_int + qrs_window < len(lead_aVF):
                            qrs_segment = lead_aVF[r_peak_int-qrs_window:r_peak_int+qrs_window]
                            aVF_amplitude = np.max(qrs_segment) - np.min(qrs_segment)
                            aVF_amplitudes.append(aVF_amplitude)
                    
                    # 计算平均振幅
                    if I_amplitudes and aVF_amplitudes:
                        mean_I_amplitude = np.mean(I_amplitudes)
                        mean_aVF_amplitude = np.mean(aVF_amplitudes)
                        
                        # 计算心脏电轴角度（度）
                        axis_angle = np.degrees(np.arctan2(mean_aVF_amplitude, mean_I_amplitude))
                        features['cardiac_axis_angle'] = axis_angle
                        
                        # 判断电轴类型
                        if -30 <= axis_angle <= 90:
                            features['cardiac_axis_type'] = 'normal'
                        elif -90 <= axis_angle < -30:
                            features['cardiac_axis_type'] = 'left_deviation'
                        elif 90 < axis_angle <= 180 or -180 <= axis_angle < -90:
                            features['cardiac_axis_type'] = 'right_deviation'
                        else:
                            features['cardiac_axis_type'] = 'indeterminate'
            except Exception as e:
                print(f"Cardiac axis calculation failed: {str(e)}")
                features['cardiac_axis_angle'] = np.nan
                features['cardiac_axis_type'] = 'unknown'
                
        except Exception as e:
            print(f"Cardiac axis feature calculation failed: {str(e)}")
        
        return features
        
    def extract_features_from_leads(self, data, leads=None):
        """从多导联数据中提取特征
        
        Args:
            data: 包含多导联ECG数据（DataFrame）
            leads: List of leads to process, defaults to all leads
            
        Returns:
            包含所有导联特征的字典
        """
        # Optimize batch processing logic - group similar leads for batch processing
        
        if leads is None:
            leads = list(data.keys())
        
        all_features = {}
        r_peaks_dict = {}
        
        # Parallel preprocessing of R peak detection for all leads
        def process_lead(lead):
            if lead not in data or data[lead] is None:
                print(f"Warning: Lead {lead} not found or data is empty")
                return lead, None, {}
            
            signal_data = data[lead]
            
            # 检查数据质量
            if len(signal_data) < 500:  # 至少需要1秒的数据
                print(f"Warning: Lead {lead} has insufficient data points ({len(signal_data)})")
                return lead, None, {}
            
            # 检测R峰
            try:
                r_peaks, _ = nk.ecg_peaks(signal_data, sampling_rate=self.sampling_rate)
                r_peaks = r_peaks['ECG_R_Peaks']
                
                if len(r_peaks) >= 3:  # 至少需要3个R峰
                    print(f"Successfully detected {len(r_peaks)} R peaks")
                    
                    # 提取特征
                    features = self.extract_features(signal_data, r_peaks)
                    return lead, r_peaks, features
                else:
                    print(f"Warning: Lead {lead} R peak detection failed or insufficient count")
                    return lead, None, {}
            except Exception as e:
                print(f"Error processing lead {lead}: {str(e)}")
                return lead, None, {}
        max_workers = min(len(leads), os.cpu_count() or 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_lead, leads))
        for lead, r_peaks, features in results:
            if r_peaks is not None and len(features) > 0:
                r_peaks_dict[lead] = r_peaks
                all_features.update({f"{lead}_{k}": v for k, v in features.items()})
        
        return all_features, r_peaks_dict

    def extract_dfa_features(self, signal, r_peaks=None):
        try:
            import neurokit2 as nk
            if r_peaks is None:
                r_peaks = nk.ecg_peaks(signal, sampling_rate=self.sampling_rate)[0]['ECG_R_Peaks']
            
            if len(r_peaks) < 10:
                print(f"Warning: Insufficient R peaks ({len(r_peaks)} < 10), cannot compute reliable DFA features")
                return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}
            rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
            if len(rr_intervals) < 50:
                try:
                    dfa_result = nk.fractal_dfa(rr_intervals, scale='default', overlap=True)
                    alpha1 = np.nan
                    if isinstance(dfa_result, tuple) and len(dfa_result) > 1:
                        if isinstance(dfa_result[1], dict) and 'DFA_alpha1' in dfa_result[1]:
                            alpha1 = dfa_result[1]['DFA_alpha1']
                    
                    return {'dfa_alpha1': alpha1, 'dfa_alpha2': np.nan}
                except Exception as e:
                    print(f"Short sequence DFA computation failed: {str(e)}")
                    return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}
            else:
                try:
                    # Use default parameters
                    dfa_result = nk.fractal_dfa(rr_intervals)
                    alpha1 = np.nan
                    alpha2 = np.nan
                    
                    # Safely extract results
                    if isinstance(dfa_result, tuple) and len(dfa_result) > 1:
                        if isinstance(dfa_result[1], dict):
                            alpha1 = dfa_result[1].get('DFA_alpha1', np.nan)
                            alpha2 = dfa_result[1].get('DFA_alpha2', np.nan)
                    
                    return {'dfa_alpha1': alpha1, 'dfa_alpha2': alpha2}
                except Exception as e:
                    print(f"DFA computation failed: {str(e)}")
                    return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}
        except Exception as e:
            print(f"Error extracting DFA features: {str(e)}")
            return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}

    def extract_basic_features(self, signal):
        """
        Extract basic signal features, suitable for signals where R peak detection fails
        
        Args:
            signal: ECG signal data
            
        Returns:
            Basic feature dictionary
        """
        try:
            features = {}
            
            # Time-domain statistical features
            features['mean'] = np.mean(signal)
            features['std'] = np.std(signal)
            features['min'] = np.min(signal)
            features['max'] = np.max(signal)
            features['range'] = np.max(signal) - np.min(signal)
            features['median'] = np.median(signal)
            features['mad'] = np.median(np.abs(signal - np.median(signal)))
            features['skew'] = scipy.stats.skew(signal)
            features['kurtosis'] = scipy.stats.kurtosis(signal)
            
            # Simple frequency domain features
            try:
                from scipy import signal as sp_signal
                f, psd = sp_signal.welch(signal, fs=self.sampling_rate)
                
                # Total power
                features['psd_total'] = np.sum(psd)
                
                # Power in different frequency bands
                if len(f) > 0:
                    # VLF: 0-0.04Hz
                    vlf_idx = np.where(f <= 0.04)[0]
                    if len(vlf_idx) > 0:
                        features['psd_vlf'] = np.sum(psd[vlf_idx])
                    
                    # LF: 0.04-0.15Hz
                    lf_idx = np.where((f > 0.04) & (f <= 0.15))[0]
                    if len(lf_idx) > 0:
                        features['psd_lf'] = np.sum(psd[lf_idx])
                    
                    # HF: 0.15-0.4Hz
                    hf_idx = np.where((f > 0.15) & (f <= 0.4))[0]
                    if len(hf_idx) > 0:
                        features['psd_hf'] = np.sum(psd[hf_idx])
            except Exception as e:
                print(f"Frequency domain feature computation failed: {str(e)}")
            
            # Simple nonlinear features
            try:
                # Zero crossing rate
                zero_crossings = np.sum(np.diff(np.signbit(signal - np.mean(signal))))
                features['zero_crossings'] = zero_crossings
                
                # Signal energy
                features['energy'] = np.sum(signal**2)
                
                # Signal entropy
                try:
                    import antropy as ant
                    features['sample_entropy'] = ant.sample_entropy(signal)
                    features['app_entropy'] = ant.app_entropy(signal)
                    features['perm_entropy'] = ant.perm_entropy(signal)
                except Exception as e:
                    print(f"Entropy feature computation failed: {str(e)}")
            except Exception as e:
                print(f"Nonlinear feature computation failed: {str(e)}")
            
            return features
        except Exception as e:
            print(f"Error extracting basic features: {str(e)}")
            return {}

def print_gpu_info():
    """Print GPU information and status"""
    if GPU_MANAGER_AVAILABLE:
        print_gpu_status()
    else:
        print("\n===== GPU Status =====")
        print(f"GPU Available: {GPU_AVAILABLE}")
        if CUPY_MODULE is not None:
            try:
                device = CUPY_MODULE.cuda.Device(0)
                props = CUPY_MODULE.cuda.runtime.getDeviceProperties(0)
                name = props['name'].decode('utf-8')
                mem_free, mem_total = CUPY_MODULE.cuda.runtime.memGetInfo()
                print(f"GPU Device: {name}")
                print(f"GPU Memory: {mem_free/1024**2:.1f}MB free / {mem_total/1024**2:.1f}MB total")
            except:
                print("Unable to get detailed GPU information")
        print("===================\n")
