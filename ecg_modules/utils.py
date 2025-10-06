import os
import numpy as np
import pandas as pd
import importlib
import joblib
import psutil
import gc
import glob
import warnings
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'antropy',
        'pywt', 'neurokit2', 'sklearn', 'tqdm', 'joblib', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Warning: Missing the following dependencies:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease use the following command to install required dependencies:")
        if missing_packages:  # Ensure it's not an empty list
            print(f"pip install {' '.join(missing_packages)}")
        return False
        
    return True

def memory_usage_monitor():
    """Monitor memory usage, return current memory usage percentage"""
    return psutil.virtual_memory().percent / 100.0

def check_memory_usage(threshold=0.9):
    """Check memory usage, perform garbage collection if threshold exceeded
    
    Args:
        threshold: Memory usage threshold (0.0-1.0)
    
    Returns:
        True if memory usage is below threshold, otherwise False
    """
    memory_usage = memory_usage_monitor()
    
    if memory_usage > threshold:
        print(f"Warning: High memory usage ({memory_usage*100:.1f}%), performing garbage collection...")
        gc.collect()
        return False
    
    return True

def load_cached_data(cache_file, message=None):
    """Load data from cache file
    
    Args:
        cache_file: Cache file path
        message: Optional message string
    
    Returns:
        Cached data, or None (if cache doesn't exist)
    """
    if os.path.exists(cache_file):
        if message:
            print(message)
        try:
            return joblib.load(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {str(e)}")
            return None
    return None

def save_cached_data(data, cache_file, message=None):
    """Save data to cache file
    
    Args:
        data: Data to cache
        cache_file: Cache file path
        message: Optional message string
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    try:
        joblib.dump(data, cache_file)
        if message:
            print(message)
    except Exception as e:
        print(f"Failed to save cache: {str(e)}")

# Define external batch processing function (solve pickle issue)
def _process_batch_helper(args):
    """Batch processing helper function for processing batches in process pool
    
    Args:
        args: Tuple (batch, start_index, process_function)
    
    Returns:
        List of processing results
    """
    batch, start_idx, process_func = args
    batch_results = []
    for idx, item in enumerate(batch):
        try:
            result = process_func(item)
            if result is not None:
                # Include original index information
                batch_results.append((start_idx + idx, result))
        except Exception as e:
            print(f"Error processing item {start_idx + idx}: {str(e)}")
            traceback.print_exc()
    return batch_results

# Define external item processing function (solve pickle issue)
def _process_item_helper(args):
    """Item processing helper function for processing single item in process pool
    
    Args:
        args: Tuple (item, index, process_function)
    
    Returns:
        Processing result tuple (index, result)
    """
    item, idx, process_func = args
    try:
        result = process_func(item)
        if result is not None:
            return (idx, result)
        return None
    except Exception as e:
        print(f"Error processing item {idx}: {str(e)}")
        traceback.print_exc()
        return None

def parallel_process(items, process_func, max_workers=None, desc=None, use_threads=False, batch_size=None, memory_limit=0.8):
    """Parallel processing of item list, supports process pool and thread pool, and batch processing
    
    Args:
        items: List of items to process
        process_func: Processing function
        max_workers: Maximum number of worker processes/threads
        desc: Progress bar description
        use_threads: Whether to use thread pool instead of process pool (beneficial for IO-intensive tasks)
        batch_size: Batch size, None means no batch processing
        memory_limit: Memory usage limit ratio (0.0-1.0), trigger GC when exceeded
    
    Returns:
        List of processing results
    """
    from tqdm import tqdm
    
    # Check batch size setting in environment variables
    if batch_size is None and 'ECG_BATCH_SIZE' in os.environ:
        try:
            batch_size = int(os.environ['ECG_BATCH_SIZE'])
            print(f"Using batch size from environment variable: {batch_size}")
        except (ValueError, TypeError):
            pass
    
    # If not specified, determine max workers based on CPU count
    if max_workers is None:
        cpu_count = os.cpu_count()
        # Prevent cpu_count from returning None
        if cpu_count is None:
            max_workers = 4
        else:
            # Adjust worker count based on task type and system
            if use_threads:
                # Threads can use more since they're lightweight
                max_workers = min(32, cpu_count * 2)
            else:
                # Processes should be more conservative
                max_workers = min(16, cpu_count)
    
    # Further adjust worker count based on available memory
    mem_usage = psutil.virtual_memory().percent / 100.0
    if mem_usage > 0.7:  # If memory usage exceeds 70%
        adjusted_workers = max(1, int(max_workers * (1 - (mem_usage - 0.7) * 2)))
        if adjusted_workers < max_workers:
            print(f"High memory usage ({mem_usage:.1%}), adjusting worker count from {max_workers} to {adjusted_workers}")
            max_workers = adjusted_workers
    
    # Batch processing logic
    if batch_size and batch_size > 0:
        # Split items into batches
        batched_items = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            # Add process_func to each task tuple
            batched_items.append((batch, i, process_func))
        
        # Use external batch processing function
        items_to_process = batched_items
        process_function = _process_batch_helper
    else:
        # Don't use batch processing, add index and process_func to each item
        items_to_process = [(item, idx, process_func) for idx, item in enumerate(items)]
        process_function = _process_item_helper
    
    results = []
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    try:
        with executor_cls(max_workers=max_workers) as executor:
            futures = {executor.submit(process_function, item): i for i, item in enumerate(items_to_process)}
            
            for future in tqdm(as_completed(futures), total=len(items_to_process), desc=desc):
                try:
                    result = future.result()
                    if result is not None:
                        # Batch processing results are a list
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                    
                    # Check memory usage, perform garbage collection if necessary
                    mem_usage = psutil.virtual_memory().percent / 100.0
                    if mem_usage > memory_limit:
                        print(f"Memory usage ({mem_usage:.1%}) exceeds limit ({memory_limit:.1%}), performing garbage collection...")
                        gc.collect()
                        # Pause briefly to let system free memory
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"Error processing result: {str(e)}")
                    traceback.print_exc()
    except Exception as e:
        print(f"Parallel processing error: {str(e)}")
        traceback.print_exc()
    
    # Sort results by original index
    if results:
        # Check results structure, ensure they are (idx, result) pairs
        if isinstance(results[0], tuple) and len(results[0]) == 2:
            sorted_results = [r[1] for r in sorted(results, key=lambda x: x[0])]
            return sorted_results
    
    return results

def find_ecg_files(directory, max_files=None, file_extensions=['.csv', '.txt', '.dat']):
    """
    Find ECG files in directory
    
    Args:
        directory: Directory to search
        max_files: Maximum number of files, None means no limit
        file_extensions: List of supported file extensions
        
    Returns:
        List of qualified file paths
    """
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return []
    
    print(f"Searching for ECG files in directory '{directory}'...")
    
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Check file extension
            if any(file.lower().endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    # Print number of files found
    if all_files:
        print(f"Found {len(all_files)} qualified files")
        if max_files and len(all_files) > max_files:
            print(f"Due to limit, will only process the first {max_files} files")
            all_files = all_files[:max_files]
    else:
        print("No qualified files found")
        # Print all files in directory for debugging
        all_directory_files = []
        for root, _, files in os.walk(directory):
            all_directory_files.extend([os.path.join(root, file) for file in files])
        if all_directory_files:
            print(f"Files in directory (first 10): {all_directory_files[:10]}")
            print(f"Supported file extensions: {file_extensions}")
    
    return all_files

def save_results(df, output_file, overwrite=False):
    """Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        output_file: Output file path
        overwrite: Whether to overwrite existing file
    """
    # Check if file exists
    if os.path.exists(output_file) and not overwrite:
        print(f"File exists, not overwriting: {output_file}")
        return False
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Save DataFrame
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False

def estimate_processing_time(file_count, avg_time_per_file=2.0):
    """Estimate processing time
    
    Args:
        file_count: Number of files
        avg_time_per_file: Average processing time per file (seconds)
    
    Returns:
        Estimated time (seconds) and formatted time string
    """
    total_seconds = file_count * avg_time_per_file
    
    # Convert seconds to more readable format
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    time_str = ""
    if hours > 0:
        time_str += f"{hours} hours "
    if minutes > 0 or hours > 0:
        time_str += f"{minutes} minutes "
    time_str += f"{seconds} seconds"
    
    return total_seconds, time_str

def load_and_merge_results(file_pattern):
    """Load and merge multiple result files
    
    Args:
        file_pattern: File pattern to match
    
    Returns:
        Merged DataFrame
    """
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No matching files found: {file_pattern}")
        return None
    
    print(f"Found {len(files)} files")
    
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
    
    if not dfs:
        return None
    
    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df

def handle_outliers(df, columns=None, method='iqr', threshold=3.0):
    """Handle outliers in DataFrame
    
    Args:
        df: Input DataFrame
        columns: List of column names to process (None means process all numeric columns)
        method: Processing method ('iqr' or 'zscore')
        threshold: Outlier threshold
    
    Returns:
        Processed DataFrame copy
    """
    if df is None or len(df) == 0:
        return df
    
    # Create copy
    cleaned_df = df.copy()
    
    # If columns not specified, select all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    else:
        # Ensure all columns are in DataFrame
        columns = [col for col in columns if col in df.columns]
    
    # Process each column
    for col in columns:
        if method == 'iqr':
            # IQR method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Replace outliers
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:  # Avoid division by zero
                z_scores = (df[col] - mean) / std
                cleaned_df[col] = df[col].mask(abs(z_scores) > threshold, mean)
    
    return cleaned_df

def detect_signal_quality(ecg_signal, sampling_rate=500):
    """Evaluate ECG signal quality
    
    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling rate
    
    Returns:
        Quality score (0-1 range) and quality metrics dictionary
    """
    quality_metrics = {}
    quality_score = 0.4  # Set higher baseline score, increase tolerance for poor signals
    
    try:
        # Calculate signal-to-noise ratio (estimated)
        if np.std(ecg_signal) > 0:
            # Estimate noise using wavelet denoising
            import pywt
            coeffs = pywt.wavedec(ecg_signal, 'db4', level=4)
            # Estimate standard deviation of highest level detail coefficients as noise estimate
            noise_std = np.std(coeffs[-1])
            signal_std = np.std(ecg_signal)
            snr = 20 * np.log10(signal_std / noise_std) if noise_std > 0 else 100
            quality_metrics['estimated_snr'] = snr
        else:
            quality_metrics['estimated_snr'] = 0
        
        # Calculate baseline drift
        if len(ecg_signal) > sampling_rate:
            # Get baseline via lowpass filtering
            from scipy import signal
            nyquist_freq = 0.5 * sampling_rate
            cutoff_freq = 0.5 / nyquist_freq
            
            try:
                # Create butterworth filter
                filter_result = signal.butter(3, cutoff_freq, 'low')
                
                # Ensure correct extraction of filter coefficients
                if isinstance(filter_result, tuple) and len(filter_result) >= 2:
                    b_coef = filter_result[0]  # Numerator coefficients
                    a_coef = filter_result[1]  # Denominator coefficients
                else:
                    # Single return value case (unlikely, but just in case)
                    print("Warning: Lowpass filter returned single value, using default denominator coefficient")
                    b_coef = filter_result if not isinstance(filter_result, tuple) else filter_result[0]
                    a_coef = 1.0
                
                # Apply filter
                baseline = signal.filtfilt(b_coef, a_coef, ecg_signal)
                
                # Calculate baseline drift amplitude
                baseline_drift = np.max(baseline) - np.min(baseline)
                signal_range = np.max(ecg_signal) - np.min(ecg_signal)
                
                if signal_range > 0:
                    baseline_drift_ratio = baseline_drift / signal_range
                    quality_metrics['baseline_drift_ratio'] = baseline_drift_ratio
                else:
                    quality_metrics['baseline_drift_ratio'] = 1.0
            except Exception as e:
                print(f"Baseline drift calculation error: {str(e)}")
                quality_metrics['baseline_drift_ratio'] = 1.0
        else:
            quality_metrics['baseline_drift_ratio'] = 1.0
            
        # Detect R peaks
        import neurokit2 as nk
        try:
            # Try multiple R peak detection methods
            methods = ['neurokit', 'pantompkins', 'hamilton', 'engzeemod']
            best_r_peaks = []
            
            for method in methods:
                try:
                    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method=method)
                    r_peaks = info['ECG_R_Peaks']
                    if len(r_peaks) > len(best_r_peaks):
                        best_r_peaks = r_peaks
                except:
                    continue
            
            # If all methods fail, use basic threshold detection
            if len(best_r_peaks) <= 3:
                from scipy import signal
                try:
                    # Preprocessing: bandpass filtering
                    nyquist = 0.5 * sampling_rate
                    low, high = 5 / nyquist, 15 / nyquist
                    
                    # Safe handling of filter coefficients
                    filter_result = signal.butter(3, [low, high], 'band')
                    
                    # Ensure correct extraction of filter coefficients
                    if isinstance(filter_result, tuple) and len(filter_result) >= 2:
                        b = filter_result[0]  # Numerator coefficients
                        a = filter_result[1]  # Denominator coefficients
                    else:
                        # Single return value case (unlikely, but just in case)
                        print("Warning: butter filter returned single value, using default denominator coefficient")
                        b = filter_result if not isinstance(filter_result, tuple) else filter_result[0]
                        a = 1.0
                    
                    # Apply filtering
                    filtered = signal.filtfilt(b, a, ecg_signal)
                    
                    # Square and absolute value to enhance R wave
                    squared = filtered ** 2
                    
                    # Find local maxima
                    max_indices = signal.find_peaks(squared, distance=int(sampling_rate * 0.5))[0]
                    if len(max_indices) > 3:
                        best_r_peaks = max_indices
                except Exception as e:
                    print(f"Basic R peak detection failed: {str(e)}")
            
            r_peaks = best_r_peaks
            
            # Calculate R peak interval variability
            if len(r_peaks) > 3:
                rr_intervals = np.diff(r_peaks)
                rr_cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 999
                quality_metrics['rr_cv'] = rr_cv
                
                # Extreme RR interval ratio
                extreme_rr = np.sum((rr_intervals < 0.4 * np.mean(rr_intervals)) | 
                                   (rr_intervals > 1.7 * np.mean(rr_intervals)))
                extreme_rr_ratio = extreme_rr / len(rr_intervals) if len(rr_intervals) > 0 else 1.0
                quality_metrics['extreme_rr_ratio'] = extreme_rr_ratio
                
            # R peak count check
            expected_beats = len(ecg_signal) / sampling_rate / 60 * 75  # Assume 75 BPM
            if expected_beats > 0:
                r_peaks_ratio = len(r_peaks) / expected_beats
                quality_metrics['r_peaks_ratio'] = min(r_peaks_ratio, 1/r_peaks_ratio) if r_peaks_ratio > 0 else 0
            else:
                quality_metrics['r_peaks_ratio'] = 0
                
            # If not enough R peaks
            if len(r_peaks) <= 3:
                quality_metrics['rr_cv'] = 999
                quality_metrics['extreme_rr_ratio'] = 1.0
                quality_metrics['r_peaks_ratio'] = 0
                
        except Exception as e:
            quality_metrics['rr_cv'] = 999
            quality_metrics['extreme_rr_ratio'] = 1.0
            quality_metrics['r_peaks_ratio'] = 0
            print(f"R peak detection error: {str(e)}")
            
        # Check signal dropout (flat line)
        flat_segments = []
        threshold = 0.005 * (np.max(ecg_signal) - np.min(ecg_signal))  # Lower flat line detection threshold
        is_flat = False
        flat_start = 0
        
        for i in range(1, len(ecg_signal)):
            if abs(ecg_signal[i] - ecg_signal[i-1]) < threshold:
                if not is_flat:
                    is_flat = True
                    flat_start = i-1
            else:
                if is_flat:
                    flat_length = i - flat_start
                    if flat_length > sampling_rate / 4:  # Extend flat line threshold
                        flat_segments.append((flat_start, i))
                    is_flat = False
                    
        # Calculate flat line ratio
        total_flat_points = sum(end - start for start, end in flat_segments)
        flat_ratio = total_flat_points / len(ecg_signal) if len(ecg_signal) > 0 else 1.0
        quality_metrics['flat_signal_ratio'] = flat_ratio
        
        # Comprehensive quality score - Lower weights, allow more signals to pass
        # Based on SNR (0-40dB mapped to 0-0.15)
        snr_score = min(0.15, max(0, quality_metrics['estimated_snr'] / 250))
        
        # Based on baseline drift (0-1 mapped to 0-0.1, inverse)
        drift_score = 0.1 * (1 - min(1.0, quality_metrics['baseline_drift_ratio'] * 2.5))
        
        # Based on RR interval variability (0-1 mapped to 0-0.1, inverse)
        if 'rr_cv' in quality_metrics and quality_metrics['rr_cv'] != 999:
            rr_cv_score = 0.1 * (1 - min(1.0, quality_metrics['rr_cv'] * 2.5))
        else:
            rr_cv_score = 0
            
        # Based on R peak count (0-1 mapped to 0-0.25)
        r_peaks_score = 0.25 * min(1.0, quality_metrics['r_peaks_ratio'] * 1.2)
        
        # Calculate total score - Increase baseline score, allow more signals to pass
        quality_score += 0.6 * (snr_score + drift_score + rr_cv_score + r_peaks_score)
        
        # If flat line ratio is too high, lower score but don't completely exclude
        if quality_metrics['flat_signal_ratio'] > 0.4:  # Increase flat line tolerance
            quality_score *= (1 - quality_metrics['flat_signal_ratio'] * 0.25)  # Lower penalty weight
            
    except Exception as e:
        print(f"Signal quality assessment error: {str(e)}")
        quality_score = 0.4  # Give a low but non-zero score, allow use when no better option
        quality_metrics['error'] = str(e)
    
    return quality_score, quality_metrics
