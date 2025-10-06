#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECG Anesthesia Delirium Prediction API Wrapper

Provides unified API interface, encapsulates existing prediction algorithms for Streamlit application
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import tempfile
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

warnings.filterwarnings("ignore")

# Add project path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from ecg_modules.preprocessing import SignalPreprocessor
    from ecg_modules.feature_extraction import FeatureExtractor
    from ecg_modules.utils import detect_signal_quality
    from predict_delirium import DeliriumPredictor
except ImportError as e:
    print(f"Warning: Module import failed - {str(e)}")

class ECGDeliriumAPI:
    """ECG Anesthesia Delirium Prediction API Class"""
    
    def __init__(self, model_path: Optional[str] = None, sampling_rate: int = 1000):
        """
        Initialize API
        
        Args:
            model_path: Model file path, auto-search if None
            sampling_rate: Signal sampling rate
        """
        self.sampling_rate = sampling_rate
        self.model_path = model_path
        
        # Initialize components
        self.preprocessor = None
        self.feature_extractor = None
        self.predictor = None
        
        # Model information
        self.model_info = {}
        self.is_initialized = False
        
    def initialize(self) -> Tuple[bool, str]:
        """
        Initialize all components
        
        Returns:
            (success flag, error message)
        """
        try:
            # Initialize preprocessor
            self.preprocessor = SignalPreprocessor(sampling_rate=self.sampling_rate)
            
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor()
            
            # Find and load model
            model_info = self._find_best_model()
            if not model_info:
                return False, "No available model file found"
            
            # Initialize predictor
            self.predictor = DeliriumPredictor(
                model_path=model_info.get('path'),
                meta_path=model_info.get('metadata')
            )
            
            # Load model
            if not self.predictor.load_model():
                return False, "Model loading failed"
            
            self.model_info = model_info
            self.is_initialized = True
            
            return True, "System initialization successful"
            
        except Exception as e:
            return False, f"Initialization failed: {str(e)}"
    
    def _find_best_model(self) -> Optional[Dict]:
        """
        Find best model file
        
        Returns:
            Model information dictionary or None
        """
        # Possible model directories
        model_dirs = [
            'delirium_model_results/models',
            'models',
            'output/models'
        ]
        
        # Model priorities (sorted by performance)
        model_priorities = [
            'RandomForest',
            'XGBoost', 
            'LogisticRegression',
            'SVM',
            'KNN'
        ]
        
        best_model = None
        
        for model_dir in model_dirs:
            if not os.path.exists(model_dir):
                continue
                
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            # Select model by priority
            for priority_model in model_priorities:
                for model_file in model_files:
                    if priority_model in model_file and 'model' in model_file:
                        model_path = os.path.join(model_dir, model_file)
                        
                        # Find metadata file
                        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
                        if not os.path.exists(metadata_path):
                            metadata_path = None
                        
                        best_model = {
                            'path': model_path,
                            'metadata': metadata_path,
                            'type': priority_model,
                            'filename': model_file,
                            'directory': model_dir
                        }
                        
                        return best_model
        
        # If no priority model found, use any available model
        for model_dir in model_dirs:
            if not os.path.exists(model_dir):
                continue
                
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f]
            if model_files:
                model_path = os.path.join(model_dir, model_files[0])
                
                return {
                    'path': model_path,
                    'metadata': None,
                    'type': 'Unknown',
                    'filename': model_files[0],
                    'directory': model_dir
                }
        
        return None
    
    def validate_ecg_data(self, data: Union[pd.DataFrame, np.ndarray, str]) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Validate ECG data format
        
        Args:
            data: ECG data (DataFrame, array, or file path)
            
        Returns:
            (is_valid, message, processed DataFrame)
        """
        try:
            # Handle different input types
            if isinstance(data, str):
                # File path
                if not os.path.exists(data):
                    return False, f"File does not exist: {data}", None
                
                try:
                    df = pd.read_csv(data)
                except Exception as e:
                    return False, f"File reading failed: {str(e)}", None
                    
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
                
            elif isinstance(data, np.ndarray):
                # Assume 12-lead data
                if data.shape[1] != 12:
                    return False, f"Array dimension error, expected 12 columns, got {data.shape[1]}", None
                
                # Create DataFrame
                lead_names = [
                    'MDC_ECG_LEAD_I', 'MDC_ECG_LEAD_II', 'MDC_ECG_LEAD_III',
                    'MDC_ECG_LEAD_aVR', 'MDC_ECG_LEAD_aVL', 'MDC_ECG_LEAD_aVF',
                    'MDC_ECG_LEAD_V1', 'MDC_ECG_LEAD_V2', 'MDC_ECG_LEAD_V3',
                    'MDC_ECG_LEAD_V4', 'MDC_ECG_LEAD_V5', 'MDC_ECG_LEAD_V6'
                ]
                df = pd.DataFrame(data, columns=lead_names)
                
            else:
                return False, f"Unsupported data type: {type(data)}", None
            
            # Check required leads
            required_leads = [
                'MDC_ECG_LEAD_I', 'MDC_ECG_LEAD_II', 'MDC_ECG_LEAD_III',
                'MDC_ECG_LEAD_aVR', 'MDC_ECG_LEAD_aVL', 'MDC_ECG_LEAD_aVF',
                'MDC_ECG_LEAD_V1', 'MDC_ECG_LEAD_V2', 'MDC_ECG_LEAD_V3',
                'MDC_ECG_LEAD_V4', 'MDC_ECG_LEAD_V5', 'MDC_ECG_LEAD_V6'
            ]
            
            missing_leads = [lead for lead in required_leads if lead not in df.columns]
            if missing_leads:
                return False, f"Missing required leads: {', '.join(missing_leads)}", None
            
            # Check data length
            if len(df) < 100:
                return False, f"Data too short: {len(df)} < 100", None
            
            # Check and convert data types
            for lead in required_leads:
                if not pd.api.types.is_numeric_dtype(df[lead]):
                    try:
                        df[lead] = pd.to_numeric(df[lead], errors='coerce')
                    except:
                        return False, f"Lead {lead} data type conversion failed", None
            
            # Check missing values
            missing_ratio = df[required_leads].isnull().sum().sum() / (len(df) * len(required_leads))
            if missing_ratio > 0.1:  # More than 10% data missing
                return False, f"Missing data ratio too high: {missing_ratio:.1%}", None
            
            # Fill small amount of missing values
            df[required_leads] = df[required_leads].fillna(method='forward').fillna(method='backward')
            
            return True, "Data validation passed", df
            
        except Exception as e:
            return False, f"Data validation failed: {str(e)}", None
    
    def extract_features(self, df: pd.DataFrame, filename: str = "unknown") -> Optional[Dict]:
        """
        Extract features from ECG data
        
        Args:
            df: ECG data DataFrame
            filename: File name
            
        Returns:
            Feature dictionary or None
        """
        if not self.is_initialized:
            raise RuntimeError("API not initialized, please call initialize() first")
        
        try:
            leads = [
                'MDC_ECG_LEAD_I', 'MDC_ECG_LEAD_II', 'MDC_ECG_LEAD_III',
                'MDC_ECG_LEAD_aVR', 'MDC_ECG_LEAD_aVL', 'MDC_ECG_LEAD_aVF',
                'MDC_ECG_LEAD_V1', 'MDC_ECG_LEAD_V2', 'MDC_ECG_LEAD_V3',
                'MDC_ECG_LEAD_V4', 'MDC_ECG_LEAD_V5', 'MDC_ECG_LEAD_V6'
            ]
            
            features = {'file': filename}
            
            for lead in leads:
                if lead not in df.columns:
                    continue
                
                signal_data = df[lead].values
                
                # Check signal quality
                try:
                    quality = detect_signal_quality(signal_data)
                    features[f"{lead}_quality"] = quality
                except:
                    features[f"{lead}_quality"] = 1.0
                
                # Preprocess signal
                try:
                    processed_signal = self.preprocessor.filter_signal(signal_data)
                    processed_signal = self.preprocessor.remove_baseline_wander(processed_signal)
                except Exception as e:
                    print(f"Preprocessing warning {lead}: {str(e)}")
                    processed_signal = signal_data
                
                # Extract features
                try:
                    # Time domain features
                    time_features = self.feature_extractor.extract_time_domain_features(processed_signal)
                    for key, value in time_features.items():
                        features[f"{lead}_{key}"] = value
                    
                    # Frequency domain features
                    freq_features = self.feature_extractor.extract_frequency_domain_features(
                        processed_signal, self.sampling_rate
                    )
                    for key, value in freq_features.items():
                        features[f"{lead}_{key}"] = value
                    
                    # Nonlinear features
                    try:
                        nonlinear_features = self.feature_extractor.extract_nonlinear_features(processed_signal)
                        for key, value in nonlinear_features.items():
                            features[f"{lead}_{key}"] = value
                    except:
                        pass  # Nonlinear feature extraction may fail, skip
                    
                except Exception as e:
                    print(f"Feature extraction warning {lead}: {str(e)}")
                    # Basic statistical features as fallback
                    features[f"{lead}_mean"] = np.mean(processed_signal)
                    features[f"{lead}_std"] = np.std(processed_signal)
                    features[f"{lead}_max"] = np.max(processed_signal)
                    features[f"{lead}_min"] = np.min(processed_signal)
                    features[f"{lead}_range"] = np.max(processed_signal) - np.min(processed_signal)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            return None
    
    def predict_single(self, data: Union[pd.DataFrame, np.ndarray, str], filename: str = None) -> Dict:
        """
        Delirium risk prediction for single ECG file
        
        Args:
            data: ECG data
            filename: File name
            
        Returns:
            Prediction result dictionary
        """
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'API not initialized',
                'filename': filename
            }
        
        # If filename is None, try to get from data
        if filename is None:
            if isinstance(data, str):
                filename = os.path.basename(data)
            else:
                filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Validate data
            is_valid, message, df = self.validate_ecg_data(data)
            if not is_valid:
                return {
                    'success': False,
                    'error': message,
                    'filename': filename
                }
            
            # Extract features
            features = self.extract_features(df, filename)
            if not features:
                return {
                    'success': False,
                    'error': 'Feature extraction failed',
                    'filename': filename
                }
            
            # Make prediction
            prediction_result = self._predict_with_features(features)
            
            result = {
                'success': True,
                'filename': filename,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_shape': df.shape,
                'signal_length': len(df),
                'sampling_rate': self.sampling_rate,
                **prediction_result
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'filename': filename
            }
    
    def predict_batch(self, data_list: List[Union[pd.DataFrame, np.ndarray, str]], 
                     filenames: Optional[List[str]] = None) -> List[Dict]:
        """
        Batch delirium risk prediction for multiple ECG files
        
        Args:
            data_list: ECG data list
            filenames: Filename list
            
        Returns:
            Prediction result list
        """
        if not self.is_initialized:
            return [{
                'success': False,
                'error': 'API not initialized',
                'filename': f'file_{i}'
            } for i in range(len(data_list))]
        
        results = []
        
        for i, data in enumerate(data_list):
            # Get filename
            if filenames and i < len(filenames):
                filename = filenames[i]
            elif isinstance(data, str):
                filename = os.path.basename(data)
            else:
                filename = f"batch_file_{i+1}"
            
            # Single prediction
            result = self.predict_single(data, filename)
            results.append(result)
        
        return results
    
    def _predict_with_features(self, features: Dict) -> Dict:
        """
        Make prediction using features
        
        Args:
            features: Feature dictionary
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Prepare feature data
            feature_df = pd.DataFrame([features])
            
            # Remove non-numeric columns
            non_numeric_cols = ['file']
            numeric_features = {k: v for k, v in features.items() 
                              if k not in non_numeric_cols and isinstance(v, (int, float, np.number))}
            
            feature_df = pd.DataFrame([numeric_features])
            
            # Handle missing and infinite values
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.fillna(0)
            
            # Align features if model has specific feature list
            if hasattr(self.predictor, 'feature_list') and self.predictor.feature_list:
                # Keep only features needed by model
                available_features = [f for f in self.predictor.feature_list if f in feature_df.columns]
                
                if available_features:
                    feature_df = feature_df[available_features]
                
                # Add missing features
                missing_features = [f for f in self.predictor.feature_list if f not in feature_df.columns]
                for feature in missing_features:
                    feature_df[feature] = 0
                
                # Reorder features
                feature_df = feature_df.reindex(columns=self.predictor.feature_list, fill_value=0)
            
            # Make prediction
            if hasattr(self.predictor.model, 'predict_proba'):
                probabilities = self.predictor.model.predict_proba(feature_df)
                risk_score = float(probabilities[0][1])  # Positive class probability
            else:
                prediction = self.predictor.model.predict(feature_df)
                risk_score = float(prediction[0])
            
            # Ensure risk score is within [0,1] range
            risk_score = max(0.0, min(1.0, risk_score))
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "High Risk"
                risk_color = "red"
            elif risk_score >= 0.4:
                risk_level = "Medium Risk"
                risk_color = "orange"
            else:
                risk_level = "Low Risk"
                risk_color = "green"
            
            # Binary prediction
            prediction_binary = risk_score >= 0.5
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'prediction': prediction_binary,
                'prediction_label': 'Delirium' if prediction_binary else 'Non-Delirium',
                'feature_count': len(feature_df.columns),
                'model_type': self.model_info.get('type', 'Unknown')
            }
            
        except Exception as e:
            return {
                'risk_score': None,
                'risk_level': None,
                'risk_color': None,
                'prediction': None,
                'prediction_label': None,
                'error': f'Prediction calculation failed: {str(e)}'
            }
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Model information dictionary
        """
        info = {
            'is_initialized': self.is_initialized,
            'sampling_rate': self.sampling_rate,
        }
        
        if self.is_initialized:
            info.update(self.model_info)
            
            if self.predictor:
                info['model_loaded'] = self.predictor.model is not None
                if hasattr(self.predictor, 'model_type'):
                    info['model_type'] = self.predictor.model_type
                if hasattr(self.predictor, 'feature_list'):
                    info['feature_count'] = len(self.predictor.feature_list) if self.predictor.feature_list else 0
        
        return info
    
    def get_supported_formats(self) -> Dict:
        """
        Get supported data format information
        
        Returns:
            Format information dictionary
        """
        return {
            'file_formats': ['csv'],
            'required_columns': [
                'MDC_ECG_LEAD_I', 'MDC_ECG_LEAD_II', 'MDC_ECG_LEAD_III',
                'MDC_ECG_LEAD_aVR', 'MDC_ECG_LEAD_aVL', 'MDC_ECG_LEAD_aVF',
                'MDC_ECG_LEAD_V1', 'MDC_ECG_LEAD_V2', 'MDC_ECG_LEAD_V3',
                'MDC_ECG_LEAD_V4', 'MDC_ECG_LEAD_V5', 'MDC_ECG_LEAD_V6'
            ],
            'optional_columns': ['Time', 'Sampling_Rate', 'Sampling_Interval', 'Sampling_Unit'],
            'min_length': 100,
            'supported_sampling_rates': [500, 1000, 2000],
            'max_file_size': '200MB'
        }


# Create global API instance
global_api = ECGDeliriumAPI()

def get_api() -> ECGDeliriumAPI:
    """Get global API instance"""
    return global_api
