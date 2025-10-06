#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
try:
    from ecg_modules.preprocessing import SignalPreprocessor
    from ecg_modules.feature_extraction import FeatureExtractor
    from ecg_modules.visualization import Visualizer
except ImportError as e:
    print(f"Module import failed: {str(e)}")
class DeliriumPredictor:
    def __init__(self, model_path=None, meta_path=None):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.scaler = None
        self.feature_list = None
        self.model_type = None
        if not self.model_path:
            self._find_model()  
    def _find_model(self):
        possible_dirs = [
            os.path.join(script_dir, 'models'),
            os.path.join(script_dir, 'output', 'models'),
            os.path.join(script_dir, 'delirium_model_results', 'models')
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                model_files = [f for f in os.listdir(dir_path) if f.endswith('_model.pkl')]
                if model_files:
                    self.model_path = os.path.join(dir_path, model_files[0])
                    print(f"Found model file: {self.model_path}")
                    
                    meta_file = os.path.join(dir_path, 'model_metadata.pkl')
                    if os.path.exists(meta_file):
                        self.meta_path = meta_file
                    
                    scaler_file = os.path.join(dir_path, 'scaler.pkl')
                    if os.path.exists(scaler_file):
                        try:
                            with open(scaler_file, 'rb') as f:
                                self.scaler = pickle.load(f)
                        except:
                            print("Failed to load scaler")
                    
                    return True
        
        print("No available model file found")
        return False
    def load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            print("Model file does not exist")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f) 
            model_name = os.path.basename(self.model_path)
            if 'RandomForest' in model_name:
                self.model_type = 'RandomForest'
            elif 'XGBoost' in model_name:
                self.model_type = 'XGBoost'
            elif 'SVM' in model_name:
                self.model_type = 'SVM'
            elif 'LogisticRegression' in model_name:
                self.model_type = 'LogisticRegression'
            elif 'KNN' in model_name:
                self.model_type = 'KNN'
            elif 'PyTorch' in model_name or 'deep' in model_name.lower():
                self.model_type = 'DeepLearning'
            else:
                self.model_type = 'Unknown'
            
            print(f"Successfully loaded model: {self.model_type}")
            
            if self.meta_path and os.path.exists(self.meta_path):
                try:
                    with open(self.meta_path, 'rb') as f:
                        metadata = pickle.load(f)
                        if isinstance(metadata, dict) and 'features' in metadata:
                            self.feature_list = metadata['features']
                except:
                    print("Failed to load model metadata")
            
            return True
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False
    
    def analyze_file(self, file_path, output_dir=None):
        print(f"\nAnalyzing file: {os.path.basename(file_path)}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if self.model is None:
            if not self.load_model():
                return False
        
        try:
            print("Extracting features...")
            print("Predicting delirium risk...")
            prediction = 
            
            print(f"Delirium risk prediction: {prediction:.2f}")
            if prediction >= 0.5:
                print("Prediction result: High delirium risk")
            else:
                print("Prediction result: Low delirium risk")
            
            if output_dir:
                result = {
                    'file': os.path.basename(file_path),
                    'prediction': prediction,
                    'high_risk': prediction >= 0.5,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                import json
                with open(os.path.join(output_dir, 'prediction_result.json'), 'w') as f:
                    json.dump(result, f, indent=4)
                
                print(f"Results saved to: {output_dir}")
            
            return True
        except Exception as e:
            print(f"Failed to analyze file: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        predictor = DeliriumPredictor()
        predictor.analyze_file(file_path)
    else:
        print("Please provide ECG file path") 