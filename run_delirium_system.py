#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import time
import logging
from datetime import datetime
import psutil
from tqdm import tqdm

script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(script_path)
ecg_module_path = os.path.join(current_dir, 'ecg_modules')
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"Added path to sys.path: {current_dir}")
if ecg_module_path not in sys.path:
    sys.path.append(ecg_module_path)
    print(f"Added path to sys.path: {ecg_module_path}")

CONFIG = {'MAX_FILES': 600, 'USE_GPU': True}

print(f"Current Python path: {sys.path}")

# Attempt to import ecg_modules
try:
    import ecg_modules
    from ecg_modules.preprocessing import SignalPreprocessor
    from ecg_modules.feature_extraction import FeatureExtractor
    from ecg_modules.feature_selection import FeatureSelector
    from ecg_modules.model_building import ModelBuilder, build_full_delirium_model
    from ecg_modules.visualization import Visualizer
    from ecg_modules.utils import find_ecg_files, detect_signal_quality
    # Import newly added modules
    # from lead_importance_analysis import LeadImportanceAnalyzer  
    from multi_model_builder import build_multiple_models
    # from deep_ecg_model import train_deep_learning_model  
    from ecg_delirium_predictor import ECGDeliriumPredictor
    print("Successfully imported ecg_modules")
except ImportError as e:
    print(f"Failed to import ecg_modules: {str(e)}")
    print("Please ensure ecg_modules directory exists and contains all necessary Python module files")
    sys.exit(1)

# Configure logging
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'delirium_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Delirium_System')

try:
    # Import custom modules
    # from feature_select_new import ECGAnalyzer  # Module missing, temporarily commented
    from predict_delirium import DeliriumPredictor
except ImportError as e:
    logger.error(f"Failed to import module: {str(e)}")
    logger.error(f"Current Python path: {sys.path}")
    logger.error(f"Current working directory: {os.getcwd()}")
    # Print current directory contents to help diagnose
    try:
        dir_content = os.listdir(current_dir)
        logger.info(f"Current directory contents: {dir_content}")
        if 'ecg_modules' in dir_content:
            modules_content = os.listdir(ecg_module_path)
            logger.info(f"ecg_modules directory contents: {modules_content}")
    except Exception as dir_err:
        logger.error(f"Failed to list directory contents: {str(dir_err)}")
    sys.exit(1)

class DeliriumSystem:
    """Anesthesia delirium prediction system class"""
    
    def __init__(self, data_dir=None, delirium_dir=None, non_delirium_dir=None, output_dir=None, model_dir=None, max_files=None):
        """Initialize system
        
        Args:
            data_dir: Data directory
            delirium_dir: Delirium group data directory
            non_delirium_dir: Non-delirium group data directory
            output_dir: Output directory
            model_dir: Model directory
            max_files: Maximum number of files to process, None means process all files
        """
        self.data_dir = data_dir or os.path.join(current_dir, 'data')
        self.delirium_dir = delirium_dir
        self.non_delirium_dir = non_delirium_dir
        self.output_dir = output_dir or os.path.join(current_dir, 'output')
        self.model_dir = model_dir or os.path.join(current_dir, 'delirium_model_results')
        self.max_files = max_files  # If None, process all files
        
        # Create directories
        for directory in [self.output_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create subdirectories
        self.feature_dir = os.path.join(self.output_dir, 'features')
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        
        for directory in [self.feature_dir, self.results_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.analyzer = None
        self.predictor = None
        self.model_builder = None
        
        logger.info("Anesthesia delirium prediction system initialized successfully")
        logger.info(f"Data directory: {self.data_dir}")
        if self.delirium_dir:
            logger.info(f"Delirium group data directory: {self.delirium_dir}")
        if self.non_delirium_dir:
            logger.info(f"Non-delirium group data directory: {self.non_delirium_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model directory: {self.model_dir}")
        if self.max_files:
            logger.info(f"Maximum file processing limit: {self.max_files}")
        else:
            logger.info("No file limit, will process all data")
    
    def check_dependencies(self):
        """Check dependencies"""
        try:
            import numpy
            import pandas
            import scipy
            import matplotlib
            import sklearn
            import antropy
            import neurokit2
            import pywt
            
            logger.info("Dependency check passed: all required libraries are installed")
            
            # Check if data directory exists
            if not os.path.exists(self.data_dir):
                logger.warning(f"Data directory does not exist: {self.data_dir}")
                os.makedirs(self.data_dir, exist_ok=True)
                logger.info(f"Created data directory: {self.data_dir}")
            
            return True
        except ImportError as e:
            logger.error(f"Dependency check failed: {str(e)}")
            return False
    
    def find_data_folders(self):
        """Find data subfolders"""
        try:
            # If delirium and non-delirium data directories are already specified, use them directly
            delirium_dir = self.delirium_dir
            non_delirium_dir = self.non_delirium_dir
            
            # If not specified, try to find them in the data directory
            if not delirium_dir:
                delirium_dir = os.path.join(self.data_dir, 'delirium')
                # If not found, try other possible names
                if not os.path.exists(delirium_dir):
                    for name in ['delirium', 'positive', 'pos', 'case', 'PND_long_sequences']:
                        test_dir = os.path.join(self.data_dir, name)
                        if os.path.exists(test_dir):
                            delirium_dir = test_dir
                            break
            
            if not non_delirium_dir:
                non_delirium_dir = os.path.join(self.data_dir, 'non_delirium')
                # If not found, try other possible names
                if not os.path.exists(non_delirium_dir):
                    for name in ['non_delirium', 'negative', 'neg', 'control', 'NPND_long_sequences']:
                        test_dir = os.path.join(self.data_dir, name)
                        if os.path.exists(test_dir):
                            non_delirium_dir = test_dir
                            break
            
            # If still not found, search top-level directory for folders with specific names
            if not os.path.exists(delirium_dir) or not os.path.exists(non_delirium_dir):
                dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
                
                for d in dirs:
                    lower_d = d.lower()
                    if any(keyword in lower_d for keyword in ['delirium', 'positive', 'pos', 'case', 'pnd']) and not os.path.exists(delirium_dir):
                        delirium_dir = os.path.join(self.data_dir, d)
                    elif any(keyword in lower_d for keyword in ['non', 'negative', 'neg', 'control', 'npnd']) and not os.path.exists(non_delirium_dir):
                        non_delirium_dir = os.path.join(self.data_dir, d)
            
            # Check if data folders were found
            if not os.path.exists(delirium_dir):
                logger.warning(f"Delirium group data folder not found")
                delirium_dir = None
            else:
                logger.info(f"Delirium group data folder: {delirium_dir}")
            
            if not os.path.exists(non_delirium_dir):
                logger.warning(f"Non-delirium group data folder not found")
                non_delirium_dir = None
            else:
                logger.info(f"Non-delirium group data folder: {non_delirium_dir}")
            
            # Update class instance variables for use by subsequent methods
            self.delirium_dir = delirium_dir
            self.non_delirium_dir = non_delirium_dir
            
            return delirium_dir, non_delirium_dir
            
        except Exception as e:
            logger.error(f"Failed to find data folders: {str(e)}")
            return None, None
    
    def process_data(self):
        """Process data and extract features"""
        logger.info("Starting data processing...")
        
        # Check if preprocessed feature files already exist
        processed_delirium_file = os.path.join(self.feature_dir, "processed_delirium_features.csv")
        processed_non_delirium_file = os.path.join(self.feature_dir, "processed_non_delirium_features.csv")
        
        if os.path.exists(processed_delirium_file) and os.path.exists(processed_non_delirium_file):
            logger.info("Detected preprocessed feature files, skipping data processing step")
            logger.info(f"Using existing file: {processed_delirium_file}")
            logger.info(f"Using existing file: {processed_non_delirium_file}")
            return True
        
        try:
            # Get data directories
            delirium_dir, non_delirium_dir = self.find_data_folders()
            
            if not delirium_dir or not non_delirium_dir:
                logger.error("Required data directories not found, cannot continue processing")
                return False
            
            # ECGAnalyzer module missing, skip raw data processing
            logger.warning("ECGAnalyzer module not available, cannot process raw data")
            logger.info("Please ensure preprocessed feature files exist, or provide ECGAnalyzer module")
            return False
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def train_model(self):
        """Train delirium prediction model - train 7 machine learning models (RF, XGBoost, GB, ET, KNN, SVM, LR)"""
        logger.info("Starting delirium prediction model training...")
        logger.info("Will train 7 models: RandomForest, XGBoost, GradientBoosting, ExtraTrees, KNN, SVM, LogisticRegression")
        
        # Prioritize using processed feature files (containing 336 features)
        processed_delirium = os.path.join(self.model_dir, 'features', 'processed_delirium_features.csv')
        processed_non_delirium = os.path.join(self.model_dir, 'features', 'processed_non_delirium_features.csv')
        
        if os.path.exists(processed_delirium) and os.path.exists(processed_non_delirium):
            logger.info(f"✓ Using processed feature files:")
            logger.info(f"  POD: {processed_delirium}")
            logger.info(f"  NPOD: {processed_non_delirium}")
            delirium_file = processed_delirium
            non_delirium_file = processed_non_delirium
        else:
            logger.error("Processed feature files not found!")
            logger.error(f"Expected location: {processed_delirium}")
            return False
        
        # Read feature data and start training
        try:
            # Read feature data
            logger.info(f"Reading delirium feature file: {delirium_file}")
            delirium_df = pd.read_csv(delirium_file)
            logger.info(f"Reading non-delirium feature file: {non_delirium_file}")
            non_delirium_df = pd.read_csv(non_delirium_file)
            
            logger.info(f"✓ POD samples: {len(delirium_df)}")
            logger.info(f"✓ NPOD samples: {len(non_delirium_df)}")
            logger.info(f"✓ Number of features: {len(delirium_df.columns) - 2}") # Subtract file and label columns
            
            # Detect GPU
            try:
                import torch
                has_gpu = torch.cuda.is_available()
                if has_gpu:
                    logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}, XGBoost will use GPU acceleration")
                else:
                    logger.info("Using CPU for training")
            except:
                logger.info("Using CPU for training")
            
            # Train 7 models using build_multiple_models
            logger.info("\n" + "=" * 80)
            logger.info("Starting training of 7 machine learning models (using 5-fold cross-validation)")
            logger.info("=" * 80)
            logger.info("Estimated time: 30-60 minutes")
            logger.info("Models: RandomForest, XGBoost, GradientBoosting, ExtraTrees, KNN, SVM, LogisticRegression")
            logger.info("-" * 80)
            
            builder = build_multiple_models(
                delirium_df=delirium_df,
                non_delirium_df=non_delirium_df,
                selected_features=None,  # Feature selection inside Pipeline to avoid data leakage
                output_dir=self.model_dir,
                optimize=True  # Optimize hyperparameters using 5-fold CV
            )
            
            if builder is not None:
                logger.info("\n" + "=" * 80)
                logger.info("✓ Model training successful!")
                logger.info("=" * 80)
                logger.info("✓ Trained 7 models: RandomForest, XGBoost, GradientBoosting, ExtraTrees, KNN, SVM, LogisticRegression")
                logger.info("✓ Used 5-fold stratified cross-validation for hyperparameter optimization")
                logger.info("✓ Test set ratio: 25%")
                logger.info(f"✓ Results saved in: {self.model_dir}")
                
                # Read performance summary
                summary_file = os.path.join(self.model_dir, 'model_performance_summary.csv')
                if os.path.exists(summary_file):
                    logger.info("\n📊 Model Performance Ranking (by AUC):")
                    summary = pd.read_csv(summary_file)
                    if 'model' in summary.columns and 'test_auc' in summary.columns:
                        summary_sorted = summary.sort_values('test_auc', ascending=False)
                        for idx, row in summary_sorted.head(3).iterrows():
                            logger.info(f"  {idx+1}. {row['model']:20s} AUC = {row['test_auc']:.4f}")
                        self.best_model = summary_sorted.iloc[0]['model']
                
                return True
            else:
                logger.error("Model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def train_model_legacy(self, delirium_file, non_delirium_file):
        """Train model using traditional method (as backup)"""
        logger.info("Building model using traditional method...")
        try:
            # Load feature data
            logger.info(f"Loading feature data: {delirium_file} and {non_delirium_file}")
            delirium_df = pd.read_csv(delirium_file)
            non_delirium_df = pd.read_csv(non_delirium_file)
            logger.info(f"Delirium samples: {len(delirium_df)}, Non-delirium samples: {len(non_delirium_df)}")
            
            # Data type handling: Remove non-numeric columns (keep file/label)
            def cleanup(df):
                non_num = df.select_dtypes(exclude=['number']).columns.tolist()
                for col in list(non_num):
                    if col not in ['file', 'label']:
                        df = df.drop(columns=[col])
                return df
            delirium_df = cleanup(delirium_df)
            non_delirium_df = cleanup(non_delirium_df)
            
            # Ensure common columns
            common_cols = list(set(delirium_df.columns) & set(non_delirium_df.columns))
            delirium_df = delirium_df[common_cols]
            non_delirium_df = non_delirium_df[common_cols]
            
            # Handle missing values
            numeric_cols = delirium_df.select_dtypes(include=['number']).columns
            delirium_df[numeric_cols] = delirium_df[numeric_cols].fillna(delirium_df[numeric_cols].mean())
            non_delirium_df[numeric_cols] = non_delirium_df[numeric_cols].fillna(non_delirium_df[numeric_cols].mean())
            
            # Add labels
            if 'label' not in delirium_df.columns:
                delirium_df['label'] = 1
            if 'label' not in non_delirium_df.columns:
                non_delirium_df['label'] = 0

            # Merge and split (split first, then select, to avoid leakage)
            combined = pd.concat([delirium_df, non_delirium_df], ignore_index=True)
            feature_cols = [c for c in combined.columns if c not in ['label', 'file']]
            X = combined[feature_cols]
            y = combined['label']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y)
                
            # Simplified feature selection on training set only: correlation filtering + mutual information ranking + EPV constraint
            import numpy as np
            from sklearn.feature_selection import mutual_info_classif

            # Correlation filtering
            try:
                corr = np.corrcoef(X_train.values, rowvar=False)
                keep = np.ones(corr.shape[0], dtype=bool)
                for i in range(corr.shape[0]):
                    if not keep[i]:
                        continue
                    high = (np.abs(corr[i]) > 0.9)
                    high[:i+1] = False
                    keep[np.where(high)[0]] = False
                kept_cols = [c for c, k in zip(feature_cols, keep) if k]
                X_train_f = X_train[kept_cols]
            except Exception:
                kept_cols = feature_cols
                X_train_f = X_train

            # Mutual information
            try:
                mi = mutual_info_classif(X_train_f, y_train, random_state=42)
                ranks = np.argsort(-mi)
            except Exception:
                # Fallback: sort by variance
                var = np.var(X_train_f.values, axis=0)
                ranks = np.argsort(-var)

            # EPV constraint
            pos = int(np.sum(y_train == 1))
            k_epv = max(5, min(30, pos // 10 if pos > 0 else 10))
            top_idx = ranks[:k_epv]
            selected_features = [kept_cols[i] for i in top_idx]

            # Apply same feature selection to test set
            X_train_sel = X_train[selected_features].copy()
            X_test_sel = X_test[selected_features].copy()
            X_train_sel = X_train_sel.fillna(0)
            X_test_sel = X_test_sel.fillna(0)

            # Build and evaluate model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            model.fit(X_train_sel, y_train)
            
            y_pred = model.predict(X_test_sel)
            try:
                y_proba = model.predict_proba(X_test_sel)[:, 1]
            except Exception:
                y_proba = (y_pred == 1).astype(float)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            logger.info(f"Model evaluation results:")
            logger.info(f"- Accuracy: {accuracy:.4f}")
            logger.info(f"- Precision: {precision:.4f}")
            logger.info(f"- Recall: {recall:.4f}")
            logger.info(f"- F1 Score: {f1:.4f}")
            logger.info(f"- AUC: {auc:.4f}")
            
            # Save model and features
            model_path = os.path.join(self.model_dir, "legacy_delirium_model.pkl")
            feature_path = os.path.join(self.model_dir, "legacy_model_features.pkl")
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(feature_path, 'wb') as f:
                pickle.dump(selected_features, f)
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Feature list saved to: {feature_path}")
            
            self.best_model = "legacy_delirium_model.pkl"
            return True
        except Exception as e:
            logger.error(f"Legacy model building failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, file_path=None):
        """Perform prediction
        
        Args:
            file_path: File path to predict, if None use test dataset
        """
        logger.info("Starting delirium risk prediction...")
        
        try:
            # Find model file
            model_file = None
            feature_file = None
            
            # First check if best model attribute exists
            if hasattr(self, 'best_model') and self.best_model:
                model_file = os.path.join(self.model_dir, self.best_model)
                logger.info(f"Using best model: {model_file}")
                
                # Try to find corresponding feature file
                feature_base = self.best_model.replace('.pkl', '_features.pkl')
                feature_file = os.path.join(self.model_dir, feature_base)
            
            # Prioritize reading best model from model metadata
            meta_path = os.path.join(self.model_dir, 'model_metadata.pkl')
            if (not model_file or not os.path.exists(model_file)) and os.path.exists(meta_path):
                try:
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    best_name = meta.get('best_model')
                    if best_name:
                        candidate = os.path.join(self.model_dir, f"{best_name}_model.pkl") if not best_name.endswith('.pkl') else os.path.join(self.model_dir, best_name)
                        if os.path.exists(candidate):
                            model_file = candidate
                            logger.info(f"Selected best model from metadata: {model_file}")
                except Exception as e:
                    logger.warning(f"Failed to read model metadata: {str(e)}")
            
            # If model file not found, search for possible model files
            if not model_file or not os.path.exists(model_file):
                logger.warning(f"Specified model file not found: {model_file}")
                
                # Search model directory
                model_candidates = []
                for root, _, files in os.walk(self.model_dir):
                    for file in files:
                        if file.endswith('.pkl') and 'model' in file.lower() and 'feature' not in file.lower():
                            model_candidates.append(os.path.join(root, file))
                
                if model_candidates:
                    # Use most recent model file
                    model_candidates.sort(key=os.path.getmtime, reverse=True)
                    model_file = model_candidates[0]
                    logger.info(f"Found most recent model file: {model_file}")
                    
                    # Try to find corresponding feature file
                    model_base = os.path.basename(model_file)
                    feature_base = model_base.replace('.pkl', '_features.pkl')
                    feature_file = os.path.join(os.path.dirname(model_file), feature_base)
                    
                    # If corresponding feature file not found, search for any feature file
                    if not os.path.exists(feature_file):
                        feature_candidates = []
                        for root, _, files in os.walk(self.model_dir):
                            for file in files:
                                if file.endswith('.pkl') and 'feature' in file.lower():
                                    feature_candidates.append(os.path.join(root, file))
                        
                        if feature_candidates:
                            feature_candidates.sort(key=os.path.getmtime, reverse=True)
                            feature_file = feature_candidates[0]
                            logger.info(f"Found feature file: {feature_file}")
                
                # If still no model file found, try to find legacy model
                if not model_file:
                    legacy_model = os.path.join(self.model_dir, "legacy_delirium_model.pkl")
                    legacy_features = os.path.join(self.model_dir, "legacy_model_features.pkl")
                    
                    if os.path.exists(legacy_model):
                        model_file = legacy_model
                        logger.info(f"Using backup model: {model_file}")
                        
                        if os.path.exists(legacy_features):
                            feature_file = legacy_features
                            logger.info(f"Using backup feature file: {feature_file}")
            
            # If still no model file found, cannot continue
            if not model_file or not os.path.exists(model_file):
                logger.error("No available model file found, cannot perform prediction")
                return False
                
            # If feature file not found, try to find any feature file in model directory
            if not feature_file or not os.path.exists(feature_file):
                logger.warning("Corresponding feature file not found, trying to find any feature file")
                for root, _, files in os.walk(self.model_dir):
                    for file in files:
                        if 'feature' in file.lower() and file.endswith('.pkl'):
                            feature_file = os.path.join(root, file)
                            logger.info(f"Found feature file: {feature_file}")
                            break
            
            # Initialize predictor
            logger.info(f"Initializing predictor, model file: {model_file}, feature file: {feature_file}")
            self.predictor = DeliriumPredictor(model_path=model_file, meta_path=feature_file)
            
            # If file not specified, use test data
            if file_path is None:
                # Find test data
                delirium_dir, non_delirium_dir = self.find_data_folders()
                
                # Find test files
                test_files = []
                for directory in [delirium_dir, non_delirium_dir]:
                    if directory:
                        files = find_ecg_files(directory, max_files=5)
                        test_files.extend(files)
                
                if not test_files:
                    logger.error("No test files found")
                    return False
                
                # Predict for each test file
                successful_predictions = 0
                for test_file in test_files:
                    output_dir = os.path.join(self.results_dir, os.path.basename(test_file).split('.')[0])
                    os.makedirs(output_dir, exist_ok=True)
                    
                    result = self.predictor.analyze_file(
                        test_file, 
                        output_dir=output_dir
                    )
                    
                    if result:
                        logger.info(f"File {os.path.basename(test_file)} prediction completed")
                        successful_predictions += 1
                    else:
                        logger.warning(f"File {os.path.basename(test_file)} prediction failed")
                
                if successful_predictions > 0:
                    logger.info(f"Successfully completed predictions for {successful_predictions}/{len(test_files)} files")
                    return True
                else:
                    logger.error("All file predictions failed")
                    return False
            else:
                # Predict single file
                output_dir = os.path.join(self.results_dir, os.path.basename(file_path).split('.')[0])
                os.makedirs(output_dir, exist_ok=True)
                
                result = self.predictor.analyze_file(
                    file_path, 
                    output_dir=output_dir
                )
                
                if result:
                    logger.info(f"File {os.path.basename(file_path)} prediction completed")
                    return True
                else:
                    logger.warning(f"File {os.path.basename(file_path)} prediction failed")
                    return False
            
        except Exception as e:
            logger.error(f"Prediction process failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        logger.info("Starting complete anesthesia delirium prediction pipeline...")
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed, cannot continue")
            return False
        
        # Process data
        if not self.process_data():
            logger.error("Data processing failed, cannot continue")
            return False
        
        # Train model
        if not self.train_model():
            logger.error("Model training failed, cannot continue")
            return False
        
        # Perform prediction
        if not self.predict():
            logger.error("Prediction process failed")
            return False
        
        logger.info("Complete pipeline executed successfully!")
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Anesthesia Delirium Prediction System')
    
    parser.add_argument('--mode', choices=['full', 'train', 'predict', 'test'], 
                        default='full',
                        help='Run mode: full (complete pipeline), train (training only), predict (prediction only), test (testing)')
    
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory path')
    
    parser.add_argument('--delirium', type=str, default=None,
                        help='Delirium group data directory path')
    
    parser.add_argument('--non_delirium', type=str, default=None,
                        help='Non-delirium group data directory path')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory path')
    
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Model directory path')
    
    parser.add_argument('--file', type=str, default=None,
                        help='File path to process in prediction mode')
    
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process, default is 100')
                        
    parser.add_argument('--check_env', action='store_true',
                        help='Run environment check and provide optimization recommendations')
    
    return parser.parse_args()


def main():
    """Main function"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # If environment check specified, run check tool
        if args.check_env:
            try:
                import check_env
                check_env.main()
                return 0
            except ImportError:
                logger.error("Environment check tool not found, please ensure check_env.py file exists")
                return 1
        
        # Check performance-related configuration
        try:
            import psutil
            memory_available = psutil.virtual_memory().available / (1024**3)  # GB
            logger.info(f"System available memory: {memory_available:.1f}GB")
            
            # If memory is less than 4GB, automatically enable memory-efficient mode
            if memory_available < 4.0 and CONFIG.get('MEMORY_EFFICIENT') is not True:
                logger.warning(f"Detected low available memory ({memory_available:.1f}GB), automatically enabling memory-efficient mode")
                CONFIG['MEMORY_EFFICIENT'] = True
            
            # If multiple CPU cores, recommend using parallel processing
            cpu_cores = psutil.cpu_count(logical=False) or 1
            if cpu_cores >= 4 and CONFIG.get('PARALLEL_PROCESSING') is not True:
                logger.info(f"Detected {cpu_cores} CPU cores, recommend enabling parallel processing")
        except ImportError:
            logger.warning("psutil module not installed, cannot detect system resources")
        
        # Set default delirium and non-delirium group directories
        if args.delirium is None:
            args.delirium = os.path.join('data', 'PND_long_sequences')
            logger.info(f"Using default delirium group directory: {args.delirium}")
        
        if args.non_delirium is None:
            args.non_delirium = os.path.join('data', 'NPND_long_sequences')
            logger.info(f"Using default non-delirium group directory: {args.non_delirium}")
        
        # Create system instance
        system = DeliriumSystem(
            data_dir=args.data_dir,
            delirium_dir=args.delirium,
            non_delirium_dir=args.non_delirium,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            max_files=args.max_files
        )
    
        # Run different functions based on mode
        if args.mode == 'full':
            logger.info("Running complete delirium prediction pipeline")
            result = system.run_full_pipeline()
        elif args.mode == 'train':
            logger.info("Training delirium prediction model")
            result = system.train_model()
        elif args.mode == 'predict':
            logger.info("Running delirium prediction")
            result = system.predict(args.file)
        else:
            logger.error(f"Invalid run mode: {args.mode}")
            return 1
        
        # Check if task completed successfully
        if result:
            logger.info("Task completed successfully")
            return 0
        else:
            logger.error("Task execution failed")
            return 1
    
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    
    try:
        main()
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Record end time
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds") 