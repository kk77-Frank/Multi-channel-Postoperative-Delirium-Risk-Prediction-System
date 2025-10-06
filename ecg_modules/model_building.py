#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Building Module

Contains functions and classes for building various machine learning models, primarily for anesthesia delirium prediction.
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from time import time
import seaborn as sns
import traceback  # Add traceback module import

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

# Define XGBClassifier variable for later checking
XGBClassifier = None
LGBMClassifier = None

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed, will use other models")

# Try to import TensorFlow and Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras not installed, will use other models")
    TF_AVAILABLE = False

# Try to import LightGBM
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not installed, will use other models")
    LGBM_AVAILABLE = False

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    TORCH_AVAILABLE = True
    
    # Check CUDA availability
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        DEVICE_COUNT = torch.cuda.device_count()
        DEVICE_NAME = torch.cuda.get_device_name(0)
        print(f"Detected GPU: {DEVICE_NAME}, device count: {DEVICE_COUNT}")
        DEFAULT_DEVICE = "cuda:0"
    else:
        print("No GPU detected, will use CPU")
        DEFAULT_DEVICE = "cpu"
        
    # Define PyTorch neural network model
    class PyTorchModel(nn.Module):
        def __init__(self, input_size, hidden_sizes, dropout_rate=0.3):
            super(PyTorchModel, self).__init__()
            self.layers = nn.ModuleList()
            
            # Input layer -> first hidden layer
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            
            # Remaining hidden layers
            for i in range(len(hidden_sizes)-1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            self.output = nn.Linear(hidden_sizes[-1], 2)  # Binary classification problem
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            x = self.output(x)
            return x
            
    # Define PyTorch classifier
    class PyTorchClassifier:
        def __init__(self, input_size, hidden_sizes=[64, 32], dropout_rate=0.3, 
                     lr=0.001, batch_size=32, epochs=100, device=None):
            self.input_size = input_size
            self.hidden_sizes = hidden_sizes
            self.dropout_rate = dropout_rate
            self.lr = lr
            self.batch_size = batch_size
            self.epochs = epochs
            
            # Improved device selection logic
            if device is None:
                if CUDA_AVAILABLE:
                    self.device = "cuda:0"
                else:
                    self.device = "cpu"
            else:
                self.device = device
            
            # If GPU specified but not available, issue warning
            if "cuda" in self.device and not CUDA_AVAILABLE:
                print(f"Warning: Requested device {self.device} not available, falling back to CPU")
                self.device = "cpu"
                
            self.model = None
            
        def fit(self, X, y):
            # Create model
            self.model = PyTorchModel(
                self.input_size,
                self.hidden_sizes,
                self.dropout_rate
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            print(f"Training PyTorch model using device: {self.device}")
            
            # Prepare data
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
            y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y).to(self.device)
            
            # Create dataset and data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Define optimizer and loss function
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()
            
            # When using GPU, try to enable mixed precision training for acceleration
            scaler = None
            use_amp = False
            
            # Check if mixed precision training is supported
            if "cuda" in self.device and torch.cuda.is_available():
                try:
                    # Correctly import modules required for mixed precision training
                    from torch.cuda.amp import GradScaler
                    from torch.cuda.amp import autocast
                    scaler = GradScaler()
                    print("Enabled mixed precision training for acceleration")
                    use_amp = True
                except ImportError:
                    print("Unable to enable mixed precision training, using standard precision")
                    use_amp = False
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                running_loss = 0.0
                for inputs, labels in dataloader:
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Use mixed precision training
                    if use_amp:
                        with autocast():
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)
                        
                        # Use scaler to handle gradients and optimization steps
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        # Standard training
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item()
                
                # Print loss every 10 epochs
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss/len(dataloader):.4f}")
            
            return self
            
        def predict(self, X):
            if self.model is None:
                raise Exception("Model not trained, please call fit() method first")
                
            # Prepare data
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
            
            # Prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                
            # Convert prediction results to NumPy array
            return predicted.cpu().numpy()
            
        def predict_proba(self, X):
            if self.model is None:
                raise Exception("Model not trained, please call fit() method first")
                
            # Prepare data
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
            
            # Prediction probabilities
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
            # Convert prediction probabilities to NumPy array
            return probabilities.cpu().numpy()
except ImportError:
    print("PyTorch not installed, will use other models")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"

# If CONFIG is not defined, provide default configuration
if 'CONFIG' not in globals():
    CONFIG = {
        'MAX_FILES': 100,
        'USE_GPU': CUDA_AVAILABLE,
        'USE_PYTORCH': TORCH_AVAILABLE,
        'USE_TENSORFLOW': TF_AVAILABLE,
        'RANDOM_SEED': 42,
        'TEST_SIZE': 0.2,
        'DEVICE': DEFAULT_DEVICE
    }
else:
    # Update configuration
    CONFIG.update({
        'USE_GPU': CUDA_AVAILABLE,
        'USE_PYTORCH': TORCH_AVAILABLE,
        'USE_TENSORFLOW': TF_AVAILABLE,
        'DEVICE': DEFAULT_DEVICE
    })

class ModelBuilder:
    """Machine learning model builder class"""
    
    def __init__(self, random_state=42):
        """
        Initialize model builder
        
        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.models = {}  # Initialize model dictionary
        self.performance_results = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.selected_features = None
        self.pca_applied = False
        self.pca_model = None
        self.scaler = None
        self.ensemble_models = {}
        
    def prepare_data(self, delirium_df, non_delirium_df, significant_features=None, test_size=0.25, apply_pca=False, pca_components=None):
        """
        Prepare training and testing data
        
        Args:
            delirium_df: Delirium group feature data
            non_delirium_df: Non-delirium group feature data
            significant_features: List of significant features
            test_size: Test set ratio
            apply_pca: Whether to apply PCA dimensionality reduction
            pca_components: Number of PCA components
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Check data type
        if not isinstance(delirium_df, pd.DataFrame) or not isinstance(non_delirium_df, pd.DataFrame):
            raise ValueError("Input data must be pandas DataFrame format")
            
        # Filter non-numeric columns
        print("Filtering non-numeric columns...")
        delirium_numeric = delirium_df.select_dtypes(include=['number'])
        non_delirium_numeric = non_delirium_df.select_dtypes(include=['number'])
        
        # Check filtered column count
        if delirium_numeric.shape[1] < delirium_df.shape[1]:
            print(f"Filtered {delirium_df.shape[1] - delirium_numeric.shape[1]} non-numeric columns from delirium group data")
        if non_delirium_numeric.shape[1] < non_delirium_df.shape[1]:
            print(f"Filtered {non_delirium_df.shape[1] - non_delirium_numeric.shape[1]} non-numeric columns from non-delirium group data")
            
        # Update to filtered data
        delirium_df = delirium_numeric
        non_delirium_df = non_delirium_numeric
        
        # Ensure both datasets have the same columns
        common_columns = list(set(delirium_df.columns) & set(non_delirium_df.columns))
        if not common_columns:
            raise ValueError("Delirium and non-delirium group data have no common feature columns")
            
        print(f"Using {len(common_columns)} common feature columns")
        delirium_df = delirium_df[common_columns]
        non_delirium_df = non_delirium_df[common_columns]
        
        # If significant features list is provided, only use these features
        if significant_features is not None and len(significant_features) > 0:
            # Ensure all significant features are in the data
            valid_features = [f for f in significant_features if f in common_columns]
            if len(valid_features) == 0:
                print("Warning: No valid significant features found, will use all features")
            else:
                print(f"Using {len(valid_features)}/{len(significant_features)} significant features")
                delirium_df = delirium_df[valid_features]
                non_delirium_df = non_delirium_df[valid_features]
        
        # Create labels
        delirium_labels = np.ones(len(delirium_df))
        non_delirium_labels = np.zeros(len(non_delirium_df))
        
        # Merge data
        X = pd.concat([delirium_df, non_delirium_df], axis=0).reset_index(drop=True)
        y = np.concatenate([delirium_labels, non_delirium_labels])
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Apply PCA dimensionality reduction (if needed)
        if apply_pca:
            from sklearn.decomposition import PCA
            
            # If component count is not specified, use default rule
            if pca_components is None:
                # Use minimum components that explain 95% variance
                pca = PCA(n_components=0.95)
                pca.fit(X)
                pca_components = pca.n_components_
                print(f"Auto-selected {pca_components} PCA components, explaining 95% variance")
            else:
                # Use specified component count
                pca = PCA(n_components=min(pca_components, X.shape[1]))
                
            # Apply PCA transformation
            X_pca = pca.fit_transform(X)
            print(f"Applied PCA dimensionality reduction: {X.shape[1]} features -> {X_pca.shape[1]} components")
            
            # Update data and flag
            X = X_pca
            self.pca_applied = True
            self.pca_model = pca
            
            # Create new feature names
            self.feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        # Split training and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Standardize features
        X_train, X_test = self.preprocess_data(X_train, X_test)
        
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def preprocess_data(self, X_train, X_test, method='standard'):
        """
        Data preprocessing
        
        Args:
            X_train: Training features
            X_test: Test features
            method: Standardization method ('standard' or 'minmax')
            
        Returns:
            Preprocessed X_train, X_test
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid standardization method, please choose 'standard' or 'minmax'")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        
        # Apply standardization and missing value handling
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns
        )
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train), 
            columns=X_train.columns
        )
        
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test), 
            columns=X_test.columns
        )
        
        return X_train, X_test
    
    def build_models(self):
        """
        Build various machine learning models
        
        Returns:
            Model dictionary
        """
        models = {}
        
        # Basic models
        models['LogisticRegression'] = LogisticRegression(random_state=self.random_state, max_iter=1000, 
                                        class_weight='balanced', solver='liblinear')
        
        models['RandomForest'] = RandomForestClassifier(random_state=self.random_state, n_estimators=100,
                                           class_weight='balanced')
        
        models['SVM'] = SVC(random_state=self.random_state, probability=True, 
                                class_weight='balanced')
        
        models['GradientBoosting'] = GradientBoostingClassifier(random_state=self.random_state)
        
        models['NeuralNetwork'] = MLPClassifier(random_state=self.random_state, max_iter=1000,
                                      hidden_layer_sizes=(100, 50), early_stopping=True)
        
        models['KNN'] = KNeighborsClassifier(n_neighbors=5)
        
        models['AdaBoost'] = AdaBoostClassifier(random_state=self.random_state)
        
        # Advanced models
        try:
            from xgboost import XGBClassifier
            models['XGBoost'] = XGBClassifier(random_state=self.random_state, 
                                           use_label_encoder=False, eval_metric='logloss')
        except ImportError:
            print("XGBoost not installed, skipping")
            
        try:
            from lightgbm import LGBMClassifier
            # Modify LightGBM parameters to avoid "no more leaves" warning
            models['LightGBM'] = LGBMClassifier(
                random_state=self.random_state,
                min_child_samples=2,  # Reduce minimum leaf samples
                min_data_in_leaf=1,   # Minimum data per leaf
                min_split_gain=0,     # Minimum split gain
                min_sum_hessian_in_leaf=1e-3,  # Minimum Hessian sum in leaf
                verbose=-1            # Disable redundant warnings
            )
        except ImportError:
            print("LightGBM not installed, skipping")
            
        # Save model dictionary
        self.models = models
        
        return models
        
    def save_model(self, model_name, file_path=None):
        """
        Save model to file
        
        Args:
            model_name: Model name
            file_path: Save file path
        """
        # Check if model exists
        if not hasattr(self, 'models') or not self.models:
            print("Warning: Model dictionary is empty, attempting to create default model")
            self.build_models()
            
        # If specified model name doesn't exist, use RandomForest as default model
        if model_name not in self.models:
            print(f"Warning: Model '{model_name}' not found, attempting to use RandomForest as default model")
            if 'RandomForest' not in self.models:
                print("Creating default RandomForest model")
                self.models['RandomForest'] = RandomForestClassifier(random_state=self.random_state)
            model_name = 'RandomForest'
            
        model = self.models[model_name]
        
        if file_path is None:
            file_path = f"{model_name}_delirium_model.pkl"
            
        # Save model
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to: {file_path}")
            
            # Save metadata
            meta_file = file_path.replace('.pkl', '_meta.pkl')
            meta_data = {
                'model_name': model_name,
                'feature_names': self.feature_names,
                'selected_features': self.selected_features,
                'pca_applied': self.pca_applied,
                'pca_model': self.pca_model,
                'scaler': self.scaler
            }
            
            with open(meta_file, 'wb') as f:
                pickle.dump(meta_data, f)
            print(f"Model metadata saved to: {meta_file}")
            
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def build_ensemble_models(self, base_models=None):
        """
        Build ensemble learning models
        
        Args:
            base_models: Base model dictionary, if None will use build_models to build
            
        Returns:
            Ensemble model dictionary
        """
        if base_models is None:
            base_models = self.build_models()
        
        # Select better performing models as base models
        estimators = []
        for name, model in base_models.items():
            if name in ["RandomForest", "GradientBoosting"]:
                estimators.append((name, model))
            elif name == "XGBoost" and XGBClassifier is not None:
                estimators.append((name, model))
            elif name == "LightGBM" and LGBMClassifier is not None:
                estimators.append((name, model))
        
        # Ensure at least two models for ensemble
        if len(estimators) < 2:
            # If not enough models, add more base models
            for name, model in base_models.items():
                if name not in [est[0] for est in estimators]:
                    estimators.append((name, model))
                    if len(estimators) >= 3:  # Use at most 3 models
                        break
        
        # Voting ensemble
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use prediction probabilities for weighted voting
            n_jobs=-1
        )
        
        # Stacking ensemble
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5,
            n_jobs=-1
        )
        
        # Bagging ensemble
        bagging_clf = BaggingClassifier(
            estimator=base_models["RandomForest"],
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        ensemble_models = {
            "VotingEnsemble": voting_clf,
            "StackingEnsemble": stacking_clf,
            "BaggingEnsemble": bagging_clf
        }
        
        self.ensemble_models = ensemble_models
        return ensemble_models
    
    def advanced_optimize_model(self, model_name, X_train, y_train, param_space=None, cv=5, n_iter=20):
        """
        Use random search optimization to find best hyperparameters
        
        Args:
            model_name: Model name
            X_train: Training features
            y_train: Training labels
            param_space: Parameter space, if None will use default space
            cv: Cross-validation folds
            n_iter: Random search iteration count
            
        Returns:
            Optimized model and best parameters
        """
        # Get model
        models = self.build_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = models[model_name]
        
        # Define default parameter space for different models
        if param_space is None:
            if model_name == "RandomForest":
                param_space = {
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [3, 4, 5, 6, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif model_name == "XGBoost":
                param_space = {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.03, 0.05, 0.1],
                    'max_depth': [3, 4, 5, 6],
                    'min_child_weight': [1, 3, 5, 7],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0.1, 0.5, 1.0, 1.5]
                }
            elif model_name == "LightGBM":
                param_space = {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.03, 0.05, 0.1],
                    'max_depth': [3, 4, 5, 6],
                    'num_leaves': [7, 15, 31],
                    'min_child_samples': [10, 20, 30],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                    'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0.0, 0.5, 1.0, 5.0],
                    'min_split_gain': [0.0, 0.1, 0.2]
                }
            elif model_name == "SVM":
                param_space = {
                    'C': [0.1, 0.5, 1, 5, 10],
                    'gamma': [0.001, 0.01, 0.1, 0.5, 'scale', 'auto'],
                    'kernel': ['linear', 'rbf', 'poly']
                }
            elif model_name == "LogisticRegression":
                param_space = {
                    'C': [0.1, 0.5, 1, 5, 10],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['liblinear', 'saga'],
                    'l1_ratio': [0, 0.2, 0.5, 0.8, 1.0]
                }
            elif model_name == "NeuralNetwork":
                param_space = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01]
                }
            else:
                # Other models use simple parameter space
                param_space = {}
        
        # If parameter space is not defined or empty, use default model
        if not param_space:
            print(f"Parameter space for {model_name} not defined, using default model")
            return model, {}
        
        # Adjust search intensity based on dataset size
        sample_size = len(X_train)
        if sample_size < 100:
            # Reduce iteration count for small datasets
            n_iter = min(10, n_iter)
            print(f"Dataset is small ({sample_size} samples), reducing search iterations to {n_iter}")
        
        # Create random search CV object
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='balanced_accuracy',  # Use balanced accuracy to handle imbalanced data
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        # Perform optimization
        print(f"Starting random search optimization for {model_name} (n_iter={n_iter})...")
        start_time = time()
        random_search.fit(X_train, y_train)
        end_time = time()
        
        print(f"Random search optimization completed, time: {end_time - start_time:.2f} seconds")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best performance: {random_search.best_score_:.4f}")
        
        # Save best parameters
        self.best_params = random_search.best_params_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def visualize_calibration_curve(self, model_name, X_test, y_test, save_path=None):
        """
        Visualize probability calibration curve
        
        Args:
            model_name: Model name
            X_test: Test features
            y_test: Test labels
            save_path: Save path
        """
        if model_name not in self.models:
            print(f"Model {model_name} not trained")
            return
        
        # Get model
        model = self.models[model_name]
        
        # Get prediction probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        try:
            from sklearn.calibration import calibration_curve
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            
            # Plot calibration curve
            plt.figure(figsize=(10, 8))
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
            
            # Plot perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Probability')
            plt.title(f'{model_name} Probability Calibration Curve')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                plt.close()
            else:
                plt.show()
        except ImportError:
            print("Cannot import calibration_curve, skipping calibration curve plotting")
    
    def plot_confusion_matrix(self, model_name, X_test, y_test, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            model_name: 模型名称
            X_test: 测试特征
            y_test: 测试标签
            save_path: 保存路径
        """
        if model_name not in self.models:
            print(f"模型 {model_name} 未训练")
            return
        
        # 获取模型
        model = self.models[model_name]
        
        # Prediction
        y_pred = model.predict(X_test)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{model_name} 混淆矩阵')
        
        # 添加具体的指标
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        plt.figtext(0.15, 0.05, f'敏感性: {sensitivity:.2f}', fontsize=12)
        plt.figtext(0.35, 0.05, f'特异性: {specificity:.2f}', fontsize=12)
        plt.figtext(0.55, 0.05, f'PPV: {ppv:.2f}', fontsize=12)
        plt.figtext(0.75, 0.05, f'NPV: {npv:.2f}', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
            
        return cm
    
    def evaluate_ensemble_models(self, X_test, y_test, verbose=True):
        """
        评估集成模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            verbose: 是否打印详细信息
            
        Returns:
            性能指标字典
        """
        ensemble_results = {}
        
        for model_name, model in self.ensemble_models.items():
            # Prediction
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # 计算各种指标
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            ensemble_results[model_name] = metrics
            
            if verbose:
                print(f"\n{model_name} 性能评估:")
                print(f"准确率: {metrics['accuracy']:.4f}")
                print(f"精确率: {metrics['precision']:.4f}")
                print(f"召回率: {metrics['recall']:.4f}")
                print(f"F1分数: {metrics['f1']:.4f}")
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
                print(f"PR AUC: {metrics['pr_auc']:.4f}")
                print("\n分类报告:")
                print(metrics['classification_report'])
        
        return ensemble_results
    
    def train_stratified_cv(self, model_name, X, y, cv=5, param_grid=None, use_advanced_opt=True, n_iter=20):
        """
        Train and evaluate model using stratified cross-validation
        
        Args:
            model_name: Model name
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds
            param_grid: Parameter grid
            use_advanced_opt: Whether to use advanced optimization (random search)
            n_iter: Random search iteration count
            
        Returns:
            Cross-validation performance metrics
        """
        # Get model
        models = self.build_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        base_model = models[model_name]
        
        # Optimize hyperparameters
        if param_grid is not None or use_advanced_opt:
            if use_advanced_opt:
                best_model, best_params = self.advanced_optimize_model(
                    model_name, X, y, param_grid, cv, n_iter)
            else:
                best_model, best_params = self.optimize_model(
                    model_name, X, y, param_grid, cv)
        else:
            best_model = base_model
        
        # Create stratified cross-validation
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Initialize performance metrics list
        cv_results = {
            'mean_accuracy': 0.0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'mean_f1': 0.0,
            'mean_auc': 0.0,
            'mean_pr_auc': 0.0,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'pr_auc': []
        }
        
        # Perform cross-validation
        print(f"\nStarting stratified {cv}-fold cross-validation for {model_name}...")
        
        for i, (train_idx, test_idx) in enumerate(stratified_cv.split(X, y)):
            # Split data
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            best_model.fit(X_train_cv, y_train_cv)
            
            # Prediction
            y_pred = best_model.predict(X_test_cv)
            
            # For models supporting predict_proba
            if hasattr(best_model, 'predict_proba'):
                y_prob = best_model.predict_proba(X_test_cv)[:, 1]
            else:
                y_prob = None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test_cv, y_pred, y_prob)
            
            # Save results
            for key, value in metrics.items():
                if key in cv_results:
                    cv_results[key].append(value)
            
            print(f"  Fold {i+1}/{cv}: Accuracy={metrics.get('Accuracy', 0):.4f}, AUC={metrics.get('AUC', 'N/A')}")
        
        # Calculate average performance
        mean_results = {key: np.mean(values) for key, values in cv_results.items() if values}
        std_results = {key: np.std(values) for key, values in cv_results.items() if values}
        
        print("\nCross-validation average performance:")
        for key, value in mean_results.items():
            print(f"{key}: {value:.4f} ± {std_results[key]:.4f}")
        
        # Train final model on entire dataset
        best_model.fit(X, y)
        self.models[model_name] = best_model
        
        return mean_results, std_results, cv_results
    
    def train_evaluate_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary of performance evaluation results
        """
        # Build multiple models
        self.build_models()
        
        # If classes are severely imbalanced, apply upsampling
        # Convert float y_train to int type for np.bincount
        y_train_int = y_train.astype(int)
        class_counts = np.bincount(y_train_int)
        if len(class_counts) > 1 and min(class_counts) / max(class_counts) < 0.5:
            print("Detected class imbalance, applying upsampling...")
            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Check dataset size
        is_small_dataset = len(X_train) < 30
        if is_small_dataset:
            print(f"Warning: Small dataset ({len(X_train)} samples), will simplify model training process")
        
        # Train all basic models
        results = {}
        self.performance_results = pd.DataFrame()  # Initialize performance results DataFrame
        
        for model_name, model in self.models.items():
            print(f"\nTraining model: {model_name}")
            
            try:
                # For small datasets, train model directly without complex optimization
                if is_small_dataset:
                    print(f"Training {model_name} directly (skipping optimization)...")
                    model.fit(X_train_resampled, y_train_resampled)
                else:
                    # Use advanced optimization method to train model
                    try:
                        optimized_model, _ = self.advanced_optimize_model(model_name, X_train_resampled, y_train_resampled)
                        self.models[model_name] = optimized_model
                    except Exception as e:
                        print(f"Optimizing {model_name} failed: {str(e)}")
                        print("Training model with default parameters...")
                        model.fit(X_train_resampled, y_train_resampled)
                
                # Evaluate model
                metrics = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = metrics
                
                # Add results to performance_results DataFrame
                metrics_df = pd.DataFrame({
                    'Model': [model_name],
                    'AUC': [metrics.get('AUC', 0.5)],
                    'Accuracy': [metrics.get('Accuracy', 0)],
                    'Precision': [metrics.get('Precision', 0)],
                    'Recall': [metrics.get('Recall', 0)],
                    'F1_Score': [metrics.get('F1_Score', 0)]
                })
                
                self.performance_results = pd.concat([self.performance_results, metrics_df], ignore_index=True)
                
            except Exception as e:
                print(f"{model_name} training failed: {str(e)}")
                traceback.print_exc()
        
        # Ensure performance_results is not empty
        if len(self.performance_results) == 0:
            print("Warning: All model training failed, creating default performance results")
            self.performance_results = pd.DataFrame({
                'Model': ['RandomForest'],
                'AUC': [0.5],
                'Accuracy': [0.5],
                'Precision': [0.5],
                'Recall': [0.5],
                'F1_Score': [0.5]
            })
        
        # Build ensemble models
        if len(results) >= 3:  # Need at least 3 base models to build ensemble
            print("\nBuilding ensemble models...")
            try:
                # Select top 3 performing models to build ensemble
                sorted_models = sorted(
                    results.items(), 
                    key=lambda x: x[1]['F1_Score'] if 'F1_Score' in x[1] else 0, 
                    reverse=True
                )
                top_models = [model[0] for model in sorted_models[:3]]
                
                # Build Voting ensemble
                self.build_voting_ensemble(X_train_resampled, y_train_resampled, top_models)
                ensemble_metrics = self.evaluate_model('voting_ensemble', X_test, y_test)
                results['voting_ensemble'] = ensemble_metrics
                
                # Build Stacking ensemble
                self.build_stacked_ensemble(X_train_resampled, y_train_resampled, top_models)
                ensemble_metrics = self.evaluate_model('stacked_ensemble', X_test, y_test)
                results['stacked_ensemble'] = ensemble_metrics
            except Exception as e:
                print(f"Ensemble model building failed: {str(e)}")
                traceback.print_exc()
        else:
            print("Insufficient models available, skipping ensemble model building")
        
        # Sort by F1 score and output results
        if results:
            sorted_results = sorted(
                results.items(), 
                key=lambda x: x[1]['F1_Score'] if 'F1_Score' in x[1] else 0, 
                reverse=True
            )
            
            print("\n=== Model Performance Ranking ===")
            for i, (model, metrics) in enumerate(sorted_results):
                print(f"{i+1}. {model}: F1={metrics.get('F1_Score', 0):.4f}, AUC={metrics.get('AUC', 0):.4f}")
        else:
            print("Warning: No successfully trained models")
            
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate model performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred)
        metrics['Recall'] = recall_score(y_true, y_pred)
        metrics['F1_Score'] = f1_score(y_true, y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC and probability-related metrics
        if y_pred_proba is not None:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
            metrics['Average_Precision'] = average_precision_score(y_true, y_pred_proba)
        else:
            metrics['AUC'] = np.nan
            metrics['Average_Precision'] = np.nan
            
        return metrics
    
    def optimize_model(self, model_name, X_train, y_train, param_grid=None, cv=5):
        """
        Optimize model hyperparameters using grid search
        
        Args:
            model_name: Model name to optimize
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid, if None will use default parameter grid
            cv: Cross-validation folds
            
        Returns:
            Best model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found, please train model first")
            
        model = self.models[model_name]
        
        # If no parameter grid provided, use default parameters
        if param_grid is None:
            if model_name == "LogisticRegression":
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['liblinear', 'saga']
                }
            elif model_name == "RandomForest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_name == "SVM":
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                    'kernel': ['linear', 'rbf', 'poly']
                }
            elif model_name == "XGBoost":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            elif model_name == "LightGBM":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 50, 100],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.8, 1.0]
                }
            else:
                print(f"No default parameter grid defined for model '{model_name}', skipping optimization")
                return model
        
        print(f"Starting parameter optimization for {model_name}...")
        start_time = time()
        
        # Create cross-validation object
        cv_stratified = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Create grid search object
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_stratified, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        # Execute hyperparameter search
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数和模型
        self.best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        self.models[model_name] = best_model
        
        print(f"Parameter optimization completed, time: {time() - start_time:.2f} seconds")
        print(f"Best parameters: {self.best_params}")
        print(f"AUC before optimization: {grid_search.cv_results_['mean_test_score'][0]:.4f}")
        print(f"AUC after optimization: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def evaluate_model(self, model_name, X_test, y_test, verbose=True):
        """
        Detailed evaluation of specified model
        
        Args:
            model_name: Model name to evaluate
            X_test: Test features
            y_test: Test labels
            verbose: Whether to print detailed results
            
        Returns:
            Evaluation metrics dictionary
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found, please train model first")
            
        model = self.models[model_name]
        
        # Prediction
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate detailed evaluation metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        if verbose:
            print(f"\n{model_name} Model Evaluation Results:")
            print(f"- Accuracy: {metrics['Accuracy']:.4f}")
            print(f"- Precision: {metrics['Precision']:.4f}")
            print(f"- Recall: {metrics['Recall']:.4f}")
            print(f"- F1 Score: {metrics['F1_Score']:.4f}")
            print(f"- Specificity: {metrics['Specificity']:.4f}")
            
            if y_pred_proba is not None:
                print(f"- AUC: {metrics['AUC']:.4f}")
                print(f"- Average Precision: {metrics['Average_Precision']:.4f}")
                
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
        return metrics
    
    def plot_roc_curve(self, model_name, X_test, y_test, save_path=None):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found, please train model first")
            
        model = self.models[model_name]
        
        if not hasattr(model, "predict_proba"):
            print(f"Warning: Model '{model_name}' does not support probability prediction, cannot plot ROC curve")
            return None

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
            
        return fig
    
    def plot_feature_importance(self, model_name, top_n=15, save_path=None):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found, please train model first")
            
        model = self.models[model_name]
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            print(f"Warning: Cannot extract feature importance for model '{model_name}'")
            return None
            
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        self.feature_importance = feature_importance
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(feature_importance['Feature'][::-1], feature_importance['Importance'][::-1])
        
        try:
            import matplotlib.cm as cm
            colors = cm.get_cmap('Blues')(np.linspace(0.4, 0.8, len(bars)))
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
        except Exception:
            pass
            
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name} - Top {top_n} Important Features')
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
            
        return fig
        
    def remove_highly_correlated_features(self, X, threshold=0.85):
        if not isinstance(X, pd.DataFrame):
            print("Input must be DataFrame to remove highly correlated features")
            return X
            
        print(f"Checking and removing redundant features with correlation > {threshold}...")
        
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        correlated_pairs = []
        for col in to_drop:
            correlated_features = upper[col][upper[col] > threshold].index.tolist()
            for feat in correlated_features:
                correlated_pairs.append((col, feat, corr_matrix.loc[col, feat]))
                
        if correlated_pairs:
            print(f"Found {len(correlated_pairs)} pairs of highly correlated features:")
            for i, (feat1, feat2, corr) in enumerate(sorted(correlated_pairs, key=lambda x: x[2], reverse=True)[:5]):
                print(f"  {i+1}. {feat1} and {feat2}: correlation = {corr:.3f}")
            
            if len(correlated_pairs) > 5:
                print(f"  ... and {len(correlated_pairs)-5} other highly correlated feature pairs")
        
        if to_drop:
            print(f"Removing {len(to_drop)} redundant features...")
            retained_features = [col for col in X.columns if col not in to_drop]
            X_reduced = X[retained_features]
            return X_reduced
        else:
            print("No highly correlated features found to remove")
            return X
    
    def check_class_imbalance(self, y):

        class_counts = Counter(y)
        if len(class_counts) > 1:
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)
            imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
            
            print(f"Class distribution: {dict(class_counts)}")
            print(f"Imbalance ratio: {imbalance_ratio:.2f} : 1 (majority:minority)")
            
            return class_counts, imbalance_ratio
        else:
            print("Warning: Only one class present")
            return class_counts, None
    
    def evaluate_model_with_cv(self, model_name, X, y, cv=5, scoring=None):
        
        models = self.build_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
            
        model = models[model_name]
        
        # Default scoring metrics
        if scoring is None:
            scoring = {
                'accuracy': 'accuracy',
                'balanced_accuracy': 'balanced_accuracy',
                'f1': 'f1',
                'precision': 'precision',
                'recall': 'recall',
                'roc_auc': 'roc_auc'
            }
        
        from sklearn.model_selection import cross_validate
        cv_results = cross_validate(
            model, X, y, 
            cv=cv, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        mean_results = {}
        for key in cv_results:
            if key.startswith('test_'):
                metric = key.replace('test_', '')
                mean_results[metric] = np.mean(cv_results[key])
                mean_results[f'{metric}_std'] = np.std(cv_results[key])
        
）
        for metric in scoring:
            train_key = f'train_{metric}'
            test_key = f'test_{metric}'
            if train_key in cv_results and test_key in cv_results:
                train_mean = np.mean(cv_results[train_key])
                test_mean = np.mean(cv_results[test_key])
                mean_results[f'{metric}_overfit'] = train_mean - test_mean
        
        return mean_results
    
    def evaluate_all_models_with_cv(self, X, y, cv=5, top_n=3):
       
        models = self.build_models()
        results = []
        
        print(f"Evaluating {len(models)} models using {cv}-fold cross-validation...")
        
        # Check class imbalance
        self.check_class_imbalance(y)
        
        for i, (name, model) in enumerate(models.items()):
            print(f"  Evaluating model {i+1}/{len(models)}: {name}...")
            try:
                cv_results = self.evaluate_model_with_cv(name, X, y, cv=cv)
                cv_results['model'] = name
                results.append(cv_results)
                
                # Output main metrics
                print(f"    - Accuracy: {cv_results['accuracy']:.4f} (±{cv_results['accuracy_std']:.4f})")
                print(f"    - Balanced Accuracy: {cv_results['balanced_accuracy']:.4f} (±{cv_results['balanced_accuracy_std']:.4f})")
                if 'roc_auc' in cv_results:
                    print(f"    - ROC AUC: {cv_results['roc_auc']:.4f} (±{cv_results['roc_auc_std']:.4f})")
                
                # Check for overfitting
                if 'accuracy_overfit' in cv_results:
                    overfit = cv_results['accuracy_overfit']
                    if overfit > 0.2:
                        print(f"    - Warning: Possible overfitting (train-test difference: {overfit:.4f})")
            except Exception as e:
                print(f"    - Evaluation failed: {str(e)}")
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Sort by balanced accuracy
        if 'balanced_accuracy' in df_results.columns:
            df_results = df_results.sort_values('balanced_accuracy', ascending=False)
        elif 'accuracy' in df_results.columns:
            df_results = df_results.sort_values('accuracy', ascending=False)
            
        return df_results.head(top_n)

    def apply_smote(self, X_train, y_train, random_state=None):
        try:
            from sklearn.utils import resample
            
            # Get samples of different classes separately
            X_majority = X_train[y_train == 0]
            X_minority = X_train[y_train == 1]
            
            y_majority = y_train[y_train == 0]
            y_minority = y_train[y_train == 1]
            
            if len(X_minority) > 0:
                X_minority_upsampled, y_minority_upsampled = resample(
                    X_minority, y_minority,
                    replace=True,
                    n_samples=len(X_majority),
                    random_state=random_state or self.random_state
                )
                
                X_resampled = pd.concat([X_majority, X_minority_upsampled])
                y_resampled = pd.concat([y_majority, y_minority_upsampled])
                
                temp_data = pd.concat([X_resampled, y_resampled], axis=1)
                temp_data = temp_data.sample(frac=1, random_state=random_state or self.random_state)
                
                X_resampled = temp_data.iloc[:, :-1]
                y_resampled = temp_data.iloc[:, -1]
                
                print(f"Applied upsampling: original class distribution {Counter(y_train)} → balanced {Counter(y_resampled)}")
                return X_resampled, y_resampled
            else:
                print("Minority class samples are empty, cannot perform upsampling")
                return X_train, y_train
        except Exception as e:
            print(f"Upsampling failed: {str(e)}")
            return X_train, y_train
    
    def build_stacked_ensemble(self, X_train, y_train, top_n_models=3):
       
        # Evaluate individual models
        cv_results = self.evaluate_all_models_with_cv(X_train, y_train, cv=5, top_n=top_n_models)
        
        if cv_results.empty:
            print("Cannot build stacked ensemble: individual model evaluation failed")
            return None
            
        # Get top models
        top_models = cv_results['model'].tolist()
        print(f"Building stacked ensemble with top {len(top_models)} models: {top_models}")
        
        # Build base model list
        models = self.build_models()
        estimators = [(name, models[name]) for name in top_models if name in models]
        
        if len(estimators) < 2:
            print(f"Stacked ensemble requires at least 2 models, but only found {len(estimators)}")
            if len(estimators) == 1:
                return estimators[0][1]
            return None
            
        # Build stacked ensemble
        stack_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5,
            n_jobs=-1
        )
        
        # Train model
        print("Training stacked ensemble model...")
        stack_model.fit(X_train, y_train)
        
        # Save to trained models dictionary
        self.models['stacked_ensemble'] = stack_model
        
        return stack_model
        
    def build_voting_ensemble(self, X_train, y_train, top_n_models=3):

        # Evaluate individual models
        cv_results = self.evaluate_all_models_with_cv(X_train, y_train, cv=3, top_n=top_n_models)
        
        if cv_results.empty:
            print("Cannot build voting ensemble: individual model evaluation failed")
            return None
            
        # Get top models
        top_models = cv_results['model'].tolist()
        print(f"Building voting ensemble with top {len(top_models)} models: {top_models}")
        
        # Build base model list
        models = self.build_models()
        estimators = [(name, models[name]) for name in top_models if name in models]
        
        if len(estimators) < 2:
            print(f"Voting ensemble requires at least 2 models, but only found {len(estimators)}")
            if len(estimators) == 1:
                return estimators[0][1]
            return None
            
        vote_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        

        print("Training voting ensemble model...")
        vote_model.fit(X_train, y_train)
        
        # Save to trained models dictionary
        self.models['voting_ensemble'] = vote_model
        
        return vote_model

    def remove_low_variance_features(self, X, threshold=0.01):

        if not isinstance(X, pd.DataFrame):
            print("Input must be DataFrame to remove low variance features")
            return X
            
        print(f"Checking and removing features with variance < {threshold}...")
        
        # Calculate variance for each feature
        variances = X.var()
        
        # Find low variance features
        low_var_features = variances[variances < threshold].index.tolist()
        
        if low_var_features:
            print(f"Found {len(low_var_features)} low variance features:")
            for i, feature in enumerate(low_var_features[:5]):
                print(f"  {i+1}. {feature}: variance = {variances[feature]:.6f}")
            
            if len(low_var_features) > 5:
                print(f"  ... and {len(low_var_features)-5} other low variance features")
                
            # Remove low variance features
            X_reduced = X.drop(columns=low_var_features)
            print(f"After removing low variance features: {X.shape[1]} -> {X_reduced.shape[1]} features")
            return X_reduced
        else:
            print("No low variance features found")
            return X
    
    def select_features_by_importance(self, X, y, method='random_forest', n_features=None):

        if not isinstance(X, pd.DataFrame):
            print("Input must be DataFrame for feature selection")
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        # If feature count not specified, use heuristic rules
        if n_features is None:
            # Heuristic rules based on sample count and feature count
            n_samples = len(X)
            if n_samples < 100:
                # Small dataset, more strictly limit feature count
                n_features = min(10, X.shape[1])
            else:
                # Adjust feature count based on sample count
                n_features = min(int(np.sqrt(n_samples) * 2), X.shape[1])
            
            print(f"Auto-selecting {n_features} features (based on {n_samples} samples)")
            
        # Ensure n_features doesn't exceed available features
        n_features = min(n_features, X.shape[1])
        
        # Calculate feature importance based on method
        if method == 'random_forest':
            print(f"Selecting {n_features} features using random forest feature importance...")
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')
            rf.fit(X, y)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif method == 'mutual_info':
            print(f"Selecting {n_features} features using mutual information...")
            # Calculate mutual information
            mi = mutual_info_classif(X, y, random_state=self.random_state)
            
            # Create feature importance DataFrame
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': mi
            }).sort_values('importance', ascending=False)
            
        elif method == 'f_test':
            print(f"Selecting {n_features} features using F-test...")
            # Calculate F statistic
            f_values, _ = f_classif(X, y)
            
            # Create feature importance DataFrame
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': f_values
            }).sort_values('importance', ascending=False)
            
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
            
        # Select top features
        selected_features = importance.head(n_features)['feature'].tolist()
        
        # Display selected features
        print(f"Selected {len(selected_features)} features")
        print("Top 10 most important features:")
        top_n = min(10, len(importance))
        for i, row in importance.head(top_n).iterrows():
            print(f"  {i+1}. {row['feature']} (importance: {row['importance']:.4f})")
            
        # Save feature importance
        self.feature_importance = importance
        
        return selected_features, importance

    def optimal_train_test_split(self, X, y, test_size=0.25, random_state=None):
        if random_state is None:
            random_state = self.random_state
            
        # Use stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Check split results
        train_class_counts = Counter(y_train)
        test_class_counts = Counter(y_test)
        
        print("Training set class distribution:")
        for label, count in train_class_counts.items():
            print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
            
        print("Test set class distribution:")
        for label, count in test_class_counts.items():
            print(f"  Class {label}: {count} samples ({count/len(y_test)*100:.1f}%)")
            
        return X_train, X_test, y_train, y_test

    def generate_comprehensive_report(self, output_dir=None):

        if output_dir is None:
            output_dir = 'model_report'
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if performance results exist
        if not hasattr(self, 'performance_results') or not isinstance(self.performance_results, pd.DataFrame) or self.performance_results.empty:
            print("Warning: No model performance results available")
            return
            
        # 1. Generate model performance comparison chart
        try:
            from ecg_modules.visualization import Visualizer
            visualizer = Visualizer()
            
            # Create performance comparison chart
            metrics = ['Accuracy', 'AUC', 'F1_Score', 'Precision', 'Recall']
            available_metrics = [m for m in metrics if m in self.performance_results.columns]
            
            if available_metrics and 'Model' in self.performance_results.columns:
                chart_path = os.path.join(output_dir, 'model_performance_comparison.png')
                visualizer.plot_model_comparison(
                    self.performance_results, 
                    metrics=available_metrics,
                    title='Delirium Prediction Model Performance Comparison',
                    save_path=chart_path
                )
                print(f"Generated model performance comparison chart: {chart_path}")
        except Exception as e:
            print(f"Error generating performance comparison chart: {str(e)}")
            
        # 2. Get best model
        try:
            if 'Model' in self.performance_results.columns and 'AUC' in self.performance_results.columns:
                sorted_results = self.performance_results.sort_values('AUC', ascending=False)
                best_model_name = sorted_results.iloc[0]['Model']
            elif 'Model' in self.performance_results.columns and 'Accuracy' in self.performance_results.columns:
                sorted_results = self.performance_results.sort_values('Accuracy', ascending=False)
                best_model_name = sorted_results.iloc[0]['Model']
            else:
                print("Warning: No model name or performance metrics in results")
                return
                
            # Check if model exists
            if not hasattr(self, 'models'):
                self.models = {}
                print("Warning: Model dictionary doesn't exist, creating empty dictionary")
                
            best_model = self.models.get(best_model_name)
            if best_model is None:
                print(f"Warning: Cannot find best model '{best_model_name}'")
                return
                
            # 3. Generate feature importance chart (if model supports it)
            try:
                if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
                    importance_path = os.path.join(output_dir, 'feature_importance.png')
                    self.plot_feature_importance(best_model_name, top_n=15, save_path=importance_path)
            except Exception as e:
                print(f"Error generating feature importance chart: {str(e)}")
                
            # 4. Generate HTML report
            try:
                self._generate_html_report(output_dir)
            except Exception as e:
                print(f"Error generating HTML report: {str(e)}")
                
        except Exception as e:
            print(f"Error generating comprehensive report: {str(e)}")
            
        print(f"Model comprehensive evaluation report generated at: {output_dir}")
        
    def _generate_html_report(self, output_dir):
        report_path = os.path.join(output_dir, 'model_report.html')
        
        try:
            if 'Model' in self.performance_results.columns and 'AUC' in self.performance_results.columns:
                sorted_results = self.performance_results.sort_values('AUC', ascending=False)
                best_model_name = sorted_results.iloc[0]['Model']
                best_metrics = sorted_results.iloc[0].to_dict()
            elif 'Model' in self.performance_results.columns and 'Accuracy' in self.performance_results.columns:
                sorted_results = self.performance_results.sort_values('Accuracy', ascending=False)
                best_model_name = sorted_results.iloc[0]['Model']
                best_metrics = sorted_results.iloc[0].to_dict()
            else:
                print("Warning: No model name or performance metrics in results")
                best_model_name = "Unknown"
                best_metrics = {}
                
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Anesthesia Delirium Prediction Model Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; 
                                    margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                    .container {{ display: flex; flex-wrap: wrap; }}
                    .image-container {{ margin: 10px; max-width: 45%; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>Anesthesia Delirium Prediction Model Evaluation Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Best Model: {best_model_name}</h2>
                
                <div style="display: flex; flex-wrap: wrap;">
            """
            
            # Add performance metric cards
            metrics_to_show = ['AUC', 'Accuracy', 'F1_Score', 'Precision', 'Recall']
            for metric in metrics_to_show:
                if metric in best_metrics:
                    value = best_metrics[metric]
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    html_content += f"""
                    <div class="metric-card" style="flex: 1; min-width: 150px; margin: 10px;">
                        <div>{metric}</div>
                        <div class="metric-value">{value_str}</div>
                    </div>
                    """
            
            html_content += """
                </div>
                
                <h2>Model Performance Comparison</h2>
            """
            
            # Add performance comparison table
            if not self.performance_results.empty:
                html_content += """
                <table>
                    <tr>
                """
                
                # Table header
                for col in self.performance_results.columns:
                    html_content += f"<th>{col}</th>"
                
                html_content += """
                    </tr>
                """
                
                # Table content
                for _, row in self.performance_results.iterrows():
                    html_content += "<tr>"
                    for col in self.performance_results.columns:
                        value = row[col]
                        if isinstance(value, (int, float)) and col != 'Model':
                            value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        html_content += f"<td>{value_str}</td>"
                    html_content += "</tr>"
                
                html_content += """
                </table>
                """
            
            # Add charts
            html_content += """
                <h2>Visualization Results</h2>
                <div class="container">
            """
            
            # Check if chart files exist
            image_files = [
                ('model_performance_comparison.png', 'Model Performance Comparison'),
                ('feature_importance.png', 'Feature Importance'),
                ('confusion_matrix.png', 'Confusion Matrix'),
                ('roc_curve.png', 'ROC Curve')
            ]
            
            for filename, title in image_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    html_content += f"""
                    <div class="image-container">
                        <h3>{title}</h3>
                        <img src="{filename}" alt="{title}">
                    </div>
                    """
            
            html_content += """
                </div>
                
                <h2>Conclusions and Recommendations</h2>
                <p>
                    Based on the analysis results, the {best_model_name} model performs best in the anesthesia delirium prediction task,
                    and it is recommended to use this model in clinical applications. The model's AUC and accuracy metrics indicate good discriminative ability,
                    but practical application still requires integration with clinical expertise for comprehensive judgment.
                </p>
                
            </body>
            </html>
            """
            
            # Write HTML file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            print(f"HTML report generated: {report_path}")
            
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
            import traceback
            traceback.print_exc()

def build_full_delirium_model(delirium_df, non_delirium_df, significant_features, output_dir=None):
    print("\n===== Building Anesthesia Delirium Prediction Model =====")
    
    # Check dataset
    print(f"Dataset class distribution: delirium {len(delirium_df)} cases, non-delirium {len(non_delirium_df)} cases")
    ratio = len(delirium_df) / len(non_delirium_df) if len(non_delirium_df) > 0 else float('inf')
    print(f"Delirium:Non-delirium ratio = {ratio:.2f}:1")
    
    # Create model builder
    model_builder = ModelBuilder(random_state=42)
    
    # Handle missing values
    print("Handling missing values...")
    delirium_df = delirium_df.fillna(0)
    non_delirium_df = non_delirium_df.fillna(0)
    print("Checking feature correlations...")
    all_data = pd.concat([delirium_df, non_delirium_df], axis=0)
    print("Filtering non-numeric columns...")
    numeric_columns = all_data.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) < all_data.shape[1]:
        print(f"Note: Filtered {len(numeric_columns)} numeric columns from {all_data.shape[1]} columns for correlation analysis")
        non_numeric = set(all_data.columns) - set(numeric_columns)
        if len(non_numeric) <= 10:
            print(f"Non-numeric columns: {', '.join(non_numeric)}")
        else:
            print(f"Number of non-numeric columns: {len(non_numeric)}")
    print("Checking and removing redundant features with correlation > 0.95...")
    if len(numeric_columns) > 1: 
        try:
            corr_matrix = all_data[numeric_columns].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = []
            for col in upper_tri.columns:
                high_corr = upper_tri[col][upper_tri[col] > 0.95].index.tolist()
                for feat in high_corr:
                    high_corr_pairs.append((col, feat, corr_matrix.loc[col, feat]))
            
            if high_corr_pairs:
                print(f"Found {len(high_corr_pairs)} pairs of highly correlated features:")
                for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:10]):
                    print(f"  {i+1}. {feat1} and {feat2}: correlation = {corr:.3f}")
                if len(high_corr_pairs) > 10:
                    print(f"  ... and {len(high_corr_pairs) - 10} other highly correlated feature pairs")
                to_drop = set()
                for feat1, feat2, _ in high_corr_pairs:
                    to_drop.add(feat2)
                
                print(f"Removing {len(to_drop)} redundant features...")
                delirium_df = delirium_df.drop(columns=list(to_drop), errors='ignore')
                non_delirium_df = non_delirium_df.drop(columns=list(to_drop), errors='ignore')
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            print("Skipping correlation analysis step")
    else:
        print("Insufficient numeric features, skipping correlation analysis")

    try:
        X_train, X_test, y_train, y_test, feature_names = model_builder.prepare_data(
            delirium_df,
            non_delirium_df,
            significant_features=significant_features,
            test_size=0.25,
            apply_pca=False
        )
        
        print(f"Data preparation complete: {len(X_train)} training samples, {len(X_test)} test samples, {X_train.shape[1]} features\n")
        
        print("Building ensemble models...")
        
        model_builder.build_models()

        print("Evaluating 9 models using 3-fold cross-validation...")
        
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class distribution: {class_dist}")
        
        if len(unique) > 1:
            majority = max(counts)
            minority = min(counts)
            imbalance_ratio = majority / minority
            print(f"Imbalance ratio: {imbalance_ratio:.2f} : 1 (majority:minority)")
        
        model_builder.train_evaluate_all_models(X_train, X_test, y_train, y_test)
        
        try:
            if len(model_builder.models) >= 3:
                model_builder.build_ensemble_models()
            else:
                print("Insufficient models available, skipping ensemble model building")
        except Exception as e:
            print(f"Ensemble model building failed: {str(e)}")
            traceback.print_exc()
        
        try:
            if hasattr(model_builder, 'ensemble_models') and model_builder.ensemble_models:
                model_builder.evaluate_ensemble_models(X_test, y_test)
            else:
                print("No ensemble models available for evaluation")
        except Exception as e:
            print(f"Ensemble model evaluation failed: {str(e)}")
            traceback.print_exc()
        
        if not hasattr(model_builder, 'performance_results') or not isinstance(model_builder.performance_results, pd.DataFrame) or model_builder.performance_results.empty:
            print("Warning: Model evaluation results are empty, creating default performance results")
            model_builder.performance_results = pd.DataFrame({
                'Model': ['RandomForest'],
                'AUC': [0.5],
                'Accuracy': [0.5],
                'Precision': [0.5],
                'Recall': [0.5],
                'F1_Score': [0.5]
            })
            
        try:
            if 'AUC' in model_builder.performance_results.columns:
                sorted_models = model_builder.performance_results.sort_values('AUC', ascending=False)
                print("\n=== Model Performance Ranking ===")
                print(sorted_models[['Model', 'AUC', 'Accuracy', 'F1_Score']].head())
                
                if len(sorted_models) > 0:
                    best_model_name = sorted_models.iloc[0]['Model']
                    best_auc = sorted_models.iloc[0]['AUC']
                    best_accuracy = sorted_models.iloc[0]['Accuracy']
                    best_precision = sorted_models.iloc[0]['Precision'] if 'Precision' in sorted_models.columns else 0.5
                    best_recall = sorted_models.iloc[0]['Recall'] if 'Recall' in sorted_models.columns else 0.5
                    best_f1 = sorted_models.iloc[0]['F1_Score'] if 'F1_Score' in sorted_models.columns else 0.5
                else:
                    best_model_name = 'RandomForest'
                    best_auc = 0.5
                    best_accuracy = 0.5
                    best_precision = 0.5
                    best_recall = 0.5
                    best_f1 = 0.5
            else:
                sorted_models = model_builder.performance_results.sort_values('Accuracy', ascending=False)
                print("\n=== Model Performance Ranking ===")
                print(sorted_models[['Model', 'Accuracy']].head())

                if len(sorted_models) > 0:
                    best_model_name = sorted_models.iloc[0]['Model']
                    best_auc = 0.5
                    best_accuracy = sorted_models.iloc[0]['Accuracy']
                    best_precision = sorted_models.iloc[0]['Precision'] if 'Precision' in sorted_models.columns else 0.5
                    best_recall = sorted_models.iloc[0]['Recall'] if 'Recall' in sorted_models.columns else 0.5
                    best_f1 = sorted_models.iloc[0]['F1_Score'] if 'F1_Score' in sorted_models.columns else 0.5
                else:

                    best_model_name = 'RandomForest'
                    best_auc = 0.5
                    best_accuracy = 0.5
                    best_precision = 0.5
                    best_recall = 0.5
                    best_f1 = 0.5
        except Exception as e:
            print(f"Error getting best model: {str(e)}")
            traceback.print_exc()
            best_model_name = 'RandomForest'  
            best_auc = 0.5
            best_accuracy = 0.5
            best_precision = 0.5
            best_recall = 0.5
            best_f1 = 0.5

        if output_dir is not None:
            try:
                os.makedirs(output_dir, exist_ok=True)
                if isinstance(model_builder.performance_results, dict):
                    try:
                        results_list = []
                        for model_name, metrics in model_builder.performance_results.items():
                            metrics['Model'] = model_name
                            results_list.append(metrics)
                        model_builder.performance_results = pd.DataFrame(results_list)
                        # If empty, create a basic DataFrame
                        if len(model_builder.performance_results) == 0:
                            model_builder.performance_results = pd.DataFrame({
                                'Model': [best_model_name],
                                'AUC': [best_auc],
                                'Accuracy': [best_accuracy],
                                'Precision': [best_precision],
                                'Recall': [best_recall],
                                'F1_Score': [best_f1]
                            })
                    except Exception as e:
                        print(f"Error converting performance results: {str(e)}")
                        traceback.print_exc()
                        model_builder.performance_results = pd.DataFrame({
                            'Model': [best_model_name],
                            'AUC': [best_auc],
                            'Accuracy': [best_accuracy],
                            'Precision': [best_precision],
                            'Recall': [best_recall],
                            'F1_Score': [best_f1]
                        })
                
                results_file = os.path.join(output_dir, "model_performance.csv")
                try:
                    if hasattr(model_builder, 'performance_results') and isinstance(model_builder.performance_results, pd.DataFrame):
                        model_builder.performance_results.to_csv(results_file, index=False)
                        print(f"Model performance results saved to: {results_file}")
                except Exception as e:
                    print(f"Error saving performance results: {str(e)}")
                    traceback.print_exc()
                best_model_path = os.path.join(output_dir, f"{best_model_name}_delirium_model.pkl")
                try:
                    model_builder.save_model(best_model_name, best_model_path)
                except Exception as e:
                    print(f"Error saving model and results: {str(e)}")
                    traceback.print_exc()
                
                summary_file = os.path.join(output_dir, "model_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write("----- Model Building Summary -----\n")
                    f.write(f"Best Model: {best_model_name}\n")
                    f.write(f"AUC: {best_auc:.4f}\n")
                    f.write(f"Accuracy: {best_accuracy:.4f}\n")
                    f.write(f"Precision: {best_precision:.4f}\n")
                    f.write(f"Recall: {best_recall:.4f}\n")
                    f.write(f"F1 Score: {best_f1:.4f}\n")
                
                print(f"Model summary saved to: {summary_file}")
                
                print("\nGenerating comprehensive model evaluation report...")
                try:
                    print(f"Generating comprehensive model evaluation report, output to: {output_dir}")
                    model_builder.generate_comprehensive_report(output_dir)
                except Exception as e:
                    print(f"Error generating comprehensive report: {str(e)}")
                    traceback.print_exc()
            except Exception as e:
                print(f"Error saving model and results: {str(e)}")
                traceback.print_exc()
        
        return model_builder
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        traceback.print_exc()
        return None