#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECG Anesthesia Delirium Prediction System - Streamlit Web Application

This application provides a user-friendly web interface for:
1. Uploading ECG data files
2. Batch processing ECG data  
3. Predicting anesthesia delirium risk
4. Visualizing analysis results
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import io
import json
import pickle
import zipfile
import tempfile
import shutil
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import warnings

warnings.filterwarnings("ignore")
# 忽略XGBoost的GPU警告
warnings.filterwarnings("ignore", message=".*tree method.*deprecated.*")
warnings.filterwarnings("ignore", message=".*mismatched devices.*")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="ECG Anesthesia Delirium Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #06A77D;
        --warning-color: #F77F00;
        --danger-color: #D62828;
        --bg-color: #F8F9FA;
        --card-bg: #FFFFFF;
    }
    
    /* Main container */
    .main {
        background-color: var(--bg-color);
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .custom-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid var(--primary-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .risk-high {
        border-left-color: var(--danger-color);
    }
    
    .risk-medium {
        border-left-color: var(--warning-color);
    }
    
    .risk-low {
        border-left-color: var(--success-color);
    }
    
    /* Probability display */
    .prob-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .prob-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .prob-label {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .prob-value-high {
        color: #D62828;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .prob-value-low {
        color: #06A77D;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2E86AB 0%, #1a4d6d 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2E86AB 0%, #1a4d6d 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* File uploader */
    .stFileUploader {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed var(--primary-color);
        transition: all 0.3s;
    }
    
    .stFileUploader:hover {
        border-color: var(--secondary-color);
        background: #f8f9fa;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        border: 2px solid #e0e0e0;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white !important;
        border-color: transparent;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* SHAP plot container */
    .shap-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Add project path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import custom modules
try:
    from ecg_modules.preprocessing import SignalPreprocessor
    from ecg_modules.feature_extraction import FeatureExtractor
    from ecg_modules.utils import find_ecg_files, detect_signal_quality
    from predict_delirium import DeliriumPredictor
except ImportError as e:
    st.error(f"❌ Module import failed: {str(e)}")
    st.error("Please ensure all dependency modules are properly installed")

class ECGDeliriumApp:
    """ECG Anesthesia Delirium Prediction Application Class"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessor = None
        self.feature_extractor = None
        self.predictor = None
        self.sampling_rate = 1000  # Default sampling rate
        
    def __del__(self):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def initialize_components(self):
        """Initialize processing components"""
        try:
            self.preprocessor = SignalPreprocessor(sampling_rate=self.sampling_rate)
            self.feature_extractor = FeatureExtractor()
            
            # Find and load the best model
            model_paths = self._find_model_files()
            if model_paths:
                self.predictor = DeliriumPredictor(
                    model_path=model_paths.get('model'),
                    meta_path=model_paths.get('metadata')
                )
                if self.predictor.load_model():
                    st.success("✅ Model loaded successfully")
                    
                    # Load training features (336) and selected features (20)
                    training_feature_file = model_paths.get('training_features')
                    selected_feature_file = model_paths.get('selected_features')
                    
                    if training_feature_file and os.path.exists(training_feature_file):
                        try:
                            train_df = pd.read_csv(training_feature_file)
                            self.predictor.training_features = train_df['feature'].tolist()
                            st.success(f"✅ Loaded {len(self.predictor.training_features)} training features")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to load training features: {str(e)}")
                            self.predictor.training_features = None
                    else:
                        st.warning("⚠️ Training features file not found")
                        self.predictor.training_features = None
                    
                    if selected_feature_file and os.path.exists(selected_feature_file):
                        try:
                            sel_df = pd.read_csv(selected_feature_file)
                            self.predictor.feature_list = sel_df['feature'].tolist()
                            st.info(f"📊 Model uses {len(self.predictor.feature_list)} selected features (EPV≥5)")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to load selected features: {str(e)}")
                    else:
                        st.warning("⚠️ Selected features file not found")
                    
                    return True
                else:
                    st.error("❌ Model loading failed")
                    return False
            else:
                st.warning("⚠️ No trained model files found")
                return False
        except Exception as e:
            st.error(f"❌ Component initialization failed: {str(e)}")
            return False
    
    def _find_model_files(self):
        """Find model files"""
        model_dirs = [
            'delirium_model_results/models',
            'models',
            'output/models'
        ]
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                # 查找模型文件（优先选择兼容的模型）
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f]
                if model_files:
                    # 优先选择已验证可用的模型（避免版本兼容问题）
                    preferred_models = ['XGBoost', 'LogisticRegression', 'SVM', 'RandomForest']
                    selected_model = None
                    
                    for pref in preferred_models:
                        for model_file in model_files:
                            if pref in model_file and 'metadata' not in model_file:
                                selected_model = model_file
                                break
                        if selected_model:
                            break
                    
                    if not selected_model:
                        selected_model = model_files[0]
                    
                    model_path = os.path.join(model_dir, selected_model)
                    
                    # 查找元数据文件
                    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
                    if not os.path.exists(metadata_path):
                        metadata_path = None
                    
                    # 查找训练特征文件（Pipeline需要完整的336个特征）
                    training_feature_file = None
                    selected_feature_file = None
                    
                    # 训练时使用的完整特征列表（336个）
                    training_paths = [
                        os.path.join(model_dir, 'training_features_336.csv'),
                        'model_training_features_336.csv'
                    ]
                    
                    for path in training_paths:
                        if os.path.exists(path):
                            training_feature_file = path
                            break
                    
                    # 模型最终选择的特征（20个，仅用于显示）
                    selected_paths = [
                        os.path.join(model_dir, 'selected_features.csv'),
                        'delirium_model_results/feature_selection_results/selected_features.csv'
                    ]
                    
                    for path in selected_paths:
                        if os.path.exists(path):
                            selected_feature_file = path
                            break
                    
                    return {
                        'model': model_path,
                        'metadata': metadata_path,
                        'training_features': training_feature_file,
                        'selected_features': selected_feature_file
                    }
        
        return None
    
    def validate_ecg_file(self, df):
        """Validate ECG file format"""
        required_leads = [
            'MDC_ECG_LEAD_I', 'MDC_ECG_LEAD_II', 'MDC_ECG_LEAD_III',
            'MDC_ECG_LEAD_aVR', 'MDC_ECG_LEAD_aVL', 'MDC_ECG_LEAD_aVF',
            'MDC_ECG_LEAD_V1', 'MDC_ECG_LEAD_V2', 'MDC_ECG_LEAD_V3',
            'MDC_ECG_LEAD_V4', 'MDC_ECG_LEAD_V5', 'MDC_ECG_LEAD_V6'
        ]
        
        # Check for required leads
        missing_leads = [lead for lead in required_leads if lead not in df.columns]
        
        if missing_leads:
            return False, f"Missing required leads: {', '.join(missing_leads)}"
        
        # Check data length
        if len(df) < 100:
            return False, "Data too short, at least 100 samples required"
        
        # Check data type
        for lead in required_leads:
            if not pd.api.types.is_numeric_dtype(df[lead]):
                try:
                    df[lead] = pd.to_numeric(df[lead], errors='coerce')
                except:
                    return False, f"Lead {lead} has incorrect data format"
        
        return True, "File format validation passed"
    
    def extract_features_from_file(self, df, filename):
        """Extract features from ECG file using the same method as training"""
        try:
            # 准备数据：将完整导联名转换为简化导联名
            leads_mapping = {
                'MDC_ECG_LEAD_I': 'I',
                'MDC_ECG_LEAD_II': 'II',
                'MDC_ECG_LEAD_III': 'III',
                'MDC_ECG_LEAD_aVR': 'aVR',
                'MDC_ECG_LEAD_aVL': 'aVL',
                'MDC_ECG_LEAD_aVF': 'aVF',
                'MDC_ECG_LEAD_V1': 'V1',
                'MDC_ECG_LEAD_V2': 'V2',
                'MDC_ECG_LEAD_V3': 'V3',
                'MDC_ECG_LEAD_V4': 'V4',
                'MDC_ECG_LEAD_V5': 'V5',
                'MDC_ECG_LEAD_V6': 'V6'
            }
            
            # 创建简化导联名的数据字典
            data_dict = {}
            for full_lead, short_lead in leads_mapping.items():
                if full_lead in df.columns:
                    signal_data = df[full_lead].values
                    
                    # 预处理信号（可选）
                    if self.preprocessor:
                        try:
                            signal_data = self.preprocessor.filter_signal(signal_data)
                            signal_data = self.preprocessor.remove_baseline_wander(signal_data)
                        except Exception as e:
                            st.warning(f"预处理导联 {short_lead} 时出错: {str(e)}")
                    
                    data_dict[short_lead] = signal_data
            
            if not data_dict:
                st.error("没有找到有效的导联数据")
                return None
            
            # 使用与训练时相同的方法提取特征
            if self.feature_extractor:
                # 调用 extract_features_from_leads 方法
                # 这会自动检测 R 峰并提取所有类型的特征（时域、频域、HRV、形态学等）
                all_features, r_peaks_dict = self.feature_extractor.extract_features_from_leads(
                    data_dict, 
                    leads=list(data_dict.keys())
                )
                
                # 添加文件名
                all_features['file'] = filename
                
                return all_features
            else:
                st.error("特征提取器未初始化")
                return None
            
        except Exception as e:
            st.error(f"特征提取失败: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def predict_delirium_risk(self, features):
        """Predict delirium risk"""
        try:
            if not self.predictor or not self.predictor.model:
                return None, "Model not loaded"
            
            # 检查是否有训练特征列表
            if not hasattr(self.predictor, 'training_features') or not self.predictor.training_features:
                return None, "Training features list not loaded. Please reinitialize the system."
            
            # Prepare feature data
            feature_df = pd.DataFrame([features])
            
            # Remove non-numeric columns
            non_numeric_cols = ['file']
            feature_df = feature_df.drop(columns=non_numeric_cols, errors='ignore')
            
            # 获取训练时使用的336个特征
            training_features = self.predictor.training_features
            
            # 创建新的DataFrame，只包含训练时使用的特征
            aligned_df = pd.DataFrame(columns=training_features)
            
            # 填充可用的特征
            for feat in training_features:
                if feat in feature_df.columns:
                    aligned_df[feat] = feature_df[feat]
                else:
                    aligned_df[feat] = 0  # 填充缺失特征为0
            
            # 确保数据类型正确
            aligned_df = aligned_df.astype(float)
            
            # Make prediction - Pipeline会自动处理特征选择
            if hasattr(self.predictor.model, 'predict_proba'):
                probabilities = self.predictor.model.predict_proba(aligned_df)
                risk_score = probabilities[0][1]  # Positive class probability
            else:
                prediction = self.predictor.model.predict(aligned_df)
                risk_score = float(prediction[0])
            
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
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'prediction': risk_score >= 0.5,
                'feature_df': aligned_df  # Include for SHAP analysis
            }, None
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            st.error(f"Detailed error:\n{error_detail}")
            return None, f"Prediction failed: {str(e)}"
    
    def generate_shap_analysis(self, feature_df, filename):
        """Generate SHAP beeswarm summary plot"""
        if not SHAP_AVAILABLE:
            return None, "SHAP library not available"
        
        try:
            # Get the final classifier from pipeline
            if hasattr(self.predictor.model, 'named_steps'):
                classifier = self.predictor.model.named_steps['clf']
            else:
                classifier = self.predictor.model
            
            # Check if it's a tree-based model
            model_name = type(classifier).__name__
            if 'Forest' not in model_name and 'Tree' not in model_name and 'XGB' not in model_name and 'Boost' not in model_name:
                return None, f"SHAP analysis only supports tree-based models (current: {model_name})"
            
            # For pipeline, we need to transform the data first
            if hasattr(self.predictor.model, 'named_steps'):
                # Get all pipeline steps except the final classifier
                steps = list(self.predictor.model.named_steps.keys())
                if 'clf' in steps:
                    steps.remove('clf')
                
                # Build a pipeline with all preprocessing steps
                from sklearn.pipeline import Pipeline
                preprocessing_steps = [(name, self.predictor.model.named_steps[name]) for name in steps]
                preprocessing_pipeline = Pipeline(preprocessing_steps) if preprocessing_steps else None
                
                if preprocessing_pipeline:
                    # Transform the data through all preprocessing steps
                    transformed_data = preprocessing_pipeline.transform(feature_df)
                    
                    # Get selected feature names
                    # Try to get from predictor's feature_list (the 20 selected features)
                    if hasattr(self.predictor, 'feature_list') and self.predictor.feature_list:
                        selected_features = self.predictor.feature_list
                    elif 'selector' in self.predictor.model.named_steps:
                        selector = self.predictor.model.named_steps['selector']
                        if hasattr(selector, 'selected_feature_names_'):
                            selected_features = selector.selected_feature_names_
                        else:
                            # Use generic names
                            selected_features = [f"Feature_{i}" for i in range(transformed_data.shape[1])]
                    else:
                        selected_features = [f"Feature_{i}" for i in range(transformed_data.shape[1])]
                    
                    # Create DataFrame with correct column names
                    transformed_df = pd.DataFrame(transformed_data, columns=selected_features)
                else:
                    transformed_df = feature_df
            else:
                transformed_df = feature_df
            
            # Create SHAP explainer for the classifier
            explainer = shap.TreeExplainer(classifier)
            
            # Calculate SHAP values
            shap_values_raw = explainer.shap_values(transformed_df)
            
            # Handle different SHAP value formats
            if isinstance(shap_values_raw, list):
                shap_values = shap_values_raw[1]  # For binary classification (positive class)
            elif len(shap_values_raw.shape) == 3:
                shap_values = shap_values_raw[:, :, 1]
            else:
                shap_values = shap_values_raw
            
            # Create beeswarm plot
            fig = self._create_shap_beeswarm_plot(shap_values, transformed_df, filename)
            
            return {
                'plot': fig,
                'shap_values': shap_values,
                'feature_data': transformed_df
            }, None
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return None, f"SHAP analysis failed: {str(e)}\n{error_detail}"
    
    def _format_feature_name(self, feature_name):
        """Format feature name for display"""
        # Simplify HRV feature names
        if '_hrv_HRV_' in feature_name:
            parts = feature_name.split('_')
            return f"{parts[0]} HRV {parts[-1]}"
        
        # Simplify R-peak delay names
        if 'r_peak_delay_' in feature_name:
            parts = feature_name.replace('r_peak_delay_', '').split('_')
            leads = '-'.join(parts[:2])
            stat = 'SD' if parts[-1] == 'std' else 'Mean' if parts[-1] == 'mean' else parts[-1].upper()
            return f"{leads} RDelay {stat}"
        
        # General simplification
        feature_name = feature_name.replace('MDC_ECG_LEAD_', '')
        feature_name = feature_name.replace('_', ' ')
        return feature_name
    
    def _create_shap_beeswarm_plot(self, shap_values, feature_data, filename):
        """Create SHAP waterfall plot for single prediction"""
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get feature names and format them
        feature_names = [self._format_feature_name(f) for f in feature_data.columns]
        
        # For single prediction
        if shap_values.ndim == 1:
            shap_vals_single = shap_values
        else:
            shap_vals_single = shap_values[0]
        
        # Get top 20 features by absolute SHAP value
        abs_shap = np.abs(shap_vals_single)
        sorted_idx = np.argsort(abs_shap)[::-1][:20]
        
        top_shap = shap_vals_single[sorted_idx]
        top_features = [feature_names[i] for i in sorted_idx]
        top_values = feature_data.iloc[0, sorted_idx].values
        
        # Sort by SHAP value for waterfall effect
        sort_order = np.argsort(top_shap)
        top_shap = top_shap[sort_order]
        top_features = [top_features[i] for i in sort_order]
        top_values = top_values[sort_order]
        
        # Create waterfall plot
        y_pos = np.arange(len(top_features))
        
        # Color based on positive/negative SHAP values
        colors = ['#E74C3C' if x > 0 else '#3498DB' for x in top_shap]
        
        # Plot horizontal bars
        bars = ax.barh(y_pos, top_shap, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels with feature values
        for i, (shap_val, feat_val) in enumerate(zip(top_shap, top_values)):
            # SHAP value label
            label_x = shap_val + (0.002 if shap_val > 0 else -0.002)
            ha = 'left' if shap_val > 0 else 'right'
            ax.text(label_x, i, f'{shap_val:.4f}', 
                   va='center', ha=ha, fontsize=9, fontweight='bold')
            
            # Feature value label (in parentheses)
            ax.text(0, i, f'  (val={feat_val:.2f})', 
                   va='center', ha='left' if shap_val < 0 else 'right',
                   fontsize=8, style='italic', alpha=0.7)
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11)
        ax.set_xlabel('SHAP Value (impact on model output)', fontsize=13, fontweight='bold')
        ax.set_title('SHAP Waterfall Plot: Feature Contributions to Prediction', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', alpha=0.8, edgecolor='black', label='Increases Risk'),
            Patch(facecolor='#3498DB', alpha=0.8, edgecolor='black', label='Decreases Risk')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
        
        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        
        # Set x-axis limits with some padding
        max_abs = np.max(np.abs(top_shap))
        ax.set_xlim(-max_abs * 1.2, max_abs * 1.2)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_ecg_signals(self, df, filename):
        """Plot ECG signals"""
        leads = [
            'MDC_ECG_LEAD_I', 'MDC_ECG_LEAD_II', 'MDC_ECG_LEAD_III',
            'MDC_ECG_LEAD_aVR', 'MDC_ECG_LEAD_aVL', 'MDC_ECG_LEAD_aVF',
            'MDC_ECG_LEAD_V1', 'MDC_ECG_LEAD_V2', 'MDC_ECG_LEAD_V3',
            'MDC_ECG_LEAD_V4', 'MDC_ECG_LEAD_V5', 'MDC_ECG_LEAD_V6'
        ]
        
        # Simplified lead names for display
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Create subplots with better spacing
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=lead_names,
            vertical_spacing=0.08,  # Increased spacing
            horizontal_spacing=0.08
        )
        
        # Calculate time axis (assuming sampling rate)
        time_axis = np.arange(len(df)) / self.sampling_rate
        
        # Show only first 10 seconds
        max_samples = min(len(df), 10 * self.sampling_rate)
        time_axis = time_axis[:max_samples]
        
        # Add traces for each lead
        for i, (lead, lead_name) in enumerate(zip(leads, lead_names)):
            if lead in df.columns:
                row = i // 3 + 1
                col = i % 3 + 1
                
                signal_data = df[lead].values[:max_samples]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=signal_data,
                        mode='lines',
                        name=lead_name,
                        line=dict(width=1.2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title={
                'text': f"ECG Signals - {filename}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=1000,  # Increased height
            showlegend=False,
            margin=dict(t=80, b=40, l=60, r=40)  # Better margins
        )
        
        # Update axes with better formatting
        fig.update_xaxes(title_text="Time (s)", title_font=dict(size=10))
        fig.update_yaxes(title_text="Amplitude (μV)", title_font=dict(size=10))
        
        # Update subplot titles font size
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=11)
        
        return fig

def main():
    """Main function"""
    # Load custom CSS
    load_custom_css()
    
    # Custom header
    st.markdown("""
    <div class="custom-header">
        <h1>🏥 ECG Anesthesia Delirium Prediction System</h1>
        <p>Advanced AI-powered delirium risk assessment from ECG signals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create app instance
    if 'app' not in st.session_state:
        st.session_state.app = ECGDeliriumApp()
    
    app = st.session_state.app
    
    # Sidebar with custom styling
    st.sidebar.markdown("### ⚙️ System Settings")
    
    # Show warning if SHAP not available
    if not SHAP_AVAILABLE:
        st.sidebar.warning("⚠️ SHAP not installed. Install with: `pip install shap`")
    
    # Initialize button with custom styling
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Initialize System", use_container_width=True):
        with st.spinner("🔄 Initializing system components..."):
            if app.initialize_components():
                st.sidebar.markdown('<div class="status-badge status-success">✅ System Ready</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<div class="status-badge status-danger">❌ Initialization Failed</div>', unsafe_allow_html=True)
    
    # Sampling rate setting
    app.sampling_rate = st.sidebar.selectbox(
        "Sampling Rate (Hz)",
        [500, 1000, 2000],
        index=1
    )
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Single File", "📁 Batch Process", "📊 Analysis", "ℹ️ System Info"])
    
    with tab1:
        st.header("Single File Delirium Risk Prediction")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Select ECG CSV File",
            type=['csv'],
            help="Please upload a CSV file containing 12-lead ECG data"
        )
        
        if uploaded_file is not None:
            try:
                # Read file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
                st.info(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Display file preview
                with st.expander("View Data Preview"):
                    st.dataframe(df.head())
                
                # Validate file format
                is_valid, message = app.validate_ecg_file(df)
                
                if is_valid:
                    # Display ECG signals
                    with st.expander("View ECG Signals", expanded=True):
                        fig = app.plot_ecg_signals(df, uploaded_file.name)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction button
                    if st.button("🔮 Start Prediction", type="primary"):
                        if app.predictor is None:
                            st.error("❌ Please initialize system first")
                        else:
                            with st.spinner("Processing and predicting..."):
                                # Extract features
                                features = app.extract_features_from_file(df, uploaded_file.name)
                                
                                if features:
                                    # Make prediction
                                    result, error = app.predict_delirium_risk(features)
                                    
                                    if result:
                                        # Display prediction results with custom cards
                                        st.markdown("### 📊 Prediction Results")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        risk_class = "risk-high" if result['risk_score'] >= 0.7 else "risk-medium" if result['risk_score'] >= 0.4 else "risk-low"
                                        
                                        with col1:
                                            st.markdown(f"""
                                            <div class="metric-card {risk_class}">
                                                <h4 style="color: #666; margin: 0; font-size: 0.9rem;">Risk Score</h4>
                                                <h2 style="margin: 0.5rem 0; color: {result['risk_color']};">{result['risk_score']:.3f}</h2>
                                                <p style="color: #999; margin: 0; font-size: 0.85rem;">Probability: {result['risk_score']:.1%}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        with col2:
                                            st.markdown(f"""
                                            <div class="metric-card {risk_class}">
                                                <h4 style="color: #666; margin: 0; font-size: 0.9rem;">Risk Level</h4>
                                                <h2 style="margin: 0.5rem 0; color: {result['risk_color']};">{result['risk_level']}</h2>
                                                <p style="color: #999; margin: 0; font-size: 0.85rem;">Assessment</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        with col3:
                                            prediction_text = "Delirium" if result['prediction'] else "Non-Delirium"
                                            prediction_color = "#D62828" if result['prediction'] else "#06A77D"
                                            st.markdown(f"""
                                            <div class="metric-card {risk_class}">
                                                <h4 style="color: #666; margin: 0; font-size: 0.9rem;">Classification</h4>
                                                <h2 style="margin: 0.5rem 0; color: {prediction_color};">{prediction_text}</h2>
                                                <p style="color: #999; margin: 0; font-size: 0.85rem;">Binary Outcome</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        # Risk visualization
                                        st.subheader("Risk Assessment Visualization")
                                        
                                        # Create gauge chart
                                        fig_gauge = go.Figure(go.Indicator(
                                            mode = "gauge+number+delta",
                                            value = result['risk_score'],
                                            domain = {'x': [0, 1], 'y': [0, 1]},
                                            title = {'text': "Delirium Risk Score"},
                                            delta = {'reference': 0.5},
                                            gauge = {
                                                'axis': {'range': [None, 1]},
                                                'bar': {'color': result['risk_color']},
                                                'steps': [
                                                    {'range': [0, 0.4], 'color': "lightgreen"},
                                                    {'range': [0.4, 0.7], 'color': "orange"},
                                                    {'range': [0.7, 1], 'color': "red"}
                                                ],
                                                'threshold': {
                                                    'line': {'color': "black", 'width': 4},
                                                    'thickness': 0.75,
                                                    'value': 0.5
                                                }
                                            }
                                        ))
                                        
                                        fig_gauge.update_layout(height=400)
                                        st.plotly_chart(fig_gauge, use_container_width=True)
                                        
                                        # Display probability breakdown with modern design
                                        st.markdown("### 📈 Probability Breakdown")
                                        prob_delirium = result['risk_score']
                                        prob_non_delirium = 1 - prob_delirium
                                        
                                        st.markdown(f"""
                                        <div class="prob-container">
                                            <div class="prob-item">
                                                <span class="prob-label">🔴 Delirium Risk</span>
                                                <span class="prob-value-high">{prob_delirium:.1%}</span>
                                            </div>
                                            <div class="prob-item">
                                                <span class="prob-label">🟢 Non-Delirium</span>
                                                <span class="prob-value-low">{prob_non_delirium:.1%}</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # SHAP Analysis with beeswarm plot
                                        if SHAP_AVAILABLE and 'feature_df' in result:
                                            st.markdown("---")
                                            st.markdown("### 🔍 SHAP Feature Importance Analysis")
                                            
                                            with st.spinner("🔄 Generating SHAP analysis..."):
                                                shap_result, shap_error = app.generate_shap_analysis(
                                                    result['feature_df'], 
                                                    uploaded_file.name
                                                )
                                                
                                                if shap_result:
                                                    st.info("""
                                                    **SHAP Waterfall Plot** explains how each feature contributes to this prediction:
                                                    
                                                    - 🔴 **Red bars**: Features that INCREASE delirium risk
                                                    - 🔵 **Blue bars**: Features that DECREASE delirium risk
                                                    - **Bar length**: The magnitude of each feature's impact
                                                    - **Feature values**: Shown in parentheses (val=X.XX)
                                                    - Features are sorted by impact strength (bottom to top)
                                                    """)
                                                    
                                                    # Display SHAP waterfall plot
                                                    st.markdown('<div class="shap-container">', unsafe_allow_html=True)
                                                    st.pyplot(shap_result['plot'])
                                                    st.markdown('</div>', unsafe_allow_html=True)
                                                else:
                                                    st.info(f"ℹ️ SHAP analysis not available: {shap_error}")
                                        
                                        # Save results (convert numpy types to Python types)
                                        result_data = {
                                            'filename': uploaded_file.name,
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'risk_score': float(result['risk_score']),  # Convert numpy float to Python float
                                            'risk_level': result['risk_level'],
                                            'prediction': bool(result['prediction'])  # Convert numpy bool to Python bool
                                        }
                                        
                                        # Download button
                                        result_json = json.dumps(result_data, indent=2, ensure_ascii=False)
                                        st.download_button(
                                            label="📥 Download Prediction Results",
                                            data=result_json,
                                            file_name=f"prediction_result_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json"
                                        )
                                    
                                    else:
                                        st.error(f"❌ {error}")
                                else:
                                    st.error("❌ Feature extraction failed")
                else:
                    st.error(f"❌ {message}")
                    
            except Exception as e:
                st.error(f"❌ File processing failed: {str(e)}")
    
    with tab2:
        st.header("Batch Delirium Risk Prediction")
        
        # Batch file upload
        uploaded_files = st.file_uploader(
            "Select Multiple ECG CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            help="You can select multiple CSV files for batch prediction"
        )
        
        if uploaded_files:
            st.success(f"✅ Selected {len(uploaded_files)} files")
            
            # Display file list
            with st.expander("View Selected Files"):
                for file in uploaded_files:
                    st.write(f"📄 {file.name}")
            
            # Batch prediction button
            if st.button("🚀 Start Batch Prediction", type="primary"):
                if app.predictor is None:
                    st.error("❌ Please initialize system first")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    
                    batch_results = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                            
                            # Read file
                            df = pd.read_csv(uploaded_file)
                            
                            # Validate file format
                            is_valid, _ = app.validate_ecg_file(df)
                            
                            if is_valid:
                                # Extract features
                                features = app.extract_features_from_file(df, uploaded_file.name)
                                
                                if features:
                                    # Make prediction
                                    result, error = app.predict_delirium_risk(features)
                                    
                                    if result:
                                        batch_results.append({
                                            'filename': uploaded_file.name,
                                            'risk_score': float(result['risk_score']),  # Convert to Python float
                                            'risk_level': result['risk_level'],
                                            'prediction': bool(result['prediction']),  # Convert to Python bool
                                            'status': 'Success'
                                        })
                                    else:
                                        batch_results.append({
                                            'filename': uploaded_file.name,
                                            'risk_score': None,
                                            'risk_level': None,
                                            'prediction': None,
                                            'status': f'Prediction failed: {error}'
                                        })
                                else:
                                    batch_results.append({
                                        'filename': uploaded_file.name,
                                        'risk_score': None,
                                        'risk_level': None,
                                        'prediction': None,
                                        'status': 'Feature extraction failed'
                                    })
                            else:
                                batch_results.append({
                                    'filename': uploaded_file.name,
                                    'risk_score': None,
                                    'risk_level': None,
                                    'prediction': None,
                                    'status': 'Invalid file format'
                                })
                                
                        except Exception as e:
                            batch_results.append({
                                'filename': uploaded_file.name,
                                'risk_score': None,
                                'risk_level': None,
                                'prediction': None,
                                'status': f'Processing failed: {str(e)}'
                            })
                        
                        # Update progress bar
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Batch prediction completed!")
                    
                    # Display results
                    with results_container:
                        st.subheader("Batch Prediction Results")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(batch_results)
                        
                        # Statistics
                        successful_predictions = len(results_df[results_df['status'] == 'Success'])
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Files", len(uploaded_files))
                        with col2:
                            st.metric("Successful Predictions", successful_predictions)
                        with col3:
                            st.metric("Success Rate", f"{successful_predictions/len(uploaded_files)*100:.1f}%")
                        
                        # Display results table
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Statistics for successful predictions
                        if successful_predictions > 0:
                            successful_results = results_df[results_df['status'] == 'Success']
                            
                            # Risk level distribution
                            risk_counts = successful_results['risk_level'].value_counts()
                            fig_pie = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Level Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Risk score distribution
                            fig_hist = px.histogram(
                                successful_results,
                                x='risk_score',
                                title="Risk Score Distribution",
                                nbins=20
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Download batch results
                        results_json = results_df.to_json(orient='records', indent=2, force_ascii=False)
                        st.download_button(
                            label="📥 Download Batch Results",
                            data=results_json,
                            file_name=f"batch_prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
    
    with tab3:
        st.header("Results Analysis")
        st.info("This feature is used to analyze and visualize prediction result statistics")
        
        # More analysis features can be added here
        st.markdown("""
        ### Feature Description
        
        - **Single File Prediction**: Upload a single ECG file for delirium risk prediction
        - **Batch Prediction**: Process multiple ECG files simultaneously
        - **Results Visualization**: Display prediction results and risk distribution through charts
        - **Results Download**: Support downloading results in JSON format
        
        ### Usage Workflow
        
        1. Click "Initialize System" button to load the model
        2. Upload ECG CSV file(s)
        3. System automatically validates file format
        4. Click prediction button to get results
        5. View risk assessment and visualization results
        """)
    
    with tab4:
        st.header("System Information")
        
        # Check if system is initialized
        is_initialized = (app.preprocessor is not None and 
                         app.feature_extractor is not None and 
                         app.predictor is not None and 
                         app.predictor.model is not None)
        
        if not is_initialized:
            st.warning("""
            ⚠️ **System Not Initialized**
            
            Please click the **"🔄 Initialize System"** button in the left sidebar to load all components.
            """)
        else:
            st.success("✅ **System Ready** - All components loaded successfully")
        
        # Display system status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Configuration")
            
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <p style='margin: 0.5rem 0;'><strong>📊 Sampling Rate:</strong> {app.sampling_rate} Hz</p>
                <p style='margin: 0.5rem 0;'><strong>💾 Temp Directory:</strong><br/><code style='font-size: 0.8rem;'>{app.temp_dir}</code></p>
                {"<p style='margin: 0.5rem 0;'><strong>🎯 Model Type:</strong> " + app.predictor.model_type + "</p>" if app.predictor and hasattr(app.predictor, 'model_type') else ""}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.subheader("Quick Actions")
            
            if is_initialized:
                st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <p style='margin: 0.5rem 0;'>✅ System is ready to use</p>
                    <p style='margin: 0.5rem 0;'>📤 Upload ECG files in the "Single File" tab</p>
                    <p style='margin: 0.5rem 0;'>🔮 Click "Start Prediction" to analyze</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <p style='margin: 0.5rem 0;'>👉 Click "Initialize System" button</p>
                    <p style='margin: 0.5rem 0;'>⏱️ Wait for components to load</p>
                    <p style='margin: 0.5rem 0;'>✅ System will be ready</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Model information
        if app.predictor and app.predictor.model:
            st.subheader("Model Details")
            
            if app.predictor.model_path:
                st.write("📁 Model Path:", app.predictor.model_path)
            
            if hasattr(app.predictor, 'feature_list') and app.predictor.feature_list:
                st.write("🏷️ Feature Count:", len(app.predictor.feature_list))
                
                with st.expander("View Feature List"):
                    for i, feature in enumerate(app.predictor.feature_list, 1):
                        st.write(f"{i}. {feature}")
        
        # Version information
        st.markdown("---")
        st.subheader("Version Information")
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 10px;'>
            <p style='margin: 0.5rem 0;'><strong>🏥 System Version:</strong> 1.0.0</p>
            <p style='margin: 0.5rem 0;'><strong>📅 Build Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
            <p style='margin: 0.5rem 0;'><strong>👨‍💻 Developer:</strong> ECG Analysis Team</p>
            <p style='margin: 0.5rem 0;'><strong>🐍 Python:</strong> {sys.version.split()[0]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start guide
        if not is_initialized:
            st.markdown("---")
            st.subheader("📚 Quick Start Guide")
            st.markdown("""
            1. **Initialize System**: Click the "🔄 Initialize System" button in the left sidebar
            2. **Upload ECG File**: Go to "Single File" tab and upload a CSV file
            3. **Start Prediction**: Click the "🔮 Start Prediction" button
            4. **View Results**: See risk assessment, probabilities, and SHAP analysis
            
            💡 **Tip**: Make sure your ECG CSV file contains all 12 leads (I, II, III, aVR, aVL, aVF, V1-V6)
            """)

if __name__ == "__main__":
    main()
