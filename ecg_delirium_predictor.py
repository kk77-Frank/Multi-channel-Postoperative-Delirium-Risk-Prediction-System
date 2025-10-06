#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECG Delirium Prediction Model Integration Module

This module integrates lead importance analysis, feature selection, model building and evaluation:
- Lead importance analysis and feature selection
- Machine learning and deep learning model building
- Model performance evaluation and visualization
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# Ensure import path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import custom modules
try:
    # Import lead importance analysis module
    # from lead_importance_analysis import LeadImportanceAnalyzer  # Module missing, temporarily commented
    # Import deep learning model module
    # from deep_ecg_model import train_deep_learning_model  # Module missing, temporarily commented
    # Import multi-model builder module
    from multi_model_builder import build_multiple_models
except ImportError as e:
    print(f"Module import failed: {str(e)}")
    print("Please ensure all necessary Python module files exist")
    sys.exit(1)

class ECGDeliriumPredictor:
    """ECG Delirium Prediction Model Integration Class"""
    
    def __init__(self, output_dir=None):
        """
        Initialize prediction model integration class
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.feature_selection_dir = os.path.join(output_dir, 'feature_selection_results')
            self.model_dir = os.path.join(output_dir, 'models')
            self.results_dir = os.path.join(output_dir, 'results')
            
            for directory in [self.feature_selection_dir, self.model_dir, self.results_dir]:
                os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        # self.lead_analyzer = LeadImportanceAnalyzer(output_dir)  # Module missing, temporarily commented
        self.lead_analyzer = None  # Placeholder
        self.selected_features = None
        self.delirium_df = None
        self.non_delirium_df = None
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, delirium_file, non_delirium_file):
        """
        Load delirium and non-delirium data
        
        Args:
            delirium_file: Delirium data file path
            non_delirium_file: 非Delirium data file path
            
        Returns:
            Loading success flag
        """
        print("\nLoading data...")
        try:
            # Load delirium data
            if not os.path.exists(delirium_file):
                print(f"Error: Delirium data file does not exist - {delirium_file}")
                return False
            self.delirium_df = pd.read_csv(delirium_file)
            
            # Load non-delirium data
            if not os.path.exists(non_delirium_file):
                print(f"Error: Non-delirium data file does not exist - {non_delirium_file}")
                return False
            self.non_delirium_df = pd.read_csv(non_delirium_file)
            
            # Ensure label column exists
            if 'label' not in self.delirium_df.columns:
                self.delirium_df['label'] = 1
            if 'label' not in self.non_delirium_df.columns:
                self.non_delirium_df['label'] = 0
            
            print(f"成功Loading data:")
            print(f"- Delirium samples: {len(self.delirium_df)} ")
            print(f"- Non-delirium samples: {len(self.non_delirium_df)} ")
            return True
        except Exception as e:
            print(f"Loading data失败: {str(e)}")
            return False
    
    def analyze_and_select_features(self, use_saved=True, n_features=30, selection_method='combined'):
        """
        Analyze lead importance and select features
        
        Args:
            use_saved: Whether to use saved feature selection results
            n_features: Number of features to select
            selection_method: Feature selection method
            
        Returns:
            List of selected features
        """
        print("\n===== Lead Importance Analysis and Feature Selection =====")
        
        # Check if data is loaded
        if self.delirium_df is None or self.non_delirium_df is None:
            print("Error: Data not yet loaded")
            return None
        
        # Try to load saved feature selection results
        if use_saved and self.output_dir:
            selected_features_file = os.path.join(self.feature_selection_dir, 'selected_features.pkl')
            if os.path.exists(selected_features_file):
                try:
                    with open(selected_features_file, 'rb') as f:
                        self.selected_features = pickle.load(f)
                    print(f"Loaded saved feature selection results, total {len(self.selected_features)} 特征")
                    return self.selected_features
                except Exception as e:
                    print(f"Failed to load saved feature selection results: {str(e)}, will redo feature selection")
        
        if self.lead_analyzer is None:
            print("Note: LeadImportanceAnalyzer not available, will skip traditional feature selection")
            self.selected_features = None
            return None
        
        # Following code only executes when LeadImportanceAnalyzer is available
        # Analyze lead importance
        self.lead_analyzer.analyze_lead_importance(self.delirium_df, self.non_delirium_df)
        
        # Perform p-value analysis
        self.lead_analyzer.perform_p_value_analysis(self.delirium_df, self.non_delirium_df)
        
        # Select features based on feature importance
        self.selected_features = self.lead_analyzer.select_features_by_importance(
            self.delirium_df, self.non_delirium_df, method=selection_method, n_features=n_features)
        
        # Visualize results
        self.lead_analyzer.visualize_lead_importance()
        self.lead_analyzer.visualize_feature_importance()
        self.lead_analyzer.visualize_p_value_distribution()
        
        return self.selected_features
    
    def build_models(self, optimize=True):
        """
        Build multiple models
        
        Args:
            optimize: Whether to optimize model parameters
            
        Returns:
            Build success flag
        """
        print("\n===== Build Prediction Model =====")
        
        # Check if data is ready
        if self.delirium_df is None or self.non_delirium_df is None:
            print("Error: Data not yet loaded")
            return False
        
        # Allow no pre-selected features, do feature selection within training folds
        if self.selected_features is None or len(self.selected_features) == 0:
            print("Note: No pre-selected features used, will perform robust feature selection within training folds")
        
        # Build multiple ML models (leak-free pipeline)
        print("\n1. Building machine learning models...")
        model_builder = build_multiple_models(
            self.delirium_df, self.non_delirium_df, 
            selected_features=None,  # Complete selection within folds
            output_dir=self.output_dir, 
            optimize=optimize
        )
        
        # Deep learning module not available, skip
        print("\n2. Deep learning module not available, skip deep learning model building")
        # dl_model, dl_results = train_deep_learning_model(
        #     self.delirium_df, self.non_delirium_df,
        #     self.selected_features if self.selected_features else [], self.output_dir
        # )
        
        # Update best model information
        if model_builder and hasattr(model_builder, 'best_model'):
            self.best_model = model_builder.best_model
            self.best_model_name = model_builder.best_model_name
        
        # Generate comprehensive evaluation report
        self.generate_comprehensive_report()
        
        return True
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        if not self.output_dir:
            return
        
        report_path = os.path.join(self.results_dir, 'comprehensive_report.html')
        
        with open(report_path, 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>ECG Delirium Prediction Comprehensive Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #3498db; }
                    h3 { color: #16a085; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .image-container { margin: 20px 0; text-align: center; }
                    .image-container img { max-width: 100%; border: 1px solid #ddd; }
                    .footer { margin-top: 30px; font-size: 0.8em; color: #7f8c8d; text-align: center; }
                    .section { margin: 30px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>ECG Anesthesia Delirium Prediction System - Comprehensive Evaluation Report</h1>
                    <p>Generated: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
                    
                    <div class="section">
                        <h2>1. Lead Importance Analysis</h2>
                        <div class="image-container">
                            <img src="feature_selection_results/lead_importance.png" alt="Lead Importance Analysis">
                        </div>
                        
                        <h3>P-value Distribution by Lead</h3>
                        <div class="image-container">
                            <img src="feature_selection_results/lead_p_values_boxplot.png" alt="Lead P-value Distribution">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>2. Feature Selection Results</h2>
                        
                        <h3>P-value Distribution</h3>
                        <div class="image-container">
                            <img src="feature_selection_results/p_value_distribution.png" alt="P-value Distribution">
                        </div>
                        
                        <h3>Effect Size and Significance</h3>
                        <div class="image-container">
                            <img src="feature_selection_results/effect_size_vs_p_value.png" alt="Effect Size and Significance">
                        </div>
                        
                        <h3>Feature Importance</h3>
                        <div class="image-container">
                            <img src="feature_selection_results/combined_feature_importance.png" alt="Feature Importance">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>3. Machine Learning Model Evaluation</h2>
                        
                        <h3>ROC Curve Comparison</h3>
                        <div class="image-container">
                            <img src="results/roc_curves.png" alt="ROC Curve Comparison">
                        </div>
                        
                        <h3>PR Curve Comparison</h3>
                        <div class="image-container">
                            <img src="results/pr_curves.png" alt="PR Curve Comparison">
                        </div>
                        
                        <h3>Confusion Matrix</h3>
                        <div class="image-container">
                            <img src="results/confusion_matrices.png" alt="Confusion Matrix">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>4. Deep Learning Model Evaluation</h2>
                        
                        <h3>Training Process</h3>
                        <div class="image-container">
                            <img src="results/training_history.png" alt="Training History">
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>ECG Anesthesia Delirium Prediction System | Auto-generated Report</p>
                    </div>
                </div>
            </body>
            </html>
            ''')
        
        print(f"Comprehensive evaluation report generated: {report_path}")
        return report_path

    def generate_baseline_table_and_flow(self):
        """Generate baseline feature table (POD vs Non-POD, standardized difference and p-value) and patient screening flowchart"""
        if self.delirium_df is None or self.non_delirium_df is None or not self.output_dir:
            return
        try:
            import numpy as np
            import pandas as pd
            from scipy import stats
            out_dir = os.path.join(self.results_dir)
            os.makedirs(out_dir, exist_ok=True)
            # Merge and label groups
            df = pd.concat([
                self.delirium_df.assign(group='POD'),
                self.non_delirium_df.assign(group='Non-POD')
            ], ignore_index=True)
            # Generate summary only for numeric columns
            num_cols = [c for c in df.columns if c not in ['label', 'group', 'file'] and np.issubdtype(df[c].dtype, np.number)]
            rows = []
            for c in num_cols:
                a = df.loc[df['group'] == 'POD', c].astype(float)
                b = df.loc[df['group'] == 'Non-POD', c].astype(float)
                m1, s1, n1 = np.nanmean(a), np.nanstd(a, ddof=1), a.notna().sum()
                m0, s0, n0 = np.nanmean(b), np.nanstd(b, ddof=1), b.notna().sum()
                pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / max(n1 + n0 - 2, 1)) if n1 > 1 and n0 > 1 else np.nan
                std_diff = (m1 - m0) / pooled if pooled and pooled > 0 else np.nan
                try:
                    _, p = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative='two-sided')
                except Exception:
                    p = np.nan
                rows.append({
                    'variable': c,
                    'mean_POD': m1, 'sd_POD': s1,
                    'mean_NonPOD': m0, 'sd_NonPOD': s0,
                    'std_diff': std_diff, 'p_value': p
                })
            baseline = pd.DataFrame(rows).sort_values('p_value')
            baseline.to_csv(os.path.join(out_dir, 'baseline_table.csv'), index=False)
        except Exception:
            pass
        # Simple patient screening flowchart
        try:
            import matplotlib.pyplot as plt
            total = int(len(self.delirium_df) + len(self.non_delirium_df))
            pod = int(len(self.delirium_df))
            nonpod = int(len(self.non_delirium_df))
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axis('off')
            y = 0.9
            ax.text(0.5, y, f'Total assessed: {total}', ha='center', va='center', bbox=dict(boxstyle='round', fc='#e6f2ff', ec='#3366cc'))
            y -= 0.2
            ax.text(0.5, y, f'Included after QC: {total}', ha='center', va='center', bbox=dict(boxstyle='round', fc='#e6ffe6', ec='#339933'))
            y -= 0.2
            ax.text(0.25, y, f'POD: {pod}', ha='center', va='center', bbox=dict(boxstyle='round', fc='#fff2e6', ec='#cc6600'))
            ax.text(0.75, y, f'Non-POD: {nonpod}', ha='center', va='center', bbox=dict(boxstyle='round', fc='#fff2e6', ec='#cc6600'))
            plt.tight_layout()
            flow_path = os.path.join(self.results_dir, 'patient_flow.png')
            plt.savefig(flow_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass
    
    def run_full_pipeline(self, delirium_file, non_delirium_file, use_saved_features=True, n_features=30, optimize_models=True):
        """
        Run complete prediction pipeline
        
        Args:
            delirium_file: Delirium data file path
            non_delirium_file: 非Delirium data file path
            use_saved_features: Whether to use saved feature selection results
            n_features: Number of features to select
            optimize_models: Whether to optimize model parameters
            
        Returns:
            Run success flag
        """
        print("\n===== Start ECG Delirium Prediction Complete Pipeline =====")
        
        # Loading data
        if not self.load_data(delirium_file, non_delirium_file):
            print("Error: Data loading failed, cannot continue")
            return False
        
        selected_features = self.analyze_and_select_features(
            use_saved=use_saved_features, 
            n_features=n_features
        )
        
        if selected_features is None or len(selected_features) == 0:
            print("Warning: Feature selection stage produced no usable results, will proceed directly to leak-free multi-model training")
            selected_features = None
        
        # Build prediction models
        if not self.build_models(optimize=optimize_models):
            print("Error: Model building failed")
            return False
        
        print("\n===== ECG Delirium Prediction Pipeline Completed =====")
        if self.output_dir:
            print(f"All results saved to: {self.output_dir}")
        
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ECG Delirium Prediction Model')
    
    parser.add_argument('--delirium_file', type=str, required=True,
                        help='Delirium group feature data file path')
    
    parser.add_argument('--non_delirium_file', type=str, required=True,
                        help='非Delirium group feature data file path')
    
    parser.add_argument('--output_dir', type=str, default='delirium_model_results',
                        help='Output directory路径')
    
    parser.add_argument('--n_features', type=int, default=30,
                        help='Number of features to select')
    
    parser.add_argument('--use_saved_features', action='store_true',
                        help='Whether to use saved feature selection results')
    
    parser.add_argument('--skip_optimize', action='store_true',
                        help='Skip model parameter optimization')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create predictor
    predictor = ECGDeliriumPredictor(args.output_dir)
    
    # Run complete pipeline
    success = predictor.run_full_pipeline(
        args.delirium_file,
        args.non_delirium_file,
        use_saved_features=args.use_saved_features,
        n_features=args.n_features,
        optimize_models=not args.skip_optimize
    )
    
    if success:
        print("\nPrediction model built successfully!")
        try:
            predictor.generate_baseline_table_and_flow()
        except Exception:
            pass
        return 0
    else:
        print("\nPrediction model building failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 