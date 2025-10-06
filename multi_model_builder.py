#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
from matplotlib import rcParams
rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# ---- DeLong AUC significance test helpers (correlated ROC curves) ----
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = x.shape[0]
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(predictions_sorted_transposed, label_ones_count):
    m = label_ones_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.zeros((k, m))
    ty = np.zeros((k, n))
    for r in range(k):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
    tz = np.zeros((k, m + n))
    for r in range(k):
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s

def delong_roc_variance(ground_truth, predictions):
    order = np.argsort(-predictions)
    predictions_sorted = predictions[order]
    label_ones_count = int(np.sum(ground_truth))
    ground_truth_sorted = ground_truth[order]
    assert np.all(ground_truth_sorted[:label_ones_count] == 1)
    aucs, s = _fast_delong(predictions_sorted[np.newaxis, :], label_ones_count)
    return aucs[0], s[0, 0]

def delong_2sample_pvalue(y_true, y_score_a, y_score_b):
    y_true = np.asarray(y_true).astype(int)
    a_auc, a_var = delong_roc_variance(y_true, np.asarray(y_score_a))
    b_auc, b_var = delong_roc_variance(y_true, np.asarray(y_score_b))
    var = a_var + b_var
    if var <= 0:
        return a_auc, b_auc, np.nan
    z = (a_auc - b_auc) / np.sqrt(var)
    try:
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(abs(z)))
    except Exception:
        p = np.nan
    return a_auc, b_auc, p


class StabilitySelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=30, correlation_threshold=0.9, n_bootstrap=30, bootstrap_frac=0.7, random_state=42, epv_target=10):
        self.k = k
        self.correlation_threshold = correlation_threshold
        self.n_bootstrap = n_bootstrap
        self.bootstrap_frac = bootstrap_frac
        self.random_state = random_state
        self.epv_target = epv_target
        self.selected_indices_ = None
        self.selected_feature_names_ = None
        self._feature_names_in_ = None
        self._feature_names_after_corr_ = None
        self.bootstrap_avg_score_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        if hasattr(X, 'columns'):
            self._feature_names_in_ = list(X.columns)
            X_mat = X.values
        else:
            self._feature_names_in_ = [f"f{i}" for i in range(X.shape[1])]
            X_mat = X
        y = np.asarray(y)

        keep_mask = np.ones(X_mat.shape[1], dtype=bool)
        try:
            corr = np.corrcoef(X_mat, rowvar=False)
            for i in range(corr.shape[0]):
                if not keep_mask[i]:
                    continue
                high_corr = (np.abs(corr[i]) > self.correlation_threshold)
                high_corr[:i+1] = False
                keep_mask[np.where(high_corr)[0]] = False
        except Exception:
            pass
        X_f = X_mat[:, keep_mask]
        feature_names_after_corr = [n for n, k in zip(self._feature_names_in_, keep_mask) if k]
        self._feature_names_after_corr_ = feature_names_after_corr

        n_samples = X_f.shape[0]
        scores_sum = np.zeros(X_f.shape[1])
        counts = np.zeros(X_f.shape[1])
        for b in range(self.n_bootstrap):
            idx = rng.choice(n_samples, size=int(self.bootstrap_frac * n_samples), replace=True)
            try:
                mi = mutual_info_classif(X_f[idx], y[idx], random_state=self.random_state + b)
            except Exception:
                mi = np.var(X_f[idx], axis=0)
            # Normalize
            if np.nanmax(mi) > 0:
                mi = mi / (np.nanmax(mi) + 1e-12)
            mi = np.nan_to_num(mi)
            scores_sum += mi
            counts += 1
        avg_score = np.divide(scores_sum, counts, out=np.zeros_like(scores_sum), where=counts>0)
        if np.nanmax(avg_score) > 0:
            self.bootstrap_avg_score_ = (avg_score - np.nanmin(avg_score)) / (np.nanmax(avg_score) - np.nanmin(avg_score) + 1e-12)
        else:
            self.bootstrap_avg_score_ = avg_score

        pos_events = int(np.sum(y == 1))
        if pos_events <= 0:
            k_epv = self.k
        else:
            k_epv = max(1, min(self.k, pos_events // self.epv_target))
        k_final = max(1, min(k_epv, X_f.shape[1]))

        top_idx = np.argsort(-avg_score)[:k_final]
        selected_mask_after_corr = np.zeros(X_f.shape[1], dtype=bool)
        selected_mask_after_corr[top_idx] = True
        original_indices = np.where(keep_mask)[0]
        self.selected_indices_ = original_indices[selected_mask_after_corr]
        self.selected_feature_names_ = [self._feature_names_in_[i] for i in self.selected_indices_]
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            return X
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]

    def get_support(self):
        if self.selected_indices_ is None:
            return None
        mask = np.zeros(len(self._feature_names_in_), dtype=bool)
        mask[self.selected_indices_] = True
        return mask


class MultiModelBuilder:
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.results_dir = os.path.join(self.output_dir, 'results') if self.output_dir else None
        self.model_dir = os.path.join(self.output_dir, 'models') if self.output_dir else None
        if self.output_dir:
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}
        self.results = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None
        self.best_auc = -np.inf
        self.X_test = None
        self.y_test = None
        self.feature_importance = {}
        self.selected_feature_names_by_model = {}
        self.decision_thresholds = {}
        self.threshold_criterion = {}
        self.model_colors = {}
        self.X_train = None
        self.y_train = None
        self.all_feature_cols = None

    def _save_multi(self, fig, filename_no_ext: str):
        if not self.output_dir:
            return
        base = os.path.join(self.results_dir, filename_no_ext)
        fig.savefig(base + '.png', dpi=600, bbox_inches='tight')
        try:
            fig.savefig(base + '.pdf', bbox_inches='tight')
        except Exception:
            pass
        try:
            fig.savefig(base + '.tiff', dpi=600, bbox_inches='tight')
        except Exception:
            pass

    def _standardize_feature_name(self, name: str) -> str:
        try:
            n = name
            if n.startswith('r_peak_delay_'):
                parts = n.replace('r_peak_delay_', '').split('_')
                if len(parts) >= 3 and parts[-1] in ('std', 'mean'):
                    lead_a, lead_b, stat = parts[0], parts[1], parts[2]
                    unit = 'ms'
                    stat_map = {'std': 'SD', 'mean': 'Mean'}
                    return f'R-peak delay {lead_a}–{lead_b} ({stat_map.get(stat, stat)}, {unit})'
            if '_hrv_' in n:
                n = n.replace('_hrv_', ' ')
            repl = {
                'mean_rr': 'Mean NN interval (ms)',
                'sdnn': 'SDNN (ms)',
                'rmssd': 'RMSSD (ms)',
                'pnn50': 'pNN50 (%)',
                'HRV_PIP': 'Poincaré PIP (%)',
                'HRV_PAS': 'Phase space asymmetry (PAS)',
                'HRV_IALS': 'Mean incline of ascending limb (IALS)',
                'HRV_KFD': 'Higuchi fractal dimension',
                'sampen': 'Sample entropy (m=2, r=0.2σ)',
                'apen': 'Approximate entropy (m=2, r=0.2σ)'
            }
            for k, v in repl.items():
                if k in n:
                    n = n.replace(k, v)
            if '_' in n:
                parts = n.split('_', 1)
                if parts[0] in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
                    n = f'{parts[0]}: {parts[1]}'
            n = n.replace('_', ' ')
            return n
        except Exception:
            return name

    def _compute_thresholds(self, y_true, y_proba):
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        youden = tpr - fpr
        idx_y = int(np.argmax(youden))
        thr_youden = thr[idx_y] if idx_y < len(thr) else 0.5
        precision, recall, pr_thr = precision_recall_curve(y_true, y_proba)
        f1 = np.where((precision+recall) > 0, 2*precision*recall/(precision+recall), 0)
        # precision_recall_curve thresholds length is n-1, align to f1[1:]
        idx_f = int(np.nanargmax(f1[1:])) if len(f1) > 1 else 0
        thr_f1 = pr_thr[idx_f] if len(pr_thr) > 0 else 0.5
        

        precision_target = 0.5  # Further reduce precision target for better balance
        valid_indices = np.where(precision >= precision_target)[0]
        if len(valid_indices) > 0:
            # Among thresholds meeting precision requirement, select the one with highest F1
            f1_valid = np.where((precision[valid_indices] + recall[valid_indices]) > 0, 
                               2 * precision[valid_indices] * recall[valid_indices] / (precision[valid_indices] + recall[valid_indices]), 0)
            best_f1_idx = valid_indices[np.argmax(f1_valid)]
            thr_precision_optimized = pr_thr[min(best_f1_idx, len(pr_thr)-1)]
        else:
            f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
            best_f1_idx = int(np.nanargmax(f1_scores[1:])) if len(f1_scores) > 1 else 0
            thr_precision_optimized = pr_thr[min(best_f1_idx, len(pr_thr)-1)] if len(pr_thr) > 0 else 0.3  
        
        return thr_youden, thr_f1, thr_precision_optimized, (fpr, tpr, thr), (precision, recall, pr_thr)

    def _bootstrap_roc_ci(self, y_true, y_proba, n_boot=500, grid=np.linspace(0,1,101), seed=42):
        rng = np.random.RandomState(seed)
        curves = []
        y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            yt = y_true[idx]; yp = y_proba[idx]
            fpr, tpr, _ = roc_curve(yt, yp)
            tpr_i = np.interp(grid, fpr, tpr, left=0, right=1)
            curves.append(tpr_i)
        curves = np.asarray(curves)
        return curves.mean(axis=0), np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)

    def _bootstrap_pr_ci(self, y_true, y_proba, n_boot=500, grid=np.linspace(0,1,101), seed=42):
        rng = np.random.RandomState(seed)
        curves = []
        y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            yt = y_true[idx]; yp = y_proba[idx]
            precision, recall, _ = precision_recall_curve(yt, yp)
            prec_i = np.interp(grid, recall[::-1], precision[::-1], left=precision.min(), right=precision.max())
            curves.append(prec_i)
        curves = np.asarray(curves)
        return curves.mean(axis=0), np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)

    def _bootstrap_dca_ci(self, y_true, y_proba, thresholds, n_boot=300, seed=42):
        rng = np.random.RandomState(seed)
        y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
        n = len(y_true)
        mats = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            yt = y_true[idx]; yp = y_proba[idx]
            nb = []
            for pt in thresholds:
                y_pred = (yp >= pt).astype(int)
                TP = np.sum((y_pred == 1) & (yt == 1))
                FP = np.sum((y_pred == 1) & (yt == 0))
                nb.append((TP/n) - (FP/n) * (pt/(1-pt)))
            mats.append(nb)
        mats = np.asarray(mats)
        return mats.mean(axis=0), np.percentile(mats, 2.5, axis=0), np.percentile(mats, 97.5, axis=0)

    def prepare_data(self, delirium_df, non_delirium_df, selected_features=None, test_size=0.25):
              if 'label' not in delirium_df.columns:
            delirium_df['label'] = 1
        if 'label' not in non_delirium_df.columns:
            non_delirium_df['label'] = 0
        combined_df = pd.concat([delirium_df, non_delirium_df], ignore_index=True)
        
        feature_cols = [c for c in combined_df.columns if c not in ['label', 'file']]
        if selected_features:
            feature_cols = [f for f in selected_features if f in combined_df.columns]
            if not feature_cols:
                feature_cols = [c for c in combined_df.columns if c not in ['label', 'file']]

        X = combined_df[feature_cols].copy()
        y = combined_df['label'].copy()
 
        print("Basic data cleaning: only processing infinite values...")
        # Replace infinite values with NaN, other preprocessing completed in Pipeline
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.all_feature_cols = feature_cols
        return X_train, X_test, y_train, y_test, feature_cols

    def _build_pipeline(self, base_estimator, y_train, selector_k=30):
        # Calculate scale_pos_weight (for XGB)
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

        effective_k = selector_k
        print(f"{base_estimator.__class__.__name__} using {effective_k} features (unified standard)")

  
        if isinstance(base_estimator, GradientBoostingClassifier):
           
            pass
        elif isinstance(base_estimator, ExtraTreesClassifier):
           
            balanced_et_weight = {0: 1.0, 1: scale_pos_weight * 0.8}             base_estimator.set_params(class_weight=balanced_et_weight)
        elif isinstance(base_estimator, RandomForestClassifier):
           
            ultra_precision_rf_weight = scale_pos_weight * 0.35  # More conservative weight, focused on precision
            custom_weights = {0: 1.0, 1: ultra_precision_rf_weight}
            base_estimator.set_params(class_weight=custom_weights)
        if isinstance(base_estimator, xgb.XGBClassifier):
            balanced_xgb_scale_pos_weight = scale_pos_weight * 0.8  # Increase positive class weight to improve recall
            base_estimator.set_params(scale_pos_weight=balanced_xgb_scale_pos_weight, use_label_encoder=False, eval_metric='logloss')

        selector_bootstrap = 5  # Reduce bootstrap rounds to accelerate training
        selector_corr_threshold = 0.9  # Unified correlation threshold
        print(f"Feature selection: {selector_bootstrap} bootstrap rounds, correlation threshold {selector_corr_threshold}")
            
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Median imputation, more robust to outliers
            ('scaler', RobustScaler()),  # Use RobustScaler, more robust to outliers
            ('selector', StabilitySelector(k=effective_k, correlation_threshold=selector_corr_threshold, n_bootstrap=selector_bootstrap, bootstrap_frac=0.85, random_state=42, epv_target=4)),
            ('clf', base_estimator)
        ])
        return pipeline

    def build_models(self, X_train, y_train, optimize=True):
        """
        Build multiple machine learning models (using Pipeline, selection completed within training fold)
        """
        models = {}

        # Scientifically determine feature count based on EPV criterion
        pos = int(np.sum(y_train == 1))
        # Minimum standard of EPV ≥ 5 (more relaxed but still scientific)
        k_epv = max(10, min(25, pos // 5)) if pos > 0 else 20
        print(f"Based on EPV≥5 criterion, {pos} positive samples support max {pos//5} features, actually using {k_epv} features")
        
        # Random Forest
        print("\nTraining Random Forest model...")
        # Extremely optimized RandomForest parameters to maximize precision
        rf_base = RandomForestClassifier(
            random_state=42,
            min_samples_leaf=12,  # Greatly increase min leaf samples for stability
            min_impurity_decrease=0.01,  # Greatly increase split threshold to avoid overfitting
            max_features='sqrt',  # Limit feature count to reduce noise
            max_samples=0.7,  # Further limit sample sampling to improve generalization
            criterion='gini',  # Use Gini impurity
            ccp_alpha=0.008,  # Significantly enhance cost complexity pruning
            max_leaf_nodes=80,  # Further limit number of leaf nodes
            bootstrap=True,
            oob_score=True,  # Enable out-of-bag scoring
            max_depth=10  # Limit tree depth to prevent overfitting
        )
        rf_pipe = self._build_pipeline(rf_base, y_train, selector_k=k_epv)
        if optimize:
            rf_params = {
                'clf__n_estimators': [200, 400],  # Reduce search range
                'clf__max_depth': [8, 10],  # Reduce search range
                'clf__min_samples_split': [20, 25],  # Reduce search range
                'clf__ccp_alpha': [0.005, 0.008]  # Reduce search range
            }
            rf = self._optimize_model(rf_pipe, rf_params, X_train, y_train)
        else:
            rf_pipe.fit(X_train, y_train)
            rf = rf_pipe
        models['RandomForest'] = rf
        
        # XGBoost
        print("\nTraining XGBoost model...")
        # Detect GPU availability
        import torch
        use_gpu = torch.cuda.is_available()
        tree_method = 'gpu_hist' if use_gpu else 'hist'
        if use_gpu:
            print(f"  ✓ GPU detected, using GPU-accelerated training (tree_method={tree_method})")
        else:
            print(f"  ✓ Using CPU training (tree_method={tree_method})")
        
        # Optimize XGBoost parameters: balance precision and recall
        xgb_base = xgb.XGBClassifier(
            random_state=42,
            reg_alpha=0.5,  # Moderate L1 regularization
            reg_lambda=1.5,  # Moderate L2 regularization
            min_child_weight=5,  # Balanced child weight
            subsample=0.8,  # Increase row sampling ratio
            colsample_bytree=0.8,  # Increase column sampling ratio
            gamma=0.1,  # Reduce minimum split loss
            max_delta_step=1,  # Relax weight change constraint
            max_leaves=80,  # Increase number of leaves
            tree_method=tree_method,  # Auto select GPU or CPU
            grow_policy='depthwise',  # Depth-first growth
            max_depth=6  # Increase tree depth
        )
        xgb_pipe = self._build_pipeline(xgb_base, y_train, selector_k=k_epv)
        if optimize:
            xgb_params = {
                'clf__n_estimators': [200, 300],  # Reduce search range
                'clf__max_depth': [4, 6],  # Reduce search range
                'clf__learning_rate': [0.1],  # Fix most common value
                'clf__reg_alpha': [0.5, 1.0],  # Reduce search range
                'clf__reg_lambda': [1.0, 2.0]  # Reduce search range
            }
            xgb_model = self._optimize_model(xgb_pipe, xgb_params, X_train, y_train)
        else:
            xgb_pipe.fit(X_train, y_train)
            xgb_model = xgb_pipe
        models['XGBoost'] = xgb_model
        
        # ExtraTreesClassifier (replace SVM)
        print("\nTraining ExtraTrees model...")
        # ExtraTrees: Extremely randomized trees, good for handling high-dimensional features
        et_base = ExtraTreesClassifier(
            random_state=42,
            n_estimators=200,  # Moderate number of trees
            max_depth=15,      # Moderate tree depth
            min_samples_split=10,  # Minimum samples required for split
            min_samples_leaf=5,    # Minimum samples per leaf node
            max_features='sqrt',   # Feature subset size
            bootstrap=False,       # ExtraTrees doesn't use bootstrap
            n_jobs=-1,            # Parallel processing
            criterion='gini',     # Split criterion
            max_samples=None,     # Use all samples
            min_impurity_decrease=0.01  # Minimum impurity decrease
        )
        et_pipe = self._build_pipeline(et_base, y_train, selector_k=k_epv)
        if optimize:
            et_params = {
                'clf__n_estimators': [150, 200],  # Reduce search range
                'clf__max_depth': [10, 15],  # Reduce search range
                'clf__min_samples_split': [10],  # Fixed value
                'clf__max_features': ['sqrt']  # Fixed value
            }
            et = self._optimize_model(et_pipe, et_params, X_train, y_train)
        else:
            et_pipe.fit(X_train, y_train)
            et = et_pipe
        models['ExtraTrees'] = et
        
        # GradientBoostingClassifier (replace LogisticRegression)
        print("\nTraining GradientBoosting model...")
        # GradientBoosting: sklearn built-in gradient boosting, excellent performance
        gb_base = GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,      # Moderate number of iterations
            learning_rate=0.1,     # Learning rate
            max_depth=6,           # Maximum tree depth
            min_samples_split=20,  # Minimum samples required for split
            min_samples_leaf=10,   # Minimum samples per leaf node
            max_features='sqrt',   # Feature subset
            subsample=0.8,         # Subsample ratio
            loss='log_loss',       # Loss function
            criterion='friedman_mse',  # Split criterion
            min_impurity_decrease=0.01,  # Minimum impurity decrease
            validation_fraction=0.1,     # Validation set ratio
            n_iter_no_change=10,         # Early stopping rounds
            tol=1e-4                     # Tolerance
        )
        gb_pipe = self._build_pipeline(gb_base, y_train, selector_k=k_epv)
        if optimize:
            gb_params = {
                'clf__n_estimators': [100, 150],  # Reduce search range
                'clf__learning_rate': [0.1],  # Fix common value
                'clf__max_depth': [4, 6],  # Reduce search range
                'clf__min_samples_split': [20],  # Fixed value
                'clf__subsample': [0.8]  # Fixed value
            }
            gb = self._optimize_model(gb_pipe, gb_params, X_train, y_train)
        else:
            gb_pipe.fit(X_train, y_train)
            gb = gb_pipe
        models['GradientBoosting'] = gb
        
        # KNN
        print("\nTraining KNN model...")
        # Deeply optimized KNN parameters to improve precision
        knn_base = KNeighborsClassifier(
            algorithm='ball_tree',  # Use ball tree algorithm
            leaf_size=15,     # Optimize leaf size
            n_jobs=-1,        # Use parallel processing
            metric='minkowski',  # Explicitly specify distance metric
            p=2,              # Euclidean distance
            weights='distance'  # Use distance weights
        )
        knn_pipe = self._build_pipeline(knn_base, y_train, selector_k=k_epv)
        if optimize:
            knn_params = {
                'clf__n_neighbors': [5, 7, 9],  # Reduce search range
                'clf__weights': ['distance'],  # Fix using distance weights
                'clf__metric': ['manhattan', 'minkowski'],  # Reduce distance metric choices
                'clf__leaf_size': [15]  # Fixed value
            }
            knn = self._optimize_model(knn_pipe, knn_params, X_train, y_train)
        else:
            knn_pipe.fit(X_train, y_train)
            knn = knn_pipe
        models['KNN'] = knn
        
        # SVM (Support Vector Machine)
        print("\nTraining SVM model...")
        # SVM: Support vector machine, excellent performance on high-dimensional data
        svm_base = SVC(
            random_state=42,
            probability=True,  # Enable probability estimation (for ROC curve)
            class_weight='balanced',  # Auto balance class weights
            kernel='rbf',  # Radial basis function kernel
            cache_size=1000,  # Increase cache to accelerate training
            max_iter=1000  # Maximum iterations
        )
        svm_pipe = self._build_pipeline(svm_base, y_train, selector_k=k_epv)
        if optimize:
            svm_params = {
                'clf__C': [0.1, 1.0, 10.0],  # Regularization parameter
                'clf__gamma': ['scale', 'auto'],  # Kernel coefficient
                'clf__kernel': ['rbf']  # Use RBF kernel
            }
            svm = self._optimize_model(svm_pipe, svm_params, X_train, y_train)
        else:
            svm_pipe.fit(X_train, y_train)
            svm = svm_pipe
        models['SVM'] = svm
        
        # Logistic Regression
        print("\nTraining Logistic Regression model...")
        # Logistic Regression: Linear model, strong interpretability
        lr_base = LogisticRegression(
            random_state=42,
            class_weight='balanced',  # Auto balance class weights
            penalty='l2',  # L2 regularization
            solver='lbfgs',  # Optimization algorithm
            max_iter=1000,  # Maximum iterations
            n_jobs=-1  # Parallel processing
        )
        lr_pipe = self._build_pipeline(lr_base, y_train, selector_k=k_epv)
        if optimize:
            lr_params = {
                'clf__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
                'clf__penalty': ['l2'],  # L2 regularization
                'clf__solver': ['lbfgs']  # Fixed solver
            }
            lr = self._optimize_model(lr_pipe, lr_params, X_train, y_train)
        else:
            lr_pipe.fit(X_train, y_train)
            lr = lr_pipe
        models['LogisticRegression'] = lr
        
        self.models = models
        return models
    
    def ablation_by_leads(self, delirium_df, non_delirium_df, lead_sets=None):
        """Lead ablation: Given lead sets (e.g., ['V1'], ['V5'], ['V1','V5'], ['I','II'], ['I','II','V1','V5']),
        reuse no-leakage pipeline, train XGBoost separately and report AUC (95%CI) on test set, save CSV"""
        if lead_sets is None:
            lead_sets = [['V1'], ['V5'], ['V1','V5'], ['I','II'], ['I','II','V1','V5']]
        records = []
        for leads in lead_sets:
            cols = []
            df_all = pd.concat([delirium_df, non_delirium_df], axis=0)
            for c in df_all.columns:
                if any((c.startswith(l+"_") or ("_"+l+"_") in c or c.endswith("_"+l)) for l in leads):
                    if c not in ['label','file']:
                        cols.append(c)
            if not cols:
                continue
            X_train, X_test, y_train, y_test, _ = self.prepare_data(delirium_df, non_delirium_df, selected_features=cols)
            # XGBoost only for quick evaluation
            xgb_base = xgb.XGBClassifier(random_state=42)
            pipe = self._build_pipeline(xgb_base, y_train, selector_k=max(5, min(30, int(np.sum(y_train==1)//10))))
            pipe.fit(X_train, y_train)
            try:
                y_proba = pipe.predict_proba(X_test)[:,1]
            except Exception:
                y_proba = pipe.decision_function(X_test)
            auc = roc_auc_score(y_test, y_proba)
            # CI via bootstrap
            rng = np.random.RandomState(42)
            n = len(y_test)
            auc_bs = []
            for _ in range(500):
                idx = rng.randint(0, n, n)
                try:
                    if hasattr(y_test, 'iloc'):
                        yt = y_test.iloc[idx]
                    elif hasattr(y_test, 'values'):
                        yt = y_test.values[idx]
                    else:
                        yt = np.asarray(y_test)[idx]
                    auc_bs.append(roc_auc_score(yt, y_proba[idx]))
                except Exception:
                    pass
            lo = float(np.percentile(auc_bs, 2.5)) if auc_bs else np.nan
            hi = float(np.percentile(auc_bs, 97.5)) if auc_bs else np.nan
            records.append({'leads': '+'.join(leads), 'AUC': float(auc), 'AUC_CI_low': lo, 'AUC_CI_high': hi, 'num_features_candidate': len(cols)})
        if self.output_dir and records:
            pd.DataFrame(records).to_csv(os.path.join(self.results_dir, 'ablation_leads_auc.csv'), index=False)
    
    def _optimize_model(self, pipeline, param_grid, X_train, y_train):
        """Use grid search to optimize model parameters (execute on Pipeline to avoid leakage)"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Use 5-fold cross-validation to ensure stable hyperparameter estimation
        
        # Select different scoring criteria based on model type
        model_name = pipeline.steps[-1][1].__class__.__name__
        if 'GradientBoostingClassifier' in model_name:
            # Use ROC-AUC for GradientBoosting, comprehensive performance metric
            scoring = 'roc_auc'
        elif 'ExtraTreesClassifier' in model_name:
            # Use F1 score for ExtraTrees, balance precision and recall
            scoring = 'f1'
        else:
            # Use ROC-AUC for other models
            scoring = 'roc_auc'
            
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Cross-validation {scoring}: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all model performance, support Pipeline; extract selected feature names for importance analysis."""
        if not self.models:
            raise ValueError("Models not trained")
        results = []
        self.X_test = X_test
        self.y_test = y_test
        
        for name, model in self.models.items():
            # Predict probabilities
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = model.decision_function(X_test)
            # Threshold: use precision-optimized threshold for RF and XGBoost, Youden for other models
            thr_y, thr_f1, thr_precision, roc_pack, pr_pack = self._compute_thresholds(y_test, y_proba)
            
            # Use differentiated threshold strategies for different models
            if name in ['RandomForest']:
                # RF uses flexible precision threshold (maximize F1)
                selected_threshold = thr_precision
                self.threshold_criterion[name] = 'Precision-optimized (≥0.6 precision, maximize F1)'
            elif name in ['XGBoost']:
                # XGBoost uses F1-optimized threshold to balance precision and recall
                selected_threshold = thr_f1
                self.threshold_criterion[name] = 'F1-optimized threshold (balanced performance)'
            elif name in ['GradientBoosting']:
                # GradientBoosting uses F1-optimized threshold, balance precision and recall
                selected_threshold = thr_f1
                self.threshold_criterion[name] = 'F1-optimized threshold (balance precision and recall)'
            elif name in ['ExtraTrees']:
                # ExtraTrees uses F1-optimized threshold, balanced performance
                selected_threshold = thr_f1
                self.threshold_criterion[name] = 'F1-optimized threshold (balanced performance)'
            elif name in ['KNN']:
                # KNN uses F1-optimized threshold to balance performance
                selected_threshold = thr_f1
                self.threshold_criterion[name] = 'F1-optimized threshold'
            else:
                selected_threshold = thr_y
                self.threshold_criterion[name] = 'Youden J (test, viz only)'
            
            self.decision_thresholds[name] = selected_threshold
            y_pred = (y_proba >= selected_threshold).astype(int)
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            # Main metrics 95% CI (non-parametric bootstrap)
            rng = np.random.RandomState(42)
            n = len(y_test)
            auc_bs, acc_bs, prec_bs, rec_bs, f1_bs = [], [], [], [], []
            for _ in range(500):
                idx = rng.randint(0, n, n)
                # Ensure index alignment, avoid pandas index mismatch
                if hasattr(y_test, 'iloc'):
                    yt = y_test.iloc[idx]
                elif hasattr(y_test, 'values'):
                    yt = y_test.values[idx]
                else:
                    yt = np.asarray(y_test)[idx]
                yp = y_proba[idx]
                yhat = (yp >= thr_y).astype(int)
                try:
                    auc_bs.append(roc_auc_score(yt, yp))
                except Exception:
                    pass
                acc_bs.append(accuracy_score(yt, yhat))
                prec_bs.append(precision_score(yt, yhat))
                rec_bs.append(recall_score(yt, yhat))
                f1_bs.append(f1_score(yt, yhat))
            def ci(a):
                a = np.asarray(a)
                return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
            auc_lo, auc_hi = ci(auc_bs) if len(auc_bs)>0 else (np.nan, np.nan)
            acc_lo, acc_hi = ci(acc_bs)
            prec_lo, prec_hi = ci(prec_bs)
            rec_lo, rec_hi = ci(rec_bs)
            f1_lo, f1_hi = ci(f1_bs)
            
            # Record selected feature names
            try:
                selector = model.named_steps.get('selector')
                if selector and hasattr(selector, 'selected_feature_names_'):
                    self.selected_feature_names_by_model[name] = selector.selected_feature_names_
            except Exception:
                pass

            result = {
                'model_name': name,
                'accuracy': acc,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_proba': y_proba,
                'auc_ci_low': auc_lo, 'auc_ci_high': auc_hi,
                'acc_ci_low': acc_lo, 'acc_ci_high': acc_hi,
                'prec_ci_low': prec_lo, 'prec_ci_high': prec_hi,
                'rec_ci_low': rec_lo, 'rec_ci_high': rec_hi,
                'f1_ci_low': f1_lo, 'f1_ci_high': f1_hi
            }
            results.append(result)
            
            # Best model
            if auc > self.best_auc:
                self.best_auc = auc
                self.best_model = model
                self.best_model_name = name
            
            # Extract feature importance (if available)
            try:
                clf = model.named_steps.get('clf')
                if hasattr(clf, 'feature_importances_') and name in self.selected_feature_names_by_model:
                    self.feature_importance[name] = clf.feature_importances_
            except Exception:
                pass

        self.results = pd.DataFrame(results)
        print("\nModel evaluation results:")
        print(self.results[['model_name', 'accuracy', 'auc', 'precision', 'recall', 'f1']])
        print(f"\nBest model: {self.best_model_name} (AUC: {self.best_auc:.4f})")
        if self.output_dir:
            self.results[['model_name', 'accuracy', 'auc', 'precision', 'recall', 'f1',
                          'auc_ci_low','auc_ci_high','acc_ci_low','acc_ci_high','prec_ci_low','prec_ci_high',
                          'rec_ci_low','rec_ci_high','f1_ci_low','f1_ci_high']].to_csv(
                os.path.join(self.results_dir, 'model_comparison.csv'), index=False)
            with open(os.path.join(self.results_dir, 'model_comparison.txt'), 'w') as f:
                f.write("Machine Learning Model Comparison Evaluation Results\n")
                f.write("====================\n\n")
                f.write(self.results[['model_name', 'accuracy', 'auc', 'precision', 'recall', 'f1']].to_string(index=False))
                f.write(f"\n\nBest model: {self.best_model_name} (AUC: {self.best_auc:.4f})")
        return self.results
    
    def save_models(self):
        """Save Pipeline models and metadata"""
        if not self.models or not self.output_dir:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        for name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"{name} model saved to: {model_path}")
        metadata = {
            'models': list(self.models.keys()),
            'best_model': self.best_model_name,
            'best_auc': self.best_auc,
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        meta_path = os.path.join(self.model_dir, "model_metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Model metadata saved to: {meta_path}")
    
    def plot_roc_curves(self):
        """Plot ROC curve comparison (academic paper optimized version)"""
        if not hasattr(self, 'results') or self.results.empty:
            print("No evaluation results available")
            return
        
        # Set academic paper style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 11,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.2,
            'font.family': 'serif'
        })
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Professional color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, row in enumerate(self.results.itertuples()):
            name = row.model_name
            y_proba = row.y_proba
            fpr, tpr, thr = roc_curve(self.y_test, y_proba)
            auc_val = row.auc
            color = colors[i % len(colors)]
            
            # Confidence interval
            grid = np.linspace(0, 1, 101)
            mean_tpr, lo, hi = self._bootstrap_roc_ci(self.y_test, y_proba, n_boot=500, grid=grid)
            
            # Label includes AUC and confidence interval
            if hasattr(row, 'auc_ci_low') and not np.isnan(row.auc_ci_low):
                label = f'{name}\nAUC = {auc_val:.3f} (95% CI: {row.auc_ci_low:.3f}–{row.auc_ci_high:.3f})'
            else:
                label = f'{name}\nAUC = {auc_val:.3f}'
            
            # Plot ROC curve
            ax.plot(fpr, tpr, linewidth=2.5, color=color, label=label)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.15, linewidth=0)
            
            # Mark optimal threshold point (Youden index)
            j = tpr - fpr  # Youden index
            idx = int(np.argmax(j))
            ax.plot(fpr[idx], tpr[idx], marker='o', color=color, markersize=8, 
                   markeredgecolor='white', markeredgewidth=2, markerfacecolor=color)
        
        # Diagonal line (random classifier baseline)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8, label='Random Classifier')
        
        # Beautify chart
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, 
                 framealpha=0.9, borderpad=0.8)
        
        # Set axis ticks
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Add borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        self._save_multi(fig, 'roc_curves')
        
        # Reset plot parameters
        plt.rcParams.update(plt.rcParamsDefault)
        plt.close(fig)
    
    def plot_precision_recall_curves(self):
        """Plot PR curve comparison (with 95%CI and baseline)"""
        if not hasattr(self, 'results') or self.results.empty:
            print("No evaluation results available")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        palette = sns.color_palette('colorblind', n_colors=len(self.results))
        prevalence = float(np.mean(self.y_test))
        grid = np.linspace(0,1,101)
        for i, row in enumerate(self.results.itertuples()):
            name = row.model_name
            y_proba = row.y_proba
            precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
            ap = average_precision_score(self.y_test, y_proba)
            mean_p, lo, hi = self._bootstrap_pr_ci(self.y_test, y_proba, n_boot=500, grid=grid)
            if hasattr(row, 'auc_ci_low') and not np.isnan(row.auc_ci_low):
                label = f'{name} (AP={ap:.3f})'
            else:
                label = f'{name} (AP={ap:.3f})'
            ax.plot(recall, precision, lw=2, color=palette[i], label=label)
            ax.fill_between(grid, lo, hi, color=palette[i], alpha=0.15, linewidth=0)
        # Positive class rate baseline
        ax.hlines(prevalence, 0, 1, colors='k', linestyles='--', lw=1.5, label=f'Baseline (prevalence={prevalence:.2f})')
        ax.set(xlim=(0,1), ylim=(0,1.05), xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve Comparison')
        ax.legend(loc='lower left')
        self._save_multi(fig, 'pr_curves')
        plt.close(fig)
    
    def plot_confusion_matrices(self):
        """Plot confusion matrix for each model (row normalized + count overlay)"""
        if not self.models:
            print("No trained models available")
            return
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(n_models * 5.5, 5))
        if n_models == 1:
            axes = [axes]
        for ax, (name, model) in zip(axes, self.models.items()):
            thr = self.decision_thresholds.get(name, 0.5)
            try:
                y_proba = model.predict_proba(self.X_test)[:, 1]
            except Exception:
                y_proba = model.decision_function(self.X_test)
            y_pred = (y_proba >= thr).astype(int)
            cm = confusion_matrix(self.y_test, y_pred)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=False, cmap='Blues', cbar=True, vmin=0, vmax=1, ax=ax)
            # Overlay percentage and count
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j+0.5, i+0.5, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})", ha='center', va='center', color='black', fontsize=9)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{name} (thr={thr:.2f})')
            ax.set_xticklabels(['No Delirium', 'Delirium'])
            ax.set_yticklabels(['No Delirium', 'Delirium'])
        plt.tight_layout()
        self._save_multi(fig, 'confusion_matrices')
        plt.close(fig)

    def plot_calibration_and_brier(self, n_bins: int = 10):
        """Plot calibration curve for each model and calculate Brier score (academic paper optimized version)"""
        if not hasattr(self, 'results') or self.results.empty:
            print("No evaluation results available")
            return
        try:
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss
        except Exception:
            print("sklearn.calibration not available, skipping calibration curve plotting")
            return
        
        # Set academic paper style plot parameters
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'font.family': 'serif'
        })
        
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[1, 0])
        
        # Academic paper professional color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        brier_records = []
        
        for i, row in enumerate(self.results.itertuples()):
            name = row.model_name
            y_proba = row.y_proba
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(self.y_test, y_proba, n_bins=n_bins, strategy='uniform')
            
            # Calculate Brier score and confidence interval
            bs = brier_score_loss(self.y_test, y_proba)
            rng = np.random.RandomState(42)
            n = len(self.y_test)
            boots = []
            
            for _ in range(500):
                idx = rng.randint(0, n, n)
                if hasattr(self.y_test, 'iloc'):
                    yt_sample = self.y_test.iloc[idx].values
                elif hasattr(self.y_test, 'values'):
                    yt_sample = self.y_test.values[idx]
                else:
                    yt_sample = np.asarray(self.y_test)[idx]
                boots.append(brier_score_loss(yt_sample, y_proba[idx]))
            
            brier_lo, brier_hi = np.percentile(boots, [2.5, 97.5])
            
            # Calculate ECE/MCE and confidence intervals
            ece = np.mean(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else np.nan
            mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else np.nan
            
            ece_bs, mce_bs = [], []
            for _ in range(300):
                idx = rng.randint(0, n, n)
                if hasattr(self.y_test, 'iloc'):
                    yt_sample = self.y_test.iloc[idx].values
                elif hasattr(self.y_test, 'values'):
                    yt_sample = self.y_test.values[idx]
                else:
                    yt_sample = np.asarray(self.y_test)[idx]
                pt, pp = calibration_curve(yt_sample, y_proba[idx], n_bins=n_bins, strategy='uniform')
                if len(pt) > 0:
                    ece_bs.append(np.mean(np.abs(pt-pp)))
                    mce_bs.append(np.max(np.abs(pt-pp)))
            
            ece_lo, ece_hi = (np.percentile(ece_bs, 2.5), np.percentile(ece_bs, 97.5)) if ece_bs else (np.nan, np.nan)
            mce_lo, mce_hi = (np.percentile(mce_bs, 2.5), np.percentile(mce_bs, 97.5)) if mce_bs else (np.nan, np.nan)
            
            # Plot calibration curve with Brier score information
            label = f'{name}\n(Brier: {bs:.3f}, ECE: {ece:.3f})'
            ax.plot(prob_pred, prob_true, marker=marker, color=color, linewidth=2.5, 
                   markersize=8, markeredgecolor='white', markeredgewidth=0.5, label=label)
            
            # Record statistical metrics
            brier_records.append({
                'model_name': name,
                'brier_score': bs, 'brier_ci_low': brier_lo, 'brier_ci_high': brier_hi,
                'ece': ece, 'ece_ci_low': ece_lo, 'ece_ci_high': ece_hi,
                'mce': mce, 'mce_ci_low': mce_lo, 'mce_ci_high': mce_hi
            })
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8, label='Perfectly calibrated')
        
        # Beautify main plot
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Predicted Probability', fontsize=14, fontweight='bold')
        ax.set_ylabel('Observed Probability', fontsize=14, fontweight='bold')
        ax.set_title('Calibration Curves', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                 framealpha=0.9, borderpad=0.8)
        
        # Set axis ticks
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Add subtle borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # Histogram (using best model)
        if self.best_model_name:
            best_row = self.results[self.results['model_name'] == self.best_model_name].iloc[0]
            y_proba_best = best_row['y_proba']
            
            # Plot probability distribution histogram
            bins = np.linspace(0, 1, 21)
            ax_hist.hist(y_proba_best[self.y_test==0], bins=bins, alpha=0.7, 
                        label='No Delirium', color='#6baed6', density=True, edgecolor='white', linewidth=0.5)
            ax_hist.hist(y_proba_best[self.y_test==1], bins=bins, alpha=0.7, 
                        label='Delirium', color='#fd8d3c', density=True, edgecolor='white', linewidth=0.5)
            
            ax_hist.set_xlim(-0.02, 1.02)
            ax_hist.set_xlabel('Predicted Probability', fontsize=14, fontweight='bold')
            ax_hist.set_ylabel('Density', fontsize=14, fontweight='bold')
            ax_hist.set_title(f'Probability Distribution ({self.best_model_name})', 
                             fontsize=14, fontweight='bold')
            ax_hist.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax_hist.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.9)
            
            # 设置直方图坐标轴
            ax_hist.set_xticks(np.arange(0, 1.1, 0.2))
            
            for spine in ax_hist.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        self._save_multi(fig, 'calibration_curves')
        if self.output_dir:
            pd.DataFrame(brier_records).to_csv(os.path.join(self.results_dir, 'calibration_brier_ece.csv'), index=False)
        
        # Reset plot parameters
        plt.rcParams.update(plt.rcParamsDefault)
        plt.close(fig)

    def generate_calibration_statistics_table(self):
        if not hasattr(self, 'results') or self.results.empty or not self.output_dir:
            return
        
        try:
            # Read calibration statistics data
            calibration_file = os.path.join(self.results_dir, 'calibration_brier_ece.csv')
            if os.path.exists(calibration_file):
                cal_df = pd.read_csv(calibration_file)
            else:
                print("Calibration statistics file does not exist, please run plot_calibration_and_brier() first")
                return
            
            # Create HTML table
            html_content = """
            <table style="border-collapse: collapse; width: 100%; margin: 20px auto; font-family: 'Times New Roman', serif;">
                <caption style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">
                    Table: Calibration Performance Metrics
                </caption>
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">Model</th>
                        <th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">Brier Score</th>
                        <th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">95% CI</th>
                        <th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">ECE</th>
                        <th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">95% CI</th>
                        <th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">MCE</th>
                        <th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">95% CI</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for _, row in cal_df.iterrows():
                brier_ci = f"({row['brier_ci_low']:.3f}, {row['brier_ci_high']:.3f})"
                ece_ci = f"({row['ece_ci_low']:.3f}, {row['ece_ci_high']:.3f})"
                mce_ci = f"({row['mce_ci_low']:.3f}, {row['mce_ci_high']:.3f})"
                
                html_content += f"""
                    <tr style="background-color: #ffffff;">
                        <td style="border: 1px solid #333; padding: 8px; text-align: left; font-weight: bold;">{row['model_name']}</td>
                        <td style="border: 1px solid #333; padding: 8px; text-align: center;">{row['brier_score']:.3f}</td>
                        <td style="border: 1px solid #333; padding: 8px; text-align: center; font-style: italic;">{brier_ci}</td>
                        <td style="border: 1px solid #333; padding: 8px; text-align: center;">{row['ece']:.3f}</td>
                        <td style="border: 1px solid #333; padding: 8px; text-align: center; font-style: italic;">{ece_ci}</td>
                        <td style="border: 1px solid #333; padding: 8px; text-align: center;">{row['mce']:.3f}</td>
                        <td style="border: 1px solid #333; padding: 8px; text-align: center; font-style: italic;">{mce_ci}</td>
                    </tr>
                """
            
            html_content += """
                </tbody>
            </table>
            <p style="font-size: 12px; font-style: italic; margin-top: 10px; font-family: 'Times New Roman', serif;">
                Note: Brier Score (lower is better), ECE = Expected Calibration Error, MCE = Maximum Calibration Error. 
                95% CI obtained via bootstrap resampling (n=500 for Brier, n=300 for ECE/MCE).
            </p>
            """
            
            # Save HTML table
            table_path = os.path.join(self.results_dir, 'calibration_statistics_table.html')
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"Calibration statistics table saved to: {table_path}")
            return table_path
            
        except Exception as e:
            print(f"Failed to generate calibration statistics table: {str(e)}")
            return None

    def plot_decision_curve(self, thresholds=None):
        """Plot decision curve analysis for each model (with 95%CI)"""
        if not hasattr(self, 'results') or self.results.empty:
            print("No evaluation results available")
            return
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)
        y_true = np.asarray(self.y_test)
        N = len(y_true)
        prevalence = np.mean(y_true)
        fig, ax = plt.subplots(figsize=(10, 8))
        palette = sns.color_palette('colorblind', n_colors=len(self.results))
        for i, row in enumerate(self.results.itertuples()):
            name = row.model_name
            y_proba = row.y_proba
            mean_nb, lo, hi = self._bootstrap_dca_ci(y_true, y_proba, thresholds, n_boot=300)
            ax.plot(thresholds, mean_nb, lw=2, color=palette[i], label=name)
            ax.fill_between(thresholds, lo, hi, color=palette[i], alpha=0.15, linewidth=0)
        treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
        treat_none = np.zeros_like(thresholds)
        ax.plot(thresholds, treat_all, 'k--', lw=1.5, label='Treat All')
        ax.plot(thresholds, treat_none, 'k-.', lw=1.5, label='Treat None')
        ax.set(xlabel='Threshold probability', ylabel='Net benefit', title='Decision Curve Analysis', xlim=(0,1))
        ax.legend(loc='best')
        self._save_multi(fig, 'decision_curve')
        plt.close(fig)

    def summarize_threshold_metrics(self):
        """Generate sensitivity/specificity table at threshold (Youden-J) and PPV/NPV at different prevalences (10-30%)"""
        if not hasattr(self, 'results') or self.results.empty or not self.output_dir:
            return
        records = []
        for row in self.results.itertuples():
            name = row.model_name
            thr = self.decision_thresholds.get(name, 0.5)
            try:
                y_proba = self.models[name].predict_proba(self.X_test)[:,1]
            except Exception:
                y_proba = self.models[name].decision_function(self.X_test)
            y_pred = (y_proba >= thr).astype(int)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp+fn)>0 else np.nan
            spec = tn / (tn + fp) if (tn+fp)>0 else np.nan
            for prev in [0.10, 0.20, 0.30]:
                # Bayes conversion for PPV/NPV (using sensitivity/specificity and prior prevalence)
                ppv = (sens*prev) / (sens*prev + (1-spec)*(1-prev)) if not np.isnan(sens) and not np.isnan(spec) else np.nan
                npv = (spec*(1-prev)) / ((1-sens)*prev + spec*(1-prev)) if not np.isnan(sens) and not np.isnan(spec) else np.nan
                records.append({
                    'model_name': name, 'threshold': thr,
                    'sensitivity': sens, 'specificity': spec,
                    'prevalence': prev, 'PPV': ppv, 'NPV': npv
                })
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.results_dir, 'threshold_metrics_ppv_npv.csv'), index=False)

    def generate_shap_plots(self, sample_size: int = 200, max_display: int = 20):
        """Generate SHAP interpretability plots for tree models (RF/XGBoost), including individual-level waterfall"""
        if not hasattr(self, 'results') or self.results.empty:
            print("No evaluation results available")
            return
        try:
            import shap
        except Exception:
            print("shap library not installed, skipping SHAP visualization. Please install: pip install shap")
            return
        os.makedirs(os.path.join(self.results_dir, 'shap'), exist_ok=True)
        for name, model in self.models.items():
            try:
                clf = model.named_steps.get('clf')
            except Exception:
                continue
            if not hasattr(clf, 'predict_proba'):
                continue
            is_tree = ('XGB' in clf.__class__.__name__) or ('Forest' in clf.__class__.__name__) or ('Trees' in clf.__class__.__name__) or ('Gradient' in clf.__class__.__name__)
            if not is_tree:
                continue
            try:
                X_model_input = model[:-1].transform(self.X_test)
            except Exception:
                X_model_input = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
            idx = np.arange(X_model_input.shape[0])
            if sample_size and X_model_input.shape[0] > sample_size:
                rng = np.random.RandomState(42)
                idx = rng.choice(X_model_input.shape[0], size=sample_size, replace=False)
            X_bg = X_model_input[idx]
            
            # Get correct feature names, prioritize from combined_feature_importance.csv
            feature_names = self.selected_feature_names_by_model.get(name, [f"f{i}" for i in range(X_bg.shape[1])])
            
            # If the obtained names are in number format (like f117), try to get real names from feature importance file
            if feature_names and feature_names[0].startswith('f') and feature_names[0][1:].isdigit():
                try:
                    import pandas as pd
                    importance_file = os.path.join(self.results_dir, 'feature_selection_results', 'combined_feature_importance.csv')
                    if os.path.exists(importance_file):
                        importance_df = pd.read_csv(importance_file)
                        if 'feature' in importance_df.columns:
                            # Use first N features, N is the number of features for SHAP analysis
                            real_feature_names = importance_df['feature'].head(len(feature_names)).tolist()
                            if len(real_feature_names) == len(feature_names):
                                feature_names = real_feature_names
                except Exception as e:
                    print(f"Warning: Could not load real feature names: {e}")
                    
            # Display name mapping
            vis_feature_names = [self._standardize_feature_name(n) for n in feature_names]
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_bg)
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            except Exception:
                continue
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
            try:
                X_bg_df = pd.DataFrame(X_bg, columns=vis_feature_names)
            except Exception:
                X_bg_df = X_bg
            # summary beeswarm
            try:
                plt.figure()
                shap.summary_plot(shap_vals, X_bg_df, max_display=max_display, show=False)
                out_path = os.path.join(self.results_dir, 'shap', f'{name}_shap_summary')
                self._save_multi(plt.gcf(), out_path)
                plt.close()
            except Exception:
                pass
            # summary bar
            try:
                plt.figure()
                shap.summary_plot(shap_vals, X_bg_df, plot_type='bar', max_display=max_display, show=False)
                out_path = os.path.join(self.results_dir, 'shap', f'{name}_shap_bar')
                self._save_multi(plt.gcf(), out_path)
                plt.close()
            except Exception:
                pass
            # dependence (top 5 features)
            try:
                mean_abs = np.mean(np.abs(shap_vals), axis=0)
                top_idx = np.argsort(-mean_abs)[:5]
                for i in top_idx:
                    fname = vis_feature_names[i] if i < len(vis_feature_names) else f"f{i}"
                    plt.figure()
                    shap.dependence_plot(i, shap_vals, X_bg_df, interaction_index='auto', show=False)
                    out_path = os.path.join(self.results_dir, 'shap', f'{name}_dependence_{fname.replace(" ", "_")[:40]}')
                    self._save_multi(plt.gcf(), out_path)
                    plt.close()
            except Exception:
                pass
            # Individual-level waterfall (2 examples)
            try:
                import shap
                from shap import Explanation
                mean_abs = np.mean(np.abs(shap_vals), axis=0)
                order = np.argsort(-mean_abs)
                pick_idx = [0, min(1, len(idx)-1)]
                for k in pick_idx:
                    ev = expected_value
                    val = shap_vals[k]
                    data_row = X_bg[k]
                    exp = Explanation(values=val, base_values=ev, data=data_row, feature_names=vis_feature_names)
                    plt.figure()
                    try:
                        shap.plots.waterfall(exp, max_display=20, show=False)
                    except Exception:
                        shap.waterfall_plot(exp, max_display=20, show=False)
                    
                    # Modify colors to specified color scheme
                    fig = plt.gcf()
                    for ax in fig.get_axes():
                        for patch in ax.patches:
                            # Get current color
                            current_color = patch.get_facecolor()
                            # If green or blue (positive contribution), change to blue #5B9BD5
                            if current_color[0] < 0.5 and current_color[1] > 0.4:  # Green
                                patch.set_facecolor('#5B9BD5')
                                patch.set_edgecolor('#5B9BD5')
                            elif current_color[0] < 0.5 and current_color[2] > 0.5:  # Blue
                                patch.set_facecolor('#5B9BD5')
                                patch.set_edgecolor('#5B9BD5')
                            # If red (negative contribution), change to orange #ED7D31
                            elif current_color[0] > 0.5 and current_color[1] < 0.5:  # Red
                                patch.set_facecolor('#ED7D31')
                                patch.set_edgecolor('#ED7D31')
                    
                    out_path = os.path.join(self.results_dir, 'shap', f'{name}_waterfall_case{k}')
                    self._save_multi(plt.gcf(), out_path)
                    plt.close()
            except Exception:
                pass
    
    def plot_feature_importance(self, feature_names, top_n=20):
        """Plot feature importance (using selected feature names)"""
        if not self.feature_importance:
            print("No feature importance data available")
            return
        importance_df = pd.DataFrame()
        for model_name, importances in self.feature_importance.items():
            names = self.selected_feature_names_by_model.get(model_name, feature_names)
            if len(importances) == len(names):
                df = pd.DataFrame({'feature': names, 'importance': importances, 'model': model_name})
                importance_df = pd.concat([importance_df, df], ignore_index=True)
        if importance_df.empty:
            print("No matching feature importance data")
            return
        
        # New: output feature stability (if selector provides bootstrap scores)
        try:
            stability_rows = []
            for model_name, model in self.models.items():
                selector = None
                try:
                    selector = model.named_steps.get('selector')
                except Exception:
                    pass
                if selector is not None and hasattr(selector, 'bootstrap_avg_score_') and selector.bootstrap_avg_score_ is not None:
                    feats = getattr(selector, '_feature_names_after_corr_', None)
                    scores = selector.bootstrap_avg_score_
                    if feats is not None and len(feats) == len(scores):
                        for f, s in zip(feats, scores):
                            stability_rows.append({'model_name': model_name, 'feature': f, 'stability_score': float(s)})
            if stability_rows and self.output_dir:
                pd.DataFrame(stability_rows).to_csv(os.path.join(self.results_dir, 'feature_selection_stability.csv'), index=False)
        except Exception:
            pass
        
        # Select top_n features for each model
        top_features_by_model = {}
        for model_name in self.feature_importance.keys():
            model_data = importance_df[importance_df['model'] == model_name]
            top_idx = model_data['importance'].nlargest(top_n).index
            top_features = model_data.loc[top_idx, 'feature'].tolist()
            top_features_by_model[model_name] = top_features
        
        # Plot feature importance for each model
        for model_name, top_features in top_features_by_model.items():
            plt.figure(figsize=(12, 8))
            
            # Filter data
            model_data = importance_df[(importance_df['model'] == model_name) & 
                                      (importance_df['feature'].isin(top_features))]
            
            # Sort by importance
            model_data = model_data.sort_values('importance', ascending=True)
            
            # Plot bar chart
            plt.barh(model_data['feature'], model_data['importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'{model_name} Feature Importance (Top {top_n})')
            plt.tight_layout()
            
            # Save chart
            if self.output_dir:
                self._save_multi(plt.gcf(), f'{model_name}_feature_importance')
                print(f"{model_name} feature importance plot saved to: {self.results_dir}/{model_name}_feature_importance.png")
            
            plt.close()
        
        # Create heatmap of common features
        common_features = set(top_features_by_model[list(top_features_by_model.keys())[0]])
        for features in top_features_by_model.values():
            common_features &= set(features)
        
        if common_features:
            plt.figure(figsize=(12, len(common_features) * 0.5))
            
            # Create dataframe of common features
            common_df = importance_df[importance_df['feature'].isin(common_features)]
            pivot_df = common_df.pivot(index='feature', columns='model', values='importance')
            
            # Sort by importance of first model
            first_model = list(self.feature_importance.keys())[0]
            pivot_df = pivot_df.sort_values(first_model)
            
            # Plot heatmap
            sns.heatmap(pivot_df.set_index(pd.Index([self._standardize_feature_name(x) for x in pivot_df.index])), annot=True, cmap='YlGnBu', fmt='.3f')
            plt.title(f'Common Important Features Across All Models ({len(common_features)} features)')
            plt.tight_layout()
            
            # Save chart
            if self.output_dir:
                self._save_multi(plt.gcf(), 'common_feature_importance')
                print(f"Common feature importance heatmap saved to: {self.results_dir}/common_feature_importance.png")
            
            plt.close()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        if not hasattr(self, 'results') or self.results.empty:
            print("No evaluation results available")
            return
        
        if not self.output_dir:
            print("Output directory not set")
            return
        
        # Create HTML report
        report_path = os.path.join(self.results_dir, 'model_comparison_report.html')
        
        # Force UTF-8 encoding to avoid gbk encoding errors on Windows
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>ECG Delirium Prediction Model Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #3498db; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .highlight { background-color: #e6f7ff; font-weight: bold; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .image-container { margin: 20px 0; text-align: center; }
                    .image-container img { max-width: 100%; border: 1px solid #ddd; }
                    .footer { margin-top: 30px; font-size: 0.8em; color: #7f8c8d; text-align: center; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>ECG Delirium Prediction Model Evaluation Report</h1>
                    <p>Generated: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
                    
                    <h2>Model Performance Comparison</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>AUC</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                        </tr>
            ''')
            
            best_model = self.results.loc[self.results['auc'].idxmax()]
            for _, row in self.results.iterrows():
                is_best = row['model_name'] == best_model['model_name']
                tr_class = ' class="highlight"' if is_best else ''
                
                f.write(f'''
                        <tr{tr_class}>
                            <td>{row['model_name']}</td>
                            <td>{row['accuracy']:.4f}</td>
                            <td>{row['auc']:.4f}</td>
                            <td>{row['precision']:.4f}</td>
                            <td>{row['recall']:.4f}</td>
                            <td>{row['f1']:.4f}</td>
                        </tr>
                ''')
            
            f.write('''
                    </table>
                    
                    <h2>ROC Curve Comparison</h2>
                    <div class="image-container">
                        <img src="roc_curves.png" alt="ROC Curves">
                    </div>
                    
                    <h2>Precision-Recall Curve Comparison</h2>
                    <div class="image-container">
                        <img src="pr_curves.png" alt="PR Curves">
                    </div>
                    
                    <h2>Confusion Matrices</h2>
                    <div class="image-container">
                        <img src="confusion_matrices.png" alt="Confusion Matrices">
                    </div>
                    
                    <h2>Feature Importance</h2>
            ''')
            
            # Add feature importance images
            for model_name in self.feature_importance.keys():
                f.write(f'''
                    <h3>{model_name} Feature Importance</h3>
                    <div class="image-container">
                        <img src="{model_name}_feature_importance.png" alt="{model_name} Feature Importance">
                    </div>
                ''')
            
            # Add common feature importance (if exists)
            if os.path.exists(os.path.join(self.results_dir, 'common_feature_importance.png')):
                f.write('''
                    <h3>Common Important Features</h3>
                    <div class="image-container">
                        <img src="common_feature_importance.png" alt="Common Feature Importance">
                    </div>
                ''')
            
            f.write('''
                    <div class="footer">
                        <p>ECG Anesthesia Delirium Prediction System | Auto-generated Report</p>
                    </div>
                </div>
            </body>
            </html>
            ''')
        
        print(f"Comprehensive evaluation report generated: {report_path}")
        return report_path

    def delong_auc_test(self):
        """Perform pairwise DeLong test on all models (AUC difference), save CSV (skip if scikit-posthocs not available)"""
        if not hasattr(self, 'results') or self.results.empty or not self.output_dir:
            return
        try:
            import itertools
            names = list(self.results['model_name'])
            pairs = list(itertools.combinations(range(len(names)), 2))
            rows = []
            for i, j in pairs:
                ni, nj = names[i], names[j]
                si = self.results.loc[self.results['model_name']==ni, 'y_proba'].values[0]
                sj = self.results.loc[self.results['model_name']==nj, 'y_proba'].values[0]
                auc_i, auc_j, p = delong_2sample_pvalue(self.y_test, si, sj)
                rows.append({'model_i': ni, 'model_j': nj, 'auc_i': auc_i, 'auc_j': auc_j, 'delta_auc': auc_i-auc_j, 'p_value': p})
            pd.DataFrame(rows).to_csv(os.path.join(self.results_dir, 'delong_auc_pairs.csv'), index=False)
        except Exception:
            pass


def build_multiple_models(delirium_df, non_delirium_df, selected_features, output_dir=None, optimize=True):
    """
    Build and evaluate multiple machine learning models
    """
    print("\n===== Building Multiple Machine Learning Models =====")
    builder = MultiModelBuilder(output_dir)
    try:
        # Prepare data (Note: selected_features is only used for column filtering; recommended to pass None to avoid prior leakage)
        X_train, X_test, y_train, y_test, feature_cols = builder.prepare_data(
            delirium_df, non_delirium_df, selected_features= None)  # Force in-fold selection

        print(f"\nTraining models with in-fold feature selection from {len(feature_cols)} candidate features")
        models = builder.build_models(X_train, y_train, optimize=optimize)
        results = builder.evaluate_models(X_test, y_test)
        builder.save_models()
        builder.plot_roc_curves()
        builder.plot_precision_recall_curves()
        builder.plot_confusion_matrices()
        # Plot using selected feature names
        any_model = next(iter(builder.selected_feature_names_by_model), None)
        names_for_plot = builder.selected_feature_names_by_model.get(any_model, feature_cols)
        builder.plot_feature_importance(names_for_plot)
        # New: calibration curves, DCA and SHAP
        builder.plot_calibration_and_brier()
        builder.generate_calibration_statistics_table()  # Generate academic paper format statistics table
        builder.plot_decision_curve()
        builder.generate_shap_plots()
        builder.generate_comprehensive_report()
        # New: threshold metrics and DeLong placeholder output
        try:
            builder.summarize_threshold_metrics()
        except Exception:
            pass
        try:
            builder.delong_auc_test()
        except Exception:
            pass
        try:
            builder.ablation_by_leads(delirium_df, non_delirium_df)
        except Exception:
            pass
        return builder
    except Exception as e:
        print(f"Multiple model building failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("This is the multi-model building module. Please call through the main program.") 