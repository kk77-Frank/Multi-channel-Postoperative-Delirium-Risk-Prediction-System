#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Selection Module

Provides feature selection functions based on statistical analysis and machine learning to screen for the best feature set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

class FeatureSelector:
    """Feature selection class providing multiple feature selection methods"""
    
    def __init__(self, random_state=42):
        """Initialize feature selector
        
        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        
        # Available feature selection methods
        self.available_methods = [
            'statistical_significance',  # Statistical significance
            'mutual_information',        # Mutual information
            'feature_importance',        # Feature importance
            'correlation_filter',        # Correlation filtering
            'variance_threshold'         # Variance threshold
        ]
    
    def rank_features_by_statistical_significance(self, df_group1, df_group2, alpha=0.05):
        """
        Rank features by statistical significance
        
        Args:
            df_group1: First group data
            df_group2: Second group data
            alpha: Significance level
            
        Returns:
            DataFrame containing features and significance
        """
        try:
            # First exclude non-numeric columns and obvious identifier columns
            exclude_columns = ['file', 'label', 'File', 'Label', 'FILE', 'LABEL']
            # Extract common numeric features
            common_features = []
            
            # Ensure file column is string type to avoid conversion issues
            if 'file' in df_group1.columns:
                df_group1['file'] = df_group1['file'].astype(str)
            if 'file' in df_group2.columns:
                df_group2['file'] = df_group2['file'].astype(str)
                
            for col in df_group1.columns:
                if col in df_group2.columns and col not in exclude_columns:
                    try:
                        # Check if column is numeric type
                        if pd.api.types.is_numeric_dtype(df_group1[col]) and pd.api.types.is_numeric_dtype(df_group2[col]):
                            # Check if there are enough non-missing values
                            if (df_group1[col].notna().sum() > 3) and (df_group2[col].notna().sum() > 3):
                                common_features.append(col)
                    except Exception as e:
                        print(f"Error checking feature {col}: {str(e)}")
                        
            if not common_features:
                print("Warning: No common numeric features found")
                return pd.DataFrame()
            
            # Create result DataFrame
            result = pd.DataFrame({
                'feature': [],
                'group1_mean': [],
                'group2_mean': [],
                'group1_std': [],
                'group2_std': [],
                'statistic': [],
                'p_value': [],
                'significant': [],
                'effect_size': []
            })
            
            # Perform statistical analysis for each feature
            for feature in common_features:
                try:
                    # Get non-missing values and filter out infinite values
                    group1_values = df_group1[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    group2_values = df_group2[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # Calculate basic statistics
                    group1_mean = np.mean(group1_values)
                    group2_mean = np.mean(group2_values)
                    group1_std = np.std(group1_values)
                    group2_std = np.std(group2_values)
                    
                    # Check if data distribution is suitable for parametric test
                    if len(group1_values) > 10 and len(group2_values) > 10:
                        is_parametric = self._check_distribution(group1_values, group2_values)
                    else:
                        # For small samples, default to non-parametric test
                        is_parametric = False
                    
                    if is_parametric:
                        # Use t-test
                        statistic, p_value = stats.ttest_ind(group1_values, group2_values, equal_var=False)
                        # Calculate Cohen's d effect size
                        pooled_std = np.sqrt((group1_std**2 + group2_std**2) / 2)
                        effect_size = abs(group1_mean - group2_mean) / (pooled_std + 1e-10)
                    else:
                        # Use Mann-Whitney U test
                        statistic, p_value = stats.mannwhitneyu(group1_values, group2_values)
                        # Calculate effect size r
                        n1, n2 = len(group1_values), len(group2_values)
                        effect_size = abs(statistic - (n1 * n2 / 2)) / (n1 * n2 + 1e-10)
                    
                    # Determine if significant
                    significant = float(p_value) < alpha if p_value is not None else False
                    
                    # Add to results
                    result = pd.concat([result, pd.DataFrame({
                        'feature': [feature],
                        'group1_mean': [group1_mean],
                        'group2_mean': [group2_mean],
                        'group1_std': [group1_std],
                        'group2_std': [group2_std],
                        'statistic': [statistic],
                        'p_value': [float(p_value) if p_value is not None else np.nan],
                        'significant': [significant],
                        'effect_size': [effect_size]
                    })], ignore_index=True)
                except Exception as e:
                    print(f"Error analyzing feature {feature}: {str(e)}")
            
            # Sort by p-value
            if not result.empty:
                result = result.sort_values(by='p_value')
            
            return result
        
        except Exception as e:
            print(f"Feature statistical analysis error: {str(e)}")
            return pd.DataFrame()
    
    def _check_distribution(self, series1, series2, threshold=0.05):
        """Check if data distribution is suitable for parametric test
        
        Use Shapiro-Wilk test to check normality
        
        Args:
            series1: First group data
            series2: Second group data
            threshold: Significance level
            
        Returns:
            bool: True if both groups follow normal distribution
        """
        try:
            from scipy import stats
            
            # Sample from each series for testing (up to 100 samples max)
            sample1 = series1.sample(min(len(series1), 100)) if len(series1) > 30 else series1
            sample2 = series2.sample(min(len(series2), 100)) if len(series2) > 30 else series2
            
            # Perform Shapiro-Wilk test on each sample
            _, p1 = stats.shapiro(sample1)
            _, p2 = stats.shapiro(sample2)
            
            # If p-value > threshold, data follows normal distribution
            return p1 > threshold and p2 > threshold
        except Exception as e:
            print(f"Normality test failed: {str(e)}")
            return False
    
    def select_by_mutual_information(self, X, y, n_features=10):
        """Select features based on mutual information
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected features
        """
        try:
            # Handle missing values
            X_filled = X.fillna(X.mean())
            
            # Ensure features are non-negative (chi2 requirement)
            min_values = X_filled.min()
            has_negative = (min_values < 0).any()
            
            if has_negative:
                # Use MinMaxScaler to scale data to 0-1 range
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_filled)
                # Use mutual information for feature selection
                selector = SelectKBest(mutual_info_classif, k=n_features)
            else:
                # Use chi2 for feature selection
                X_scaled = X_filled
                selector = SelectKBest(chi2, k=n_features)
            
            # Fit and transform
            selector.fit(X_scaled, y)
            
            # Get feature scores and p-values
            scores = selector.scores_
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'score': scores
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values(by='score', ascending=False)
            
            # Return top n_features features
            return feature_importance.head(n_features)['feature'].tolist()
            
        except Exception as e:
            print(f"Mutual information feature selection failed: {str(e)}")
            return []
    
    def select_by_feature_importance(self, X, y, n_features=10, method='random_forest'):
        """Select features based on feature importance
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            method: Method to use ('random_forest', 'logistic_regression')
            
        Returns:
            List of selected features
        """
        try:
            # Handle missing values
            X_filled = X.fillna(X.mean())
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filled)
            
            # Based on selected method
            if method == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                model.fit(X_scaled, y)
                importance = model.feature_importances_
            elif method == 'logistic_regression':
                model = LogisticRegression(penalty='l2', random_state=self.random_state, max_iter=1000, solver='liblinear')
                model.fit(X_scaled, y)
                importance = np.abs(model.coef_[0])
            elif method == 'permutation_importance':
                # Calculate permutation importance using random forest
                rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                rf.fit(X_scaled, y)
                result = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=self.random_state)
                importance = result.importances_mean if hasattr(result, 'importances_mean') else np.zeros(X.shape[1])
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values(by='importance', ascending=False)
            
            # Return top n_features features
            return feature_importance.head(n_features)['feature'].tolist()
            
        except Exception as e:
            print(f"{method} feature importance calculation failed: {str(e)}")
            return []
    
    def filter_by_correlation(self, X, threshold=0.85):
        """Filter highly correlated features
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold
            
        Returns:
            List of features to keep after filtering
        """
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Extract upper triangle matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find highly correlated features
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            # Return features to keep after filtering
            return [col for col in X.columns if col not in to_drop]
            
        except Exception as e:
            print(f"Correlation filtering failed: {str(e)}")
            return list(X.columns)  # Return all features
    
    def filter_by_variance(self, X, threshold=0.01):
        """Remove low variance features
        
        Args:
            X: Feature matrix
            threshold: Variance threshold
            
        Returns:
            List of features to keep after filtering
        """
        try:
            # Calculate variance of each feature
            variances = X.var()
            
            # Select features with variance greater than threshold
            selected_features = [X.columns[i] for i, var in enumerate(variances) if var > threshold]
            
            return selected_features
            
        except Exception as e:
            print(f"Variance filtering failed: {str(e)}")
            return list(X.columns)  # Return all features
    
    def select_features(self, df_group1, df_group2, method='statistical_significance', n_features=20, alpha=0.05, threshold=0.85):
        """Select features using specified method
        
        Args:
            df_group1: First group data
            df_group2: Second group data
            method: Feature selection method
            n_features: Number of features to select
            alpha: Significance level
            threshold: Correlation or variance threshold
            
        Returns:
            List of selected features
        """
        try:
            # Ensure correct data types
            exclude_columns = ['file', 'label', 'File', 'Label', 'FILE', 'LABEL']
            numeric_cols1 = [col for col in df_group1.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(df_group1[col])]
            numeric_cols2 = [col for col in df_group2.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(df_group2[col])]
            
            # Get common numeric columns from both groups
            common_features = list(set(numeric_cols1) & set(numeric_cols2))
            
            if not common_features:
                print("Warning: No common numeric features found")
                return []
            
            # Based on selected method
            if method == 'statistical_significance':
                # Use statistical significance
                stats_df = self.rank_features_by_statistical_significance(df_group1, df_group2, alpha)
                
                if not stats_df.empty:
                    # Select significant features, maximum n_features
                    significant_features = stats_df[stats_df['significant'] == True]['feature'].tolist()
                    
                    if len(significant_features) > n_features:
                        # Sort by p-value
                        return stats_df.sort_values(by='p_value')['feature'].tolist()[:n_features]
                    elif significant_features:
                        return significant_features
                    else:
                        # If no significant features, return top n_features sorted by p-value
                        return stats_df.sort_values(by='p_value')['feature'].tolist()[:n_features]
                else:
                    print("Statistical analysis returned no results, trying other methods")
                    
            elif method == 'mutual_information':
                # Prepare merged dataset
                X1 = df_group1[common_features]
                X2 = df_group2[common_features]
                X = pd.concat([X1, X2])
                
                # Prepare labels
                y1 = pd.Series(1, index=df_group1.index)
                y2 = pd.Series(0, index=df_group2.index)
                y = pd.concat([y1, y2])
                
                return self.select_by_mutual_information(X, y, n_features)
                
            elif method == 'feature_importance':
                # Prepare merged dataset
                X1 = df_group1[common_features]
                X2 = df_group2[common_features]
                X = pd.concat([X1, X2])
                
                # Prepare labels
                y1 = pd.Series(1, index=df_group1.index)
                y2 = pd.Series(0, index=df_group2.index)
                y = pd.concat([y1, y2])
                
                return self.select_by_feature_importance(X, y, n_features, method='random_forest')
                
            elif method == 'correlation_filter':
                # Merge data
                X = pd.concat([df_group1[common_features], df_group2[common_features]])
                
                # Filter correlated features
                filtered_features = self.filter_by_correlation(X, threshold)
                
                # If too many features after filtering, further filter by variance
                if len(filtered_features) > n_features:
                    X_filtered = X[filtered_features]
                    variance_filtered = self.filter_by_variance(X_filtered)
                    
                    # If still too many, take first n_features
                    if len(variance_filtered) > n_features:
                        return variance_filtered[:n_features]
                    else:
                        return variance_filtered
                else:
                    return filtered_features
                    
            elif method == 'variance_threshold':
                # Merge data
                X = pd.concat([df_group1[common_features], df_group2[common_features]])
                
                # Filter based on variance
                return self.filter_by_variance(X, threshold)
                
            else:
                print(f"Unsupported feature selection method: {method}")
                # Default to statistical significance
                stats_df = self.rank_features_by_statistical_significance(df_group1, df_group2, alpha)
                
                if not stats_df.empty:
                    # Select significant features, maximum n_features
                    significant_features = stats_df[stats_df['significant'] == True]['feature'].tolist()
                    
                    if len(significant_features) > n_features:
                        return significant_features[:n_features]
                    elif significant_features:
                        return significant_features
                    else:
                        # If no significant features, return top n_features by effect size
                        return stats_df.sort_values(by='effect_size', ascending=False)['feature'].tolist()[:n_features]
                else:
                    # If statistical analysis fails, return all common features, maximum n_features
                    return common_features[:n_features]
        except Exception as e:
            print(f"Feature selection failed: {str(e)}")
            # Return all common features, maximum n_features
            return common_features[:min(len(common_features), n_features)]
    
    def plot_feature_importance(self, feature_importance, title='Feature Importance', figsize=(10, 8)):
        """Plot feature importance
        
        Args:
            feature_importance: DataFrame containing 'feature' and 'importance' columns
            title: Chart title
            figsize: Chart size
            
        Returns:
            matplotlib.figure.Figure: Chart object
        """
        try:
            # Sort by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Plot bar chart
            fig, ax = plt.subplots(figsize=figsize)
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            
            # Add labels and title
            ax.set_xlabel('Importance')
            ax.set_title(title)
            ax.invert_yaxis()  # Invert Y-axis to display most important features at the top
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Failed to plot feature importance: {str(e)}")
            return None
