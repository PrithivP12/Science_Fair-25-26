"""
Preprocessing pipeline for redox potential prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

from .schema import QCRules, CRITICAL_FEATURES, FEATURE_TYPES, MISSING_VALUE_POLICY

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Preprocess features for ML models."""
    
    def __init__(
        self,
        missing_strategy: str = 'indicator',
        scaler_type: str = 'robust',
        handle_categorical: bool = True,
    ):
        """
        Args:
            missing_strategy: 'drop', 'mean', 'median', 'indicator', 'tree_friendly'
            scaler_type: 'standard', 'robust', or None
            handle_categorical: Whether to one-hot encode categoricals
        """
        self.missing_strategy = missing_strategy
        self.scaler_type = scaler_type
        self.handle_categorical = handle_categorical
        
        self.scaler = None
        self.imputer = None
        self.feature_names_ = None
        self.categorical_columns_ = []
        self.continuous_columns_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit preprocessor on training data."""
        X = X.copy()
        
        # Identify column types
        self.continuous_columns_ = []
        self.categorical_columns_ = []
        binary_columns = []
        
        for col in X.columns:
            if col in FEATURE_TYPES.get('categorical', []):
                self.categorical_columns_.append(col)
            elif col in FEATURE_TYPES.get('binary', []):
                binary_columns.append(col)
            elif col in FEATURE_TYPES.get('count', []):
                self.continuous_columns_.append(col)
            else:
                # Try to infer
                if X[col].dtype in ['float64', 'int64']:
                    self.continuous_columns_.append(col)
                elif X[col].dtype == 'object':
                    self.categorical_columns_.append(col)
        
        # Handle categoricals
        if self.handle_categorical and self.categorical_columns_:
            X = pd.get_dummies(X, columns=self.categorical_columns_, prefix=self.categorical_columns_)
        
        # Handle missing values for continuous features
        continuous_cols = [c for c in X.columns if c in self.continuous_columns_ or c not in self.categorical_columns_]
        
        if self.missing_strategy == 'tree_friendly':
            # Add missing indicators for high-missing features
            for col in continuous_cols:
                missing_frac = X[col].isna().sum() / len(X)
                if missing_frac > MISSING_VALUE_POLICY['tree_friendly_threshold']:
                    X[f'{col}_missing'] = X[col].isna().astype(int)
                    logger.info(f"Added missing indicator for {col} (missing: {missing_frac:.1%})")
        
        # Impute continuous features
        if self.missing_strategy in ['mean', 'median']:
            strategy = 'mean' if self.missing_strategy == 'mean' else 'median'
            self.imputer = SimpleImputer(strategy=strategy)
            X[continuous_cols] = self.imputer.fit_transform(X[continuous_cols])
        elif self.missing_strategy == 'indicator':
            # Add indicators and impute with 0 (tree models can handle this)
            for col in continuous_cols:
                if X[col].isna().any():
                    X[f'{col}_missing'] = X[col].isna().astype(int)
                    X[col] = X[col].fillna(0)
        elif self.missing_strategy == 'drop':
            X = X.dropna(subset=continuous_cols)
        
        # Scale continuous features
        if self.scaler_type:
            continuous_cols_clean = [c for c in continuous_cols if c in X.columns]
            if self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            elif self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            
            if self.scaler:
                X[continuous_cols_clean] = self.scaler.fit_transform(X[continuous_cols_clean])
        
        self.feature_names_ = list(X.columns)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        X = X.copy()
        
        # Handle categoricals
        if self.handle_categorical and self.categorical_columns_:
            X = pd.get_dummies(X, columns=self.categorical_columns_, prefix=self.categorical_columns_)
            # Ensure all categorical columns from training are present
            for col in self.feature_names_:
                if col not in X.columns and any(cat in col for cat in self.categorical_columns_):
                    X[col] = 0
        
        # Handle missing values
        continuous_cols = [c for c in X.columns if c in self.continuous_columns_ or c not in self.categorical_columns_]
        
        if self.missing_strategy == 'tree_friendly':
            for col in continuous_cols:
                if f'{col}_missing' in self.feature_names_:
                    X[f'{col}_missing'] = X[col].isna().astype(int)
        
        if self.imputer:
            X[continuous_cols] = self.imputer.transform(X[continuous_cols])
        elif self.missing_strategy == 'indicator':
            for col in continuous_cols:
                if f'{col}_missing' in self.feature_names_:
                    X[f'{col}_missing'] = X[col].isna().astype(int)
                    X[col] = X[col].fillna(0)
        elif self.missing_strategy == 'drop':
            X = X.dropna(subset=continuous_cols)
        
        # Scale
        if self.scaler:
            continuous_cols_clean = [c for c in continuous_cols if c in X.columns]
            X[continuous_cols_clean] = self.scaler.transform(X[continuous_cols_clean])
        
        # Ensure feature order matches training
        missing_cols = set(self.feature_names_) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_names_]
        return X


def apply_qc_filters(df: pd.DataFrame, log_rejections: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply QC filters to dataframe.
    
    Returns:
        (passed_df, rejected_df): DataFrames with passed and rejected samples
    """
    passed_indices = []
    rejected_data = []
    
    for idx, row in df.iterrows():
        passes, reasons = QCRules.check_sample(row.to_dict())
        if passes:
            passed_indices.append(idx)
        else:
            rejected_data.append({
                'id': row.get('id', idx),
                'reasons': '; '.join(reasons),
            })
            if log_rejections:
                logger.debug(f"Rejected {row.get('id', idx)}: {reasons}")
    
    passed_df = df.loc[passed_indices].copy()
    rejected_df = pd.DataFrame(rejected_data)
    
    if log_rejections:
        logger.info(f"QC filtering: {len(passed_df)} passed, {len(rejected_df)} rejected")
        if len(rejected_df) > 0:
            reason_counts = rejected_df['reasons'].str.split('; ').explode().value_counts()
            logger.info("Top rejection reasons:")
            for reason, count in reason_counts.head(10).items():
                logger.info(f"  {reason}: {count}")
    
    return passed_df, rejected_df

