import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import re
from datetime import datetime

class DataTransformer:
    """
    Handles comprehensive data transformation including:
    - Column name normalization
    - Data type inference
    - Missing value filling
    - Outlier detection & handling
    - Duplicate removal
    """
    
    def __init__(self):
        self.transformation_report = {}
    
    def transform_data(self, df, missing_strategy='mean', outlier_method='iqr'):
        """
        Main transformation pipeline
        
        Args:
            df (pd.DataFrame): Raw dataframe
            missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'knn')
            outlier_method (str): Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        
        Returns:
            tuple: (cleaned_df, transformation_report)
        """
        
        self.transformation_report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'missing_strategy': missing_strategy,
            'outlier_method': outlier_method
        }
        
        # Step 1: Normalize column names
        df = self._normalize_column_names(df)
        
        # Step 2: Infer and optimize data types
        df = self._infer_data_types(df)
        
        # Step 3: Handle missing values
        df = self._handle_missing_values(df, strategy=missing_strategy)
        
        # Step 4: Handle outliers
        df = self._handle_outliers(df, method=outlier_method)
        
        # Step 5: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Final report
        self.transformation_report['final_shape'] = df.shape
        self.transformation_report['rows_removed'] = (
            self.transformation_report['original_shape'][0] - df.shape[0]
        )
        
        return df, self.transformation_report
    
    def _normalize_column_names(self, df):
        """Normalize column names: lowercase, replace spaces/special chars with underscores"""
        original_columns = df.columns.tolist()
        normalized_columns = []
        
        for col in original_columns:
            # Convert to lowercase and replace spaces/special chars with underscores
            normalized = re.sub(r'[^a-zA-Z0-9_]', '_', str(col).lower().strip())
            # Remove multiple consecutive underscores
            normalized = re.sub(r'_+', '_', normalized)
            # Remove leading/trailing underscores
            normalized = normalized.strip('_')
            # Ensure it doesn't start with a number
            if normalized and normalized[0].isdigit():
                normalized = 'col_' + normalized
            # Handle empty names
            if not normalized:
                normalized = f'col_{len(normalized_columns)}'
            
            normalized_columns.append(normalized)
        
        # Handle duplicate column names
        seen = {}
        final_columns = []
        for col in normalized_columns:
            if col in seen:
                seen[col] += 1
                final_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_columns.append(col)
        
        # Track changes
        changes = {orig: new for orig, new in zip(original_columns, final_columns) 
                  if orig != new}
        
        if changes:
            self.transformation_report['columns_renamed'] = changes
        
        df.columns = final_columns
        return df
    
    def _infer_data_types(self, df):
        """Infer and optimize data types for better performance and accuracy"""
        type_changes = {}
        
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            
            # Skip if already optimal
            if df[col].dtype in ['datetime64[ns]', 'category']:
                continue
            
            # Try to convert to datetime
            if self._is_datetime_column(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    type_changes[col] = {'from': original_dtype, 'to': 'datetime64[ns]'}
                    continue
                except:
                    pass
            
            # Try to convert to numeric
            if df[col].dtype == 'object':
                # Try numeric conversion
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if not numeric_col.isna().all():
                    # Check if it should be integer
                    if numeric_col.notna().all() and (numeric_col % 1 == 0).all():
                        df[col] = numeric_col.astype('Int64')  # Nullable integer
                        type_changes[col] = {'from': original_dtype, 'to': 'Int64'}
                    else:
                        df[col] = numeric_col
                        type_changes[col] = {'from': original_dtype, 'to': 'float64'}
                    continue
            
            # Convert to category if low cardinality
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5 and df[col].nunique() < 50:
                    df[col] = df[col].astype('category')
                    type_changes[col] = {'from': original_dtype, 'to': 'category'}
        
        if type_changes:
            self.transformation_report['data_type_changes'] = type_changes
        
        return df
    
    def _is_datetime_column(self, series):
        """Check if a series contains datetime-like values"""
        if series.dtype == 'object':
            # Sample a few non-null values
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            
            # Common datetime patterns
            datetime_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            ]
            
            for value in sample:
                str_value = str(value)
                if any(re.search(pattern, str_value) for pattern in datetime_patterns):
                    return True
        
        return False
    
    def _handle_missing_values(self, df, strategy='mean'):
        """Handle missing values using specified strategy"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                original_missing = missing_count
                
                if strategy == 'mean' and df[col].dtype in ['float64', 'int64', 'Int64']:
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                    missing_info[col] = {
                        'count': original_missing,
                        'method': f'mean ({fill_value:.2f})'
                    }
                
                elif strategy == 'median' and df[col].dtype in ['float64', 'int64', 'Int64']:
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    missing_info[col] = {
                        'count': original_missing,
                        'method': f'median ({fill_value:.2f})'
                    }
                
                elif strategy == 'mode':
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        fill_value = mode_val.iloc[0]
                        df[col] = df[col].fillna(fill_value)
                        missing_info[col] = {
                            'count': original_missing,
                            'method': f'mode ({fill_value})'
                        }
                
                elif strategy == 'knn' and df[col].dtype in ['float64', 'int64', 'Int64']:
                    # Use KNN imputation for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        df_numeric = df[numeric_cols].copy()
                        df_imputed = pd.DataFrame(
                            imputer.fit_transform(df_numeric),
                            columns=numeric_cols,
                            index=df.index
                        )
                        df[col] = df_imputed[col]
                        missing_info[col] = {
                            'count': original_missing,
                            'method': 'KNN (k=5)'
                        }
                
                # For non-numeric columns or when other methods fail, use mode
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        fill_value = mode_val.iloc[0]
                        df[col] = df[col].fillna(fill_value)
                        if col not in missing_info:
                            missing_info[col] = {
                                'count': original_missing,
                                'method': f'mode ({fill_value})'
                            }
        
        if missing_info:
            self.transformation_report['missing_values_filled'] = missing_info
        
        return df
    
    def _handle_outliers(self, df, method='iqr'):
        """Detect and handle outliers using specified method"""
        outlier_info = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            outlier_mask = self._detect_outliers(df[col], method)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outlier_info[col] = outlier_count
                
                # Cap outliers instead of removing (more conservative approach)
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                
                elif method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    
                    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        if outlier_info:
            self.transformation_report['outliers_removed'] = outlier_info
        
        return df
    
    def _detect_outliers(self, series, method='iqr'):
        """Detect outliers using specified method"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            threshold = 3
            outlier_indices = series.dropna().index[z_scores > threshold]
            return series.index.isin(outlier_indices)
        
        elif method == 'isolation_forest':
            if len(series.dropna()) < 10:  # Need minimum samples
                return pd.Series([False] * len(series), index=series.index)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_pred = iso_forest.fit_predict(series.dropna().values.reshape(-1, 1))
            outlier_indices = series.dropna().index[outlier_pred == -1]
            return series.index.isin(outlier_indices)
        
        return pd.Series([False] * len(series), index=series.index)
    
    def _remove_duplicates(self, df):
        """Remove duplicate rows"""
        original_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = original_count - len(df)
        
        if duplicates_removed > 0:
            self.transformation_report['duplicates_removed'] = duplicates_removed
        
        return df