import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataProfiler:
    """
    Professional data profiling with smart column categorization
    """
    
    def __init__(self):
        self.profile_report = {}
        self.column_classifications = {}
    
    def generate_profile(self, df):
        """
        Generate comprehensive data profile
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Complete profiling report
        """
        
        self.profile_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'total_missing': df.isnull().sum().sum(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Column type analysis with categorization
        self._analyze_column_types(df)
        
        # Missing value analysis
        self._analyze_missing_values(df)
        
        # Constant column detection
        self._detect_constant_columns(df)
        
        return self.profile_report
    
    def _analyze_column_types(self, df):
        """Analyze and categorize column types"""
        type_counts = {
            'numerical': 0,
            'categorical': 0,
            'datetime': 0,
            'other': 0
        }
        
        self.column_classifications = {}
        
        for col in df.columns:
            col_type = self._classify_column(df[col])
            self.column_classifications[col] = col_type
            type_counts[col_type] += 1
        
        self.profile_report['column_type_counts'] = type_counts
        self.profile_report['column_classifications'] = self.column_classifications
    
    def _classify_column(self, series):
        """Classify a column into one of four categories"""
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Check if it looks like datetime but stored as string
        if series.dtype == 'object' and self._is_datetime_like(series):
            return 'datetime'
        
        # Check for numerical
        if pd.api.types.is_numeric_dtype(series):
            return 'numerical'
        
        # Try to convert to numerical
        if series.dtype == 'object':
            numeric_converted = pd.to_numeric(series, errors='coerce')
            if not numeric_converted.isna().all():
                return 'numerical'
        
        # Check for categorical (string/object with reasonable cardinality)
        if series.dtype == 'object' or series.dtype.name == 'category':
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.8:  # Less than 80% unique values suggests categorical
                return 'categorical'
        
        # Default to other
        return 'other'
    
    def _is_datetime_like(self, series):
        """Check if a series contains datetime-like strings"""
        if len(series.dropna()) == 0:
            return False
        
        # Sample a few values
        sample = series.dropna().head(10)
        datetime_count = 0
        
        for value in sample:
            str_value = str(value)
            # Common datetime patterns
            patterns = [
                r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
                r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or M/D/YYYY
                r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
                r'\d{4}/\d{1,2}/\d{1,2}',  # YYYY/MM/DD
                r'\d{4}\d{2}\d{2}',        # YYYYMMDD
            ]
            
            if any(re.search(pattern, str_value) for pattern in patterns):
                datetime_count += 1
        
        return datetime_count >= len(sample) * 0.7  # At least 70% match
    
    def _analyze_missing_values(self, df):
        """Analyze missing values across all columns"""
        missing_analysis = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                missing_analysis[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_percentage, 2)
                }
        
        self.profile_report['missing_analysis'] = missing_analysis
    
    def _detect_constant_columns(self, df):
        """Detect columns with constant values"""
        constant_columns = []
        
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        
        self.profile_report['constant_columns'] = constant_columns
    
    def get_column_type_summary(self):
        """Get summary of column types for overview"""
        type_counts = self.profile_report.get('column_type_counts', {})
        return {
            'total_columns': sum(type_counts.values()),
            'breakdown': type_counts,
            'percentages': {
                col_type: round((count / sum(type_counts.values())) * 100, 1) 
                for col_type, count in type_counts.items()
            }
        }
    
    def get_data_quality_metrics(self):
        """Get comprehensive data quality metrics"""
        rows, cols = self.profile_report['dataset_shape']
        
        return {
            'dataset_size': {
                'rows': rows,
                'columns': cols,
                'total_cells': rows * cols
            },
            'missing_data': {
                'total_missing': self.profile_report['total_missing'],
                'missing_percentage': round((self.profile_report['total_missing'] / (rows * cols)) * 100, 2),
                'columns_with_missing': len(self.profile_report.get('missing_analysis', {}))
            },
            'data_types': self.profile_report.get('column_type_counts', {}),
            'constant_columns': len(self.profile_report.get('constant_columns', [])),
            'memory_usage_mb': round(self.profile_report['memory_usage'] / (1024 * 1024), 2)
        }
    
    def get_summary_insights(self):
        """Generate executive summary insights"""
        if not self.profile_report:
            return []
        
        insights = []
        quality_metrics = self.get_data_quality_metrics()
        
        # Dataset overview
        rows = quality_metrics['dataset_size']['rows']
        cols = quality_metrics['dataset_size']['columns']
        insights.append(f"Dataset: {rows:,} rows × {cols} columns ({quality_metrics['memory_usage_mb']} MB)")
        
        # Data quality status
        missing_pct = quality_metrics['missing_data']['missing_percentage']
        if missing_pct == 0:
            insights.append("✅ Data Quality: Excellent (no missing values)")
        elif missing_pct < 5:
            insights.append(f"⚠️ Data Quality: Good ({missing_pct}% missing values)")
        else:
            insights.append(f"❌ Data Quality: Needs attention ({missing_pct}% missing values)")
        
        # Column composition
        type_summary = self.get_column_type_summary()
        breakdown = type_summary['breakdown']
        insights.append(f"Column Types: {breakdown['numerical']} numerical, {breakdown['categorical']} categorical, {breakdown['datetime']} datetime, {breakdown['other']} other")
        
        # Actionable recommendations
        if quality_metrics['constant_columns'] > 0:
            insights.append(f"🔧 Recommendation: Remove {quality_metrics['constant_columns']} constant columns")
        
        if quality_metrics['missing_data']['columns_with_missing'] > 0:
            insights.append(f"🔧 Recommendation: Address missing values in {quality_metrics['missing_data']['columns_with_missing']} columns")
        
        return insights