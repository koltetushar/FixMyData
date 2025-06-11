import streamlit as st
import pandas as pd
import json
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our custom modules
from transform import DataTransformer
from profiling import DataProfiler
from visualizer import DataVisualizer

def main():
    st.set_page_config(
        page_title="Fix My Data - Professional Data Analysis Platform",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Fix My Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Professional Data Analysis & Transformation Platform</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'transformation_report' not in st.session_state:
        st.session_state.transformation_report = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration Panel")
        
        # File Upload Section
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader(
            "Select CSV File",
            type=['csv'],
            help="Upload your CSV file to begin analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load the CSV
                df = pd.read_csv(uploaded_file)
                st.session_state.df_raw = df
                st.success("File loaded successfully")
                st.info(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        # Show configuration options only if file is loaded
        if st.session_state.df_raw is not None:
            st.divider()
            
            # Data Cleaning Configuration
            st.subheader("Processing Configuration")
            
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ["mean", "median", "mode", "knn"],
                help="Strategy for handling missing values"
            )
            
            outlier_method = st.selectbox(
                "Outlier Detection Method",
                ["iqr", "zscore", "isolation_forest"],
                help="Method for detecting outliers"
            )
            
            if st.button("Process Dataset", type="primary", use_container_width=True):
                process_data(missing_strategy, outlier_method)
    
    # Main content area
    if st.session_state.df_raw is not None:
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Data Overview", 
            "Data Profiling", 
            "Data Processing", 
            "Analytics & Visualization", 
            "Export Results",
            "Help & Documentation"
        ])
        
        with tab1:
            show_data_overview()
            
        with tab2:
            show_data_profiling()
            
        with tab3:
            show_data_processing()
            
        with tab4:
            show_analytics_visualization()
            
        with tab5:
            show_export_options()
            
        with tab6:
            show_help_documentation()
    
    else:
        # Welcome message when no file is uploaded
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Welcome to Fix My Data
            
            **Professional Data Analysis & Transformation Platform**
            
            **Core Features:**
            - **Data Upload & Validation** - Support for CSV files with automatic format detection
            - **Intelligent Data Profiling** - Automated analysis of data types, patterns, and quality
            - **Smart Data Cleaning** - Handles missing values and outliers
            - **Professional Visualizations** - Charts and statistical summaries
            - **Export & Reporting** - Generate clean datasets and comprehensive transformation reports
            
            **Getting Started:**
            1. Upload your CSV file using the sidebar
            2. Configure processing parameters
            3. Review profiling results and insights
            4. Process and clean your data
            5. Analyze with professional visualizations
            6. Export cleaned data and reports
            """)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>Quick Start Guide</h4>
            <p>Upload a CSV file in the sidebar to begin your data analysis journey.</p>
            <p>For detailed instructions, check the Help & Documentation tab after uploading your data.</p>
            </div>
            """, unsafe_allow_html=True)

def process_data(missing_strategy, outlier_method):
    """Process the uploaded data with selected strategies"""
    with st.spinner("Processing dataset..."):
        try:
            transformer = DataTransformer()
            df_cleaned, report = transformer.transform_data(
                st.session_state.df_raw.copy(),
                missing_strategy=missing_strategy,
                outlier_method=outlier_method
            )
            
            st.session_state.df_cleaned = df_cleaned
            st.session_state.transformation_report = report
            st.success("Dataset processed successfully")
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

def show_data_overview():
    """Show raw data preview and basic information"""
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df_raw
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_count:,}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True, height=400)
    
    # Basic data info
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info, use_container_width=True)

def show_data_profiling():
    """Show comprehensive data profiling with professional styling"""
    st.markdown('<h2 class="section-header">Data Profiling Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_raw is not None:
        profiler = DataProfiler()
        profile_report = profiler.generate_profile(st.session_state.df_raw)
        
        # Column type distribution
        st.subheader("Column Type Distribution")
        
        type_counts = profile_report.get('column_type_counts', {})

        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Type Distribution",
            color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing value analysis
        st.subheader("Missing Value Analysis")
        missing_data = profile_report['missing_analysis']
        
        if missing_data:
            missing_df = pd.DataFrame([
                {"Column": col, "Missing Count": info['count'], "Missing %": info['percentage']}
                for col, info in missing_data.items()
            ])
            
            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing %",
                title="Missing Values by Column",
                text="Missing Count",
                color="Missing %",
                color_continuous_scale="Reds"
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_tickangle=-45,
                font=dict(family="Arial, sans-serif", size=12),
                title_font_size=16
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values detected")
        
        # Data quality warnings
        if profile_report['constant_columns']:
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ Data Quality Alert:</strong> Constant columns detected: {', '.join(profile_report['constant_columns'])}
            </div>
            """, unsafe_allow_html=True)

def show_data_processing():
    """Show data cleaning results and transformation report"""
    st.markdown('<h2 class="section-header">Data Processing Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        # Before/After comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Dataset")
            st.metric("Records", f"{st.session_state.df_raw.shape[0]:,}")
            st.metric("Missing Values", f"{st.session_state.df_raw.isnull().sum().sum():,}")
        
        with col2:
            st.subheader("Processed Dataset")
            st.metric(
                "Records", 
                f"{st.session_state.df_cleaned.shape[0]:,}",
                delta=int(st.session_state.df_cleaned.shape[0] - st.session_state.df_raw.shape[0])
            )
            st.metric(
                "Missing Values", 
                f"{st.session_state.df_cleaned.isnull().sum().sum():,}",
                delta=int(st.session_state.df_cleaned.isnull().sum().sum() - st.session_state.df_raw.isnull().sum().sum())
            )
        
        # Transformation report
        st.subheader("Transformation Summary")
        if st.session_state.transformation_report:
            report = st.session_state.transformation_report
            
            st.info(f"Processing completed: {report.get('timestamp', 'Unknown')}")
            
            # Details in expandable sections
            if report.get('columns_renamed'):
                with st.expander("Column Normalization"):
                    for old, new in report['columns_renamed'].items():
                        st.write(f"• `{old}` → `{new}`")
            
            if report.get('missing_values_filled'):
                with st.expander("Missing Value Treatment"):
                    for col, info in report['missing_values_filled'].items():
                        st.write(f"• **{col}**: {info['count']} values filled using {info['method']}")
            
            if report.get('outliers_removed'):
                with st.expander("Outlier Detection"):
                    for col, count in report['outliers_removed'].items():
                        st.write(f"• **{col}**: {count} outliers detected")
            
            if report.get('duplicates_removed'):
                with st.expander("Duplicate Removal"):
                    st.write(f"• Duplicate records removed: {report['duplicates_removed']}")
        
        # Show processed data preview
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.df_cleaned.head(100), use_container_width=True, height=400)
    
    else:
        st.markdown("""
        <div class="info-box">
        <h4>Processing Required</h4>
        <p>Configure processing parameters in the sidebar and click 'Process Dataset' to view results.</p>
        </div>
        """, unsafe_allow_html=True)

def show_analytics_visualization():
    """Show professional analytics and visualizations"""
    st.markdown('<h2 class="section-header">Analytics & Visualizations</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        visualizer = DataVisualizer()
        charts = visualizer.generate_professional_visualizations(st.session_state.df_cleaned)

        
        # Display charts
        for chart_info in charts:
            st.subheader(chart_info['title'])
            st.plotly_chart(chart_info['figure'], use_container_width=True)
            if 'description' in chart_info:
                st.markdown(f"""
                <div class="info-box">
                <p>{chart_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary statistics
        st.subheader("Statistical Summary")
        numeric_cols = st.session_state.df_cleaned.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(
                st.session_state.df_cleaned[numeric_cols].describe().round(3),
                use_container_width=True
            )
        else:
            st.info("No numeric columns available for statistical analysis.")
    
    else:
        st.markdown("""
        <div class="info-box">
        <h4>Analytics Pending</h4>
        <p>Process your dataset first to access analytics and visualizations.</p>
        </div>
        """, unsafe_allow_html=True)

def show_export_options():
    """Show options to export cleaned data and reports"""
    st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Download Processed Dataset")
            
            # CSV export
            csv_buffer = BytesIO()
            st.session_state.df_cleaned.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Download CSV File",
                data=csv_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the processed dataset",
                use_container_width=True
            )
        
        with col2:
            st.subheader("Download Analysis Report")
            
            # JSON report export
            if st.session_state.transformation_report:
                report_json = json.dumps(st.session_state.transformation_report, indent=2, default=str)
                
                st.download_button(
                    label="Download Report (JSON)",
                    data=report_json,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download detailed transformation report",
                    use_container_width=True
                )
    
    else:
        st.markdown("""
        <div class="info-box">
        <h4>Export Unavailable</h4>
        <p>Process your dataset first to access export options.</p>
        </div>
        """, unsafe_allow_html=True)

def show_help_documentation():
    """Show comprehensive help and documentation"""
    st.markdown('<h2 class="section-header">Help & Documentation</h2>', unsafe_allow_html=True)
    
    # Create tabs for different help sections
    help_tab1, help_tab2, help_tab3, help_tab4 = st.tabs([
        "Getting Started", 
        "Features Overview", 
        "Processing Options", 
        "Best Practices"
    ])
    
    with help_tab1:
        st.markdown("""
        ## Getting Started with Fix My Data
        
        ### Step-by-Step Guide
        
        1. **Upload Your Data**
           - Click "Select CSV File" in the sidebar
           - Choose your CSV file (up to 200MB)
           - Wait for successful upload confirmation
        
        2. **Review Data Overview**
           - Check the "Data Overview" tab
           - Review basic statistics and column information
           - Examine data preview to understand structure
        
        3. **Analyze Data Profile**
           - Visit "Data Profiling" tab
           - Review column type distribution
           - Identify missing values and data quality issues
        
        4. **Configure Processing**
           - Select missing value strategy in sidebar
           - Choose outlier detection method
           - Click "Process Dataset" button
        
        5. **Review Results**
           - Check "Data Processing" tab for transformation summary
           - Analyze visualizations in "Analytics & Visualization"
           - Export cleaned data and reports
        """)
    
    with help_tab2:
        st.markdown("""
        ## Platform Features
        
        ### Data Upload & Validation
        - **Supported Formats**: CSV files
        - **File Size Limit**: 200MB
        - **Automatic Detection**: Column types, encoding, delimiters
        
        ### Data Profiling
        - **Column Classification**: Numeric, categorical, datetime, unknown
        - **Missing Value Analysis**: Count and percentage by column
        - **Data Quality Checks**: Constant columns warnings
        
        ### Data Processing
        - **Missing Value Strategies**:
          - Mean: Replace with column average (numeric only)
          - Median: Replace with column median (numeric only)  
          - Mode: Replace with most frequent value
          - KNN: K-Nearest Neighbors imputation
        
        - **Outlier Detection Methods**:
          - IQR: Interquartile Range method
          - Z-Score: Standard deviation based
          - Isolation Forest: Machine learning approach (Detection only)
        
        ### Visualizations
        - **Distribution Analysis**: Histograms for numeric data
        - **Category Analysis**: Bar charts for categorical data
        - **Correlation Analysis**: Heatmaps for relationships
        - **Statistical Summaries**: Comprehensive descriptive statistics
        
        ### Export Options
        - **Processed Dataset**: Clean CSV file
        - **Analysis Report**: Detailed JSON transformation log
        """)
    
    with help_tab3:
        st.markdown("""
        ## Processing Configuration
        
        ### Missing Value Strategies
        
        **Mean Imputation**
        - Best for: Normally distributed numeric data
        - Preserves: Overall mean of the column
        - Caution: May reduce variance
        
        **Median Imputation**  
        - Best for: Skewed numeric data with outliers
        - Preserves: Central tendency without outlier influence
        - Robust: Less sensitive to extreme values
        
        **Mode Imputation**
        - Best for: Categorical data or discrete numeric
        - Preserves: Most common category/value
        - Simple: Easy to interpret and explain
        
        **KNN Imputation**
        - Best for: Complex patterns in data
        - Advanced: Uses similar records for prediction
        - Resource intensive: Slower for large datasets
        
        ### Outlier Detection Methods
        
        **IQR Method**
        - Formula: Q1 - 1.5×IQR to Q3 + 1.5×IQR
        - Conservative: Detects extreme outliers
        - Interpretable: Easy to understand boundaries
        
        **Z-Score Method**
        - Threshold: Typically |z| > 3 standard deviations
        - Assumption: Normally distributed data
        - Sensitive: To mean and standard deviation
        
        **Isolation Forest**
        - Machine Learning: Ensemble method
        - No assumptions: About data distribution
        - Effective: For high-dimensional data
        """)
    
    with help_tab4:
        st.markdown("""
        ## Best Practices & Tips
        
        ### Data Preparation
        - **Clean File Names**: Use descriptive, simple names without special characters
        - **Consistent Headers**: Ensure column names are clear and consistent
        - **Data Types**: Verify numeric columns contain only numbers
        - **Date Formats**: Use standard formats (YYYY-MM-DD) when possible
        
        ### Processing Strategy Selection
        - **Small Datasets (<1000 rows)**: KNN imputation often works well
        - **Large Datasets (>10000 rows)**: Mean/median/mode for performance
        - **Business Critical**: Always validate results against domain knowledge
        - **Mixed Data Types**: Consider processing different column types separately
        
        ### Quality Assurance
        - **Always Review**: Check processed data before using downstream
        - **Compare Statistics**: Ensure processing didn't change data fundamentally  
        - **Validate Outliers**: Confirm outliers are actual errors, not valid extremes
        - **Document Changes**: Keep transformation reports for audit trails
        
        ### Performance Optimization
        - **Large Files**: Consider sampling for initial exploration
        - **Memory Issues**: Process in chunks if encountering memory errors
        - **Processing Time**: Isolation Forest is slowest, IQR is fastest
        - **Iterative Approach**: Process, review, adjust, repeat as needed
        
        ### Common Issues & Solutions
        
        **File Upload Errors**
        - Check file encoding (UTF-8 recommended)
        - Verify CSV format and delimiter consistency
        - Ensure file size is under 200MB
        
        **Processing Failures**
        - Review data types and column content
        - Check for completely empty columns
        - Verify sufficient data for chosen methods
        
        **Unexpected Results**
        - Compare before/after statistics
        - Review transformation report details
        - Consider different strategy combinations
        """)

if __name__ == "__main__":
    main()