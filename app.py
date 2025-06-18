import streamlit as st
import pandas as pd
import json
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import time
import gc
# Import our custom modules
from transform import DataTransformer
from profiling import DataProfiler
from visualizer import DataVisualizer


# Initialize 'last_activity' if it doesn't exist
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()


# Configure Streamlit for mobile optimization
st.set_page_config(
    page_title="Fix My Data",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="auto",  # Auto-collapse on mobile
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Fix My Data - Professional Data Analysis Platform"
    }
)

# Mobile-first responsive CSS
MOBILE_CSS = """
<style>
/* Base styles for all devices */
.main-header {
    font-size: clamp(1.8rem, 4vw, 2.5rem);
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
    text-align: center;
}

.subtitle {
    font-size: clamp(0.9rem, 2.5vw, 1.1rem);
    color: #6b7280;
    margin-bottom: 1rem;
    text-align: center;
}

.metric-card {
    background: #f8fafc;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin-bottom: 0.5rem;
}

.section-header {
    font-size: clamp(1.2rem, 3vw, 1.5rem);
    font-weight: 600;
    color: #1f2937;
    margin: 1rem 0 0.5rem 0;
}

.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}

.warning-box {
    background: #fef3c7;
    border: 1px solid #fcd34d;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}

.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-success { background-color: #10b981; }
.status-warning { background-color: #f59e0b; }
.status-error { background-color: #ef4444; }

/* Mobile optimizations */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
    }
    
    .stDataFrame {
        font-size: 0.8rem;
    }
    
    .stMetric {
        text-align: center;
    }
    
    .stButton button {
        width: 100%;
        padding: 0.75rem;
        font-size: 0.9rem;
    }
    
    .stSelectbox label {
        font-size: 0.9rem;
    }
    
    /* Reduce plotly chart heights on mobile */
    .js-plotly-plot {
        max-height: 300px !important;
    }
}

/* Tablet optimizations */
@media (min-width: 769px) and (max-width: 1024px) {
    .js-plotly-plot {
        max-height: 400px !important;
    }
}

/* Loading states */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Reduce animation for performance */
* {
    transition: none !important;
    animation-duration: 0.1s !important;
}
</style>
"""

# Cache configuration for Render free tier
@st.cache_data(ttl=1800, max_entries=3, show_spinner=False)  # 30 minutes TTL, max 3 entries
def load_csv_cached(file_content, file_name):
    """Cache CSV loading to prevent re-reading on every interaction"""
    try:
        # Create a hash of the file content for caching
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Read CSV with optimized settings for mobile
        df = pd.read_csv(
            BytesIO(file_content),
            encoding='utf-8',
            low_memory=False,
            na_values=['', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A'],
            keep_default_na=True
        )
        
        # Limit dataset size for free tier (max 50MB processed data)
        max_rows = 10000  # Reduced for free tier
        if len(df) > max_rows:
            st.warning(f"Dataset truncated to {max_rows:,} rows for performance (originally {len(df):,} rows)")
            df = df.head(max_rows)
        
        return df, file_hash, None
        
    except Exception as e:
        return None, None, str(e)

@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)
def generate_profile_cached(df_hash, df_shape):
    """Cache profiling results"""
    # Reconstruct minimal df info for profiling
    # This is a placeholder - you'd need to modify based on your profiling logic
    return {"cached": True, "timestamp": datetime.now().isoformat()}

@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)
def process_data_cached(df_hash, missing_strategy, outlier_method):
    """Cache data processing results"""
    # This would contain your actual processing logic
    return {"cached": True, "processed_at": datetime.now().isoformat()}

def get_device_type():
    """Detect device type for responsive design"""
    # Basic device detection - in production, you might want more sophisticated detection
    return "mobile" if st.session_state.get('mobile_detected', False) else "desktop"

def init_session_state():
    """Initialize session state with default values"""
    defaults = {
        'df_raw': None,
        'df_cleaned': None,
        'transformation_report': None,
        'file_hash': None,
        'processing_status': 'idle',  # idle, processing, completed, error
        'last_activity': time.time(),
        'mobile_detected': False,
        'cache_hits': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    # Apply mobile-first CSS
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Update last activity timestamp
    st.session_state.last_activity = time.time()
    
    # Header with status indicator
    col_header, col_status = st.columns([4, 1])
    with col_header:
        st.markdown('<h1 class="main-header">🔧 Fix My Data</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Mobile-Optimized Data Analysis Platform</p>', unsafe_allow_html=True)
    
    with col_status:
        status_color = {
            'idle': 'status-warning',
            'processing': 'status-error', 
            'completed': 'status-success',
            'error': 'status-error'
        }.get(st.session_state.processing_status, 'status-warning')
        
        st.markdown(f'<div style="text-align: right; margin-top: 1rem;"><span class="status-indicator {status_color}"></span><small>{st.session_state.processing_status.title()}</small></div>', unsafe_allow_html=True)
    
    # Sidebar with mobile optimization
    with st.sidebar:
        st.header("📁 Upload & Config")
        
        # File Upload with mobile-friendly settings
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader(
            "Select CSV File (Max 10MB)",
            type=['csv'],
            help="Mobile tip: Ensure stable connection before uploading",
            accept_multiple_files=False,
            key="csv_uploader"
        )
        
        # File size limit for Render free tier
        max_file_size = 10 * 1024 * 1024  # 10MB limit for free tier
        
        if uploaded_file is not None:
            # Check file size
            if uploaded_file.size > max_file_size:
                st.error(f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Maximum allowed: 10MB")
                st.stop()
            
            # Show upload progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Reading file...")
                progress_bar.progress(25)
                
                # Read file content
                file_content = uploaded_file.read()
                progress_bar.progress(50)
                
                status_text.text("Processing data...")
                # Use cached loading
                df, file_hash, error = load_csv_cached(file_content, uploaded_file.name)
                progress_bar.progress(75)
                
                if error:
                    st.error(f"Error loading file: {error}")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                if df is not None:
                    st.session_state.df_raw = df
                    st.session_state.file_hash = file_hash
                    st.session_state.processing_status = 'idle'
                    progress_bar.progress(100)
                    
                    # Show success with file info
                    st.success("✅ File loaded successfully")
                    st.info(f"📊 {df.shape[0]:,} rows × {df.shape[1]} columns")
                    
                    # Memory cleanup
                    del file_content
                    gc.collect()
                    
                    time.sleep(0.5)  # Brief pause for UX
                    progress_bar.empty()
                    status_text.empty()
                
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                return
        
        # Configuration options (only show if file loaded)
        if st.session_state.df_raw is not None:
            st.divider()
            st.subheader("⚙️ Processing Config")
            
            # Mobile-friendly selectboxes
            missing_strategy = st.selectbox(
                "Missing Values",
                ["mean", "median", "mode", "knn"],  # Removed KNN for performance on free tier
                help="How to handle missing values",
                key="missing_strategy"
            )
            
            outlier_method = st.selectbox(
                "Outlier Detection", 
                ["iqr", "zscore"],  # Removed isolation_forest for performance
                help="How to detect outliers",
                key="outlier_method"
            )
            
            # Processing button with loading state
            if st.session_state.processing_status == 'processing':
                st.button("🔄 Processing...", disabled=True, use_container_width=True)
            else:
                if st.button("🚀 Process Dataset", type="primary", use_container_width=True):
                    process_data_optimized(missing_strategy, outlier_method)
    
    # Main content with responsive tabs
    if st.session_state.df_raw is not None:
        # Mobile-optimized tab layout
        tab_names = ["📊 Overview", "🔍 Profile", "⚡ Process", "📈 Charts", "💾 Export", "❓ Help"]
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            show_data_overview_mobile()
            
        with tabs[1]:
            show_data_profiling_mobile()
            
        with tabs[2]:
            show_data_processing_mobile()
            
        with tabs[3]:
            show_analytics_mobile()
            
        with tabs[4]:
            show_export_mobile()
            
        with tabs[5]:
            show_help_mobile()
    
    else:
        show_welcome_screen()
    
    # Footer with cache info (for debugging)
    if st.session_state.get('cache_hits', 0) > 0:
        st.caption(f"⚡ Cache hits: {st.session_state.cache_hits} | Last activity: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_activity))}")

def process_data_optimized(missing_strategy, outlier_method):
    """Optimized data processing with progress tracking"""
    st.session_state.processing_status = 'processing'
    
    # Create progress tracking
    progress_container = st.sidebar.empty()
    with progress_container.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        status_text.text("Initializing processor...")
        progress_bar.progress(10)
        
        transformer = DataTransformer()
        
        status_text.text("Processing missing values...")
        progress_bar.progress(40)
        
        # Process data with error handling
        df_cleaned, report = transformer.transform_data(
            st.session_state.df_raw.copy(),
            missing_strategy=missing_strategy,
            outlier_method=outlier_method
        )
        
        progress_bar.progress(80)
        status_text.text("Finalizing results...")
        
        # Store results
        st.session_state.df_cleaned = df_cleaned
        st.session_state.transformation_report = report
        st.session_state.processing_status = 'completed'
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        time.sleep(1)  # Brief pause to show completion
        progress_container.empty()
        
        # Force garbage collection
        gc.collect()
        
        st.rerun()  # Refresh to show results
        
    except Exception as e:
        st.session_state.processing_status = 'error'
        progress_container.empty()
        st.error(f"Processing failed: {str(e)}")

def show_data_overview_mobile():
    """Mobile-optimized data overview"""
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df_raw
    
    # Mobile-friendly metrics in 2x2 grid
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Records", f"{df.shape[0]:,}")
        st.metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col2:
        st.metric("📋 Columns", f"{df.shape[1]:,}")
        missing_count = df.isnull().sum().sum()
        st.metric("❌ Missing", f"{missing_count:,}")
    
    # Responsive data preview
    st.subheader("📝 Data Preview")
    
    # Mobile: show fewer rows, desktop: show more
    display_rows = 20 if get_device_type() == "mobile" else 50
    
    st.dataframe(
        df.head(display_rows), 
        use_container_width=True, 
        height=300 if get_device_type() == "mobile" else 400
    )
    
    # Collapsible column info
    with st.expander("📋 Column Details"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null': df.isnull().sum(),
            'Unique': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)

def show_data_profiling_mobile():
    """Mobile-optimized data profiling"""
    st.markdown('<h2 class="section-header">🔍 Data Profile</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_raw is not None:
        with st.spinner("Analyzing data..."):
            profiler = DataProfiler()
            profile_report = profiler.generate_profile(st.session_state.df_raw)
        
        # Column types - mobile-friendly pie chart
        st.subheader("📊 Column Types")
        type_counts = profile_report.get('column_type_counts', {})
        
        if type_counts:
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                height=300 if get_device_type() == "mobile" else 400
            )
            fig.update_layout(
                font_size=10 if get_device_type() == "mobile" else 12,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing values analysis
        st.subheader("❌ Missing Values")
        missing_data = profile_report['missing_analysis']
        
        if missing_data:
            missing_df = pd.DataFrame([
                {"Column": col, "Count": info['count'], "Percentage": info['percentage']}
                for col, info in missing_data.items()
            ])
            
            # Mobile-friendly bar chart
            fig = px.bar(
                missing_df,
                x="Column",
                y="Percentage",
                text="Count",
                height=300 if get_device_type() == "mobile" else 400
            )
            fig.update_layout(
                xaxis_tickangle=-45 if get_device_type() == "desktop" else -90,
                font_size=10 if get_device_type() == "mobile" else 12
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected")

def show_data_processing_mobile():
    """Mobile-optimized processing results"""
    st.markdown('<h2 class="section-header">⚡ Processing Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        # Before/After comparison - mobile stacked layout
        if get_device_type() == "mobile":
            st.subheader("📊 Before")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", f"{st.session_state.df_raw.shape[0]:,}")
            with col2:
                st.metric("Missing", f"{st.session_state.df_raw.isnull().sum().sum():,}")
            
            st.subheader("✨ After")
            col3, col4 = st.columns(2)
            with col3:
                st.metric(
                    "Records", 
                    f"{st.session_state.df_cleaned.shape[0]:,}",
                    delta=int(st.session_state.df_cleaned.shape[0] - st.session_state.df_raw.shape[0])
                )
            with col4:
                st.metric(
                    "Missing", 
                    f"{st.session_state.df_cleaned.isnull().sum().sum():,}",
                    delta=int(st.session_state.df_cleaned.isnull().sum().sum() - st.session_state.df_raw.isnull().sum().sum())
                )
        else:
            # Desktop side-by-side layout
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Original")
                st.metric("Records", f"{st.session_state.df_raw.shape[0]:,}")
                st.metric("Missing", f"{st.session_state.df_raw.isnull().sum().sum():,}")
            
            with col2:
                st.subheader("✨ Processed")
                st.metric(
                    "Records", 
                    f"{st.session_state.df_cleaned.shape[0]:,}",
                    delta=int(st.session_state.df_cleaned.shape[0] - st.session_state.df_raw.shape[0])
                )
                st.metric(
                    "Missing", 
                    f"{st.session_state.df_cleaned.isnull().sum().sum():,}",
                    delta=int(st.session_state.df_cleaned.isnull().sum().sum() - st.session_state.df_raw.isnull().sum().sum())
                )
        
        # Transformation details in expandable sections
        if st.session_state.transformation_report:
            report = st.session_state.transformation_report
            
            st.markdown(f"""
            <div class="info-box">
            ✅ <strong>Processing completed:</strong> {report.get('timestamp', 'Unknown')}
            </div>
            """, unsafe_allow_html=True)
            
            # Collapsible report sections
            if report.get('missing_values_filled'):
                with st.expander("🔧 Missing Values Fixed"):
                    for col, info in report['missing_values_filled'].items():
                        st.write(f"• **{col}**: {info['count']} values filled using {info['method']}")
            
            if report.get('outliers_removed'):
                with st.expander("🎯 Outliers Detected"):
                    for col, count in report['outliers_removed'].items():
                        st.write(f"• **{col}**: {count} outliers found")
        
        # Processed data preview
        st.subheader("📝 Processed Data")
        display_rows = 20 if get_device_type() == "mobile" else 50
        st.dataframe(
            st.session_state.df_cleaned.head(display_rows), 
            use_container_width=True, 
            height=300 if get_device_type() == "mobile" else 400
        )
    
    else:
        st.markdown("""
        <div class="info-box">
        <h4>⚙️ Processing Required</h4>
        <p>Configure settings in the sidebar and click 'Process Dataset'</p>
        </div>
        """, unsafe_allow_html=True)

def show_analytics_mobile():
    """Mobile-optimized analytics"""
    st.markdown('<h2 class="section-header">📈 Analytics</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        with st.spinner("Generating charts..."):
            visualizer = DataVisualizer()
            charts = visualizer.generate_professional_visualizations(st.session_state.df_cleaned)
        
        # Display charts with mobile optimization
        for chart_info in charts:
            st.subheader(chart_info['title'])
            
            # Adjust chart height for mobile
            if hasattr(chart_info['figure'], 'update_layout'):
                chart_info['figure'].update_layout(
                    height=300 if get_device_type() == "mobile" else 400,
                    font_size=10 if get_device_type() == "mobile" else 12
                )
            
            st.plotly_chart(chart_info['figure'], use_container_width=True)
        
        # Statistical summary in expandable section
        with st.expander("📊 Statistical Summary"):
            numeric_cols = st.session_state.df_cleaned.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(
                    st.session_state.df_cleaned[numeric_cols].describe().round(3),
                    use_container_width=True
                )
            else:
                st.info("No numeric columns for statistics")
    
    else:
        st.markdown("""
        <div class="info-box">
        <h4>📊 Analytics Pending</h4>
        <p>Process your dataset first to see visualizations</p>
        </div>
        """, unsafe_allow_html=True)

def show_export_mobile():
    """Mobile-optimized export options"""
    st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        # Mobile-friendly export layout
        st.subheader("📥 Download Options")
        
        # CSV export
        csv_buffer = BytesIO()
        st.session_state.df_cleaned.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"processed_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.session_state.transformation_report:
                report_json = json.dumps(st.session_state.transformation_report, indent=2, default=str)
                st.download_button(
                    label="📋 Download Report",
                    data=report_json,
                    file_name=f"report_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Export summary
        st.markdown(f"""
        <div class="info-box">
        <strong>📊 Export Summary:</strong><br>
        • Processed dataset: {st.session_state.df_cleaned.shape[0]:,} rows<br>
        • File size: ~{len(csv_data) / 1024:.1f} KB<br>
        • Format: CSV (UTF-8)
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="info-box">
        <h4>💾 Export Unavailable</h4>
        <p>Process your dataset first to enable exports</p>
        </div>
        """, unsafe_allow_html=True)

def show_help_mobile():
    """Mobile-optimized help section"""
    st.markdown('<h2 class="section-header">❓ Help & Tips</h2>', unsafe_allow_html=True)
    
    # Mobile-friendly accordion-style help
    with st.expander("🚀 Quick Start"):
        st.markdown("""
        **5 Simple Steps:**
        1. 📁 Upload CSV file (max 10MB)
        2. ⚙️ Choose processing options
        3. 🚀 Click 'Process Dataset'
        4. 📊 Review results & charts
        5. 💾 Download cleaned data
        """)
    
    with st.expander("📱 Mobile Tips"):
        st.markdown("""
        **For Best Mobile Experience:**
        • Use stable WiFi for uploads
        • Files under 5MB work best
        • Rotate to landscape for charts
        • Use swipe gestures in tables
        • Check processing status indicator
        """)
    
    with st.expander("⚙️ Processing Options"):
        st.markdown("""
        **Missing Values:**
        • **Mean**: Average for numbers
        • **Median**: Middle value (good for outliers)
        • **Mode**: Most common value
        
        **Outlier Detection:**
        • **IQR**: Conservative detection
        • **Z-Score**: Standard deviation based
        """)
    
    with st.expander("🚨 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        • **Upload fails**: Check file size (<10MB)
        • **Processing errors**: Verify CSV format
        • **Slow performance**: Try smaller dataset
        • **Charts not loading**: Refresh page
        
        **Performance Tips:**
        • Close other browser tabs
        • Use latest browser version
        • Clear browser cache if needed
        """)
    
    with st.expander("🔧 Technical Limits"):
        st.markdown("""
        **Free Tier Limitations:**
        • Max file size: 10MB
        • Max rows processed: 10,000
        • Session timeout: 30 minutes
        • Cache duration: 30 minutes
        
        **Supported Formats:**
        • CSV files only
        • UTF-8 encoding recommended
        """)

def show_welcome_screen():
    """Mobile-optimized welcome screen"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
    <h2>👋 Welcome to Fix My Data</h2>
    <p>Mobile-optimized data analysis platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🚀 Key Features:**
        • Mobile-first design
        • Smart data cleaning
        • Professional charts
        • One-click export
        """)
    
    with col2:
        st.markdown("""
        **⚡ Optimized for:**
        • Render free hosting
        • Mobile devices
        • Fast processing
        • Low memory usage
        """)
    
    st.markdown("""
    <div class="info-box" style="margin: 2rem 0; text-align: center;">
    <h4>🎯 Ready to Start?</h4>
    <p>Upload your CSV file using the sidebar to begin analysis</p>
    <small>Maximum file size: 10MB • Supports mobile uploads</small>
    </div>
    """, unsafe_allow_html=True)

# Additional utility functions for mobile optimization
def detect_mobile_browser():
    """Basic mobile detection"""
    try:
        # This is a basic implementation - in production you might want more sophisticated detection
        return False  # Streamlit doesn't provide direct access to user agent
    except:
        return False

def cleanup_memory():
    """Force garbage collection to free memory"""
    import gc
    gc.collect()

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def validate_csv_structure(df):
    """Validate CSV structure for common issues"""
    issues = []
    
    # Check for empty DataFrame
    if df.empty:
        issues.append("Dataset is empty")
    
    # Check for columns with all null values
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        issues.append(f"Columns with all null values: {', '.join(null_columns)}")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        issues.append("Duplicate column names detected")
    
    # Check for very wide datasets (performance concern)
    if df.shape[1] > 100:
        issues.append(f"Dataset has {df.shape[1]} columns - may impact performance")
    
    return issues

def create_performance_monitor():
    """Create a simple performance monitor for debugging"""
    if 'performance_log' not in st.session_state:
        st.session_state.performance_log = []
    
    def log_performance(action, start_time):
        duration = time.time() - start_time
        st.session_state.performance_log.append({
            'action': action,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 entries
        if len(st.session_state.performance_log) > 10:
            st.session_state.performance_log = st.session_state.performance_log[-10:]
    
    return log_performance

# Enhanced error handling
def safe_execute(func, error_message="An error occurred", *args, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        error_details = f"{error_message}: {str(e)}"
        st.error(error_details)
        return None, error_details

# Session management for Render free tier
def check_session_health():
    """Check if session is healthy and clean up if needed"""
    current_time = time.time()
    
    # Check for session timeout (30 minutes)
    if current_time - st.session_state.last_activity > 1800:  # 30 minutes
        st.session_state.clear()
        st.warning("Session expired due to inactivity. Please refresh the page.")
        return False
    
    # Check memory usage and cleanup if needed
    if len(str(st.session_state)) > 1000000:  # ~1MB session size limit
        cleanup_memory()
        st.info("Session cleaned up to optimize performance")
    
    return True

# Mobile-specific data display functions
def display_dataframe_mobile(df, max_rows=20):
    """Display DataFrame optimized for mobile"""
    if get_device_type() == "mobile":
        # Show fewer columns on mobile
        if len(df.columns) > 5:
            selected_cols = st.multiselect(
                "Select columns to display:",
                df.columns.tolist(),
                default=df.columns[:5].tolist(),
                key="mobile_columns"
            )
            if selected_cols:
                df_display = df[selected_cols]
            else:
                df_display = df.iloc[:, :5]  # First 5 columns
        else:
            df_display = df
        
        # Truncate long text for mobile
        df_display = df_display.head(max_rows)
        
        return st.dataframe(
            df_display,
            use_container_width=True,
            height=300
        )
    else:
        return st.dataframe(df.head(max_rows), use_container_width=True, height=400)

# Configuration for different deployment environments
def get_deployment_config():
    """Get configuration based on deployment environment"""
    # Check if running on Render (you can set an environment variable)
    import os
    
    if os.getenv('RENDER'):
        return {
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'max_rows': 10000,
            'cache_ttl': 1800,  # 30 minutes
            'max_cache_entries': 3,
            'enable_analytics': False,  # Disable analytics on free tier
            'optimize_for_mobile': True
        }
    else:
        # Local development or other hosting
        return {
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            'max_rows': 50000,
            'cache_ttl': 3600,  # 1 hour
            'max_cache_entries': 10,
            'enable_analytics': True,
            'optimize_for_mobile': False
        }

# Health check endpoint for Render
def health_check():
    """Simple health check for monitoring"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage': f"{st.session_state.get('memory_usage', 0)} MB",
        'active_sessions': 1,  # Streamlit handles this
        'cache_size': len(st.session_state)
    }

if __name__ == "__main__":
    # Check session health before starting
    if check_session_health():
        main()
    else:
        st.error("Session health check failed. Please refresh the page.")
        st.stop()
