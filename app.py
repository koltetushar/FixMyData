# Fix My Data - Professional Data Cleaning App
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
from transform import DataTransformer
from profiling import DataProfiler
from visualizer import DataVisualizer

# Init session state
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

st.set_page_config(
    page_title="Fix My Data",
    layout="wide",
    initial_sidebar_state="auto"
)

def init_session_state():
    defaults = {
        'df_raw': None,
        'df_cleaned': None,
        'transformation_report': None,
        'file_hash': None,
        'processing_status': 'idle',
        'cache_hits': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data(ttl=1800, max_entries=3, show_spinner=False)
def load_csv(file_content):
    file_hash = hashlib.md5(file_content).hexdigest()
    df = pd.read_csv(
        BytesIO(file_content),
        encoding='utf-8',
        low_memory=False,
        na_values=['', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A']
    )
    return df.head(10000), file_hash

def process_data(missing_strategy, outlier_method):
    st.session_state.processing_status = 'processing'
    progress = st.sidebar.empty()
    bar = st.sidebar.progress(0)
    try:
        bar.progress(10)
        transformer = DataTransformer()
        bar.progress(30)
        df_cleaned, report = transformer.transform_data(
            st.session_state.df_raw.copy(),
            missing_strategy=missing_strategy,
            outlier_method=outlier_method
        )
        st.session_state.df_cleaned = df_cleaned
        st.session_state.transformation_report = report
        st.session_state.processing_status = 'completed'
        bar.progress(100)
        progress.empty()
        st.rerun()
    except Exception as e:
        st.session_state.processing_status = 'error'
        progress.empty()
        st.error(f"Processing failed: {e}")

def show_sidebar():
    with st.sidebar:
        st.title("Fix My Data")
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")

        if uploaded_file:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large (max 10MB)")
                return
            content = uploaded_file.read()
            df, file_hash = load_csv(content)
            st.session_state.df_raw = df
            st.session_state.file_hash = file_hash
            st.success("File loaded successfully")
            del content
            gc.collect()

        if st.session_state.df_raw is not None:
            st.markdown("---")
            strategy = st.selectbox("Missing Value Strategy", ["mean", "median", "mode", "knn"])
            outlier = st.selectbox("Outlier Detection Method", ["iqr", "zscore"])
            if st.button("Clean My Data"):
                process_data(strategy, outlier)

def show_data_overview():
    df = st.session_state.df_raw
    st.header("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

def show_profiling():
    st.header("Data Profiling Report")
    profiler = DataProfiler()
    profile_report = profiler.generate_profile(st.session_state.df_raw)

    st.subheader("Column Types Distribution")
    if 'column_type_counts' in profile_report:
        fig = px.pie(
            names=list(profile_report['column_type_counts'].keys()),
            values=list(profile_report['column_type_counts'].values())
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missing Value Analysis")
    missing_data = profile_report.get('missing_analysis', {})
    if missing_data:
        missing_df = pd.DataFrame([
            {"Column": k, "Missing": v['count'], "%": v['percentage']}
            for k, v in missing_data.items()
        ])
        st.dataframe(missing_df)
    else:
        st.info("No missing values found.")

def show_cleaned_results():
    df = st.session_state.df_cleaned
    report = st.session_state.transformation_report
    st.header("Cleaned Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cleaned Rows", df.shape[0])
    col2.metric("Remaining Missing", df.isnull().sum().sum())
    col3.metric("Columns", df.shape[1])

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("Download Cleaned CSV", data=csv_buffer.getvalue(), file_name="cleaned_data.csv", mime="text/csv")

    if report:
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            "Download Transformation Report",
            report_json,
            file_name="transformation_report.json",
            mime="application/json"
        )

def show_visualizations():
    st.header("Visualizations")
    visualizer = DataVisualizer()
    charts = visualizer.generate_professional_visualizations(st.session_state.df_cleaned)
    for chart in charts:
        st.subheader(chart['title'])
        st.plotly_chart(chart['figure'], use_container_width=True)

def show_help():
    st.title("Getting Started")
    st.markdown("""
    **Fix My Data** is a tool designed for quick and efficient data cleaning.

    **Steps to Use:**
    1. **Upload your CSV file** using the sidebar.
    2. **Choose how to handle missing values** (mean, median, mode, or KNN).
    3. **Select outlier detection** method (IQR or Z-score).
    4. Click on **"Clean My Data"**.
    5. Review cleaned data and download it.

    **Tips:**
    - Keep files under 10MB for best performance.
    - Use datasets with numeric and categorical values for full feature experience.
    - Check the visualization tab for insights post-cleaning.
    """)

def main():
    init_session_state()
    if time.time() - st.session_state.last_activity > 1800:
        st.session_state.clear()
        st.warning("Session expired. Please reload.")
        st.stop()
    st.session_state.last_activity = time.time()

    show_sidebar()

    if st.session_state.df_raw is not None:
        tabs = st.tabs(["Overview", "Profiling", "Cleaned Data", "Visualizations", "Help"])

        with tabs[0]:
            show_data_overview()
        with tabs[1]:
            show_profiling()
        with tabs[2]:
            if st.session_state.df_cleaned is not None:
                show_cleaned_results()
            else:
                st.info("Run processing to see cleaned data.")
        with tabs[3]:
            if st.session_state.df_cleaned is not None:
                show_visualizations()
            else:
                st.info("Process the dataset to view visualizations.")
        with tabs[4]:
            show_help()
    else:
        st.title("Welcome to Fix My Data")
        st.markdown("Please upload a CSV file to get started.")

if __name__ == '__main__':
    main()
