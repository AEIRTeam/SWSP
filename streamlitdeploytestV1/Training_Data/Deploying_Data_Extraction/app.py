# app.py
from sheets import display_google_sheets_section, fetch_google_sheets
import streamlit as st
import pandas as pd
import time
import io
from datetime import datetime
import spacy
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your custom extractor
from extractors import RobustPatternExtractor, EnsembleVotingExtractor

# =========================
# SpaCy Model Loader
# =========================
@st.cache_resource
def load_spacy_model():
    """Load spaCy model with fallback."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.warning("No trained spaCy model found. Using basic model.")
        return spacy.blank("en")

# =========================
# Reset Function
# =========================
def reset_app():
    """Reset all session state variables."""
    keys_to_reset = ["results_df", "data_loaded", "current_df", "data_source", "error_log"]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Incident Report Entity Extractor",
    page_icon="📊",
    layout="wide"
)

# =========================
# Session State Initialization
# =========================
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None
if "error_log" not in st.session_state:
    st.session_state.error_log = []

# =========================
# Main App
# =========================
st.title("📊 Incident Report Entity Extractor")
st.write("Extract structured data from unstructured incident reports using pattern matching and machine learning.")

# =========================
# Data Source Selection Section
# =========================
st.header("📁 Data Source")

# Radio button for data source selection
data_source_option = st.radio(
    "Choose your data source:",
    ("File Upload", "Google Sheets"),
    horizontal=True,
    key="data_source_radio"
)

df = None

# =========================
# File Upload Section
# =========================
if data_source_option == "File Upload":
    uploaded_file = st.file_uploader(
        "Upload your CSV file with incident reports",
        type=["csv"],
        help="CSV must contain a 'text' column with incident descriptions"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column")
                st.stop()
            
            # Store in session state
            st.session_state.current_df = df
            st.session_state.data_loaded = True
            st.session_state.data_source = "File Upload"
            
            # Data summary
            valid_rows = df['text'].notna().sum()
            empty_rows = len(df) - valid_rows
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{len(df):,}")
            col2.metric("Valid Rows", f"{valid_rows:,}")
            col3.metric("Empty Rows", f"{empty_rows:,}")
            
            # Preview data
            with st.expander("Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()

# =========================
# Google Sheets Section
# =========================
elif data_source_option == "Google Sheets":
    st.write("Configure your Google Sheets connection:")
    
    col1, col2 = st.columns(2)
    with col1:
        spreadsheet_name = st.text_input(
            "Spreadsheet Name", 
            value="Botpress Chat Output",
            help="Enter the exact name of your Google Spreadsheet"
        )
    with col2:
        worksheet_name = st.text_input(
            "Worksheet Name", 
            value="Sheet2",
            help="Enter the name of the worksheet/tab"
        )
    
    if st.button("📊 Load Google Sheets Data", type="primary", use_container_width=True):
        with st.spinner("Fetching data from Google Sheets..."):
            df_sheets = fetch_google_sheets(spreadsheet_name, worksheet_name)
            
            if df_sheets is not None:
                # Check for required 'text' column
                if "text" not in df_sheets.columns:
                    st.error("Google Sheets data must contain a 'text' column")
                    st.write("Available columns:", list(df_sheets.columns))
                    st.stop()
                
                # Store in session state
                df = df_sheets
                st.session_state.current_df = df
                st.session_state.data_loaded = True
                st.session_state.data_source = "Google Sheets"
                
                # Data summary
                valid_rows = df['text'].notna().sum()
                empty_rows = len(df) - valid_rows
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", f"{len(df):,}")
                col2.metric("Valid Rows", f"{valid_rows:,}")
                col3.metric("Empty Rows", f"{empty_rows:,}")

# Use data from session state if available
if st.session_state.data_loaded and st.session_state.current_df is not None:
    df = st.session_state.current_df

# =========================
# Processing Options Section
# =========================
if df is not None:
    st.header("⚙️ Processing Options")
    
    # Show current data source
    st.info(f"📊 Data Source: **{st.session_state.data_source}** | Rows: **{len(df):,}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        processing_method = st.radio(
            "Processing Method:",
            options=["Enhanced Pattern Matching"],
            index=0,
            help="Enhanced Pattern Matching is recommended for most use cases"
        )
    
    with col2:
        show_progress = st.checkbox("Show detailed progress", value=True)

# =========================
# Processing Section
# =========================
if df is not None:
    st.header("🚀 Start Processing")
    
    if st.button("Process Data", type="primary", use_container_width=True):
        
        # Clear previous error log
        st.session_state.error_log = []
        
        # Initialize extractors
        pattern_extractor = RobustPatternExtractor()
        
        if processing_method == "Ensemble + Pattern Hybrid":
            nlp_model = load_spacy_model()
            ensemble_extractor = EnsembleVotingExtractor(nlp_model)
        
        # Processing setup
        results = []
        total_rows = len(df)
        batch_size = 100  # Fixed batch size
        
        # Progress tracking
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()
            with metrics_container:
                col_m1, col_m2 = st.columns(2)
                processed_metric = col_m1.empty()
                errors_metric = col_m2.empty()
        
        start_time = time.time()
        error_count = 0
        
        # Process data
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:batch_end]
            
            batch_results = []
            
            for idx, row in batch_df.iterrows():
                try:
                    text = str(row["text"]) if pd.notna(row["text"]) else ""
                    
                    if not text.strip():
                        result_row = {
                            "original_index": idx,
                            **{field: "" for field in ['reporter_name', 'person_involved', 'incident_date', 
                                                     'incident_time', 'department', 'incident_description', 
                                                     'location', 'label', 'was_injured', 'injury_description']},
                            "processing_error": "Empty or null text input"
                        }
                        error_count += 1
                        st.session_state.error_log.append({
                            "row_index": idx,
                            "error_type": "Empty Input",
                            "error_message": "Empty or null text input",
                            "text_preview": "N/A"
                        })
                    else:
                        if processing_method == "Enhanced Pattern Matching":
                            final_result = pattern_extractor.extract_comprehensive(text)
                        else:  # Ensemble + Pattern Hybrid
                            ensemble_result, _ = ensemble_extractor.extract_with_voting(text)
                            final_result = ensemble_result
                        
                        result_row = {
                            "original_index": idx,
                            **final_result,
                            "processing_error": ""
                        }
                
                except Exception as e:
                    error_count += 1
                    error_message = str(e)
                    traceback_info = traceback.format_exc()
                    
                    logger.error(f"Error processing row {idx}: {error_message}")
                    logger.error(f"Traceback: {traceback_info}")
                    
                    # Store detailed error info
                    st.session_state.error_log.append({
                        "row_index": idx,
                        "error_type": type(e).__name__,
                        "error_message": error_message,
                        "text_preview": str(row.get("text", ""))[:100] + "..." if len(str(row.get("text", ""))) > 100 else str(row.get("text", "")),
                        "traceback": traceback_info
                    })
                    
                    result_row = {
                        "original_index": idx,
                        **{field: "" for field in ['reporter_name', 'person_involved', 'incident_date', 
                                                 'incident_time', 'department', 'incident_description', 
                                                 'location', 'label', 'was_injured', 'injury_description']},
                        "processing_error": f"{type(e).__name__}: {error_message}"
                    }
                
                batch_results.append(result_row)
            
            results.extend(batch_results)
            
            # Update progress
            if show_progress:
                progress = batch_end / total_rows
                progress_bar.progress(progress)
                
                elapsed_time = time.time() - start_time
                processing_rate = batch_end / elapsed_time if elapsed_time > 0 else 0
                
                status_text.text(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} | Rate: {processing_rate:.1f} rows/sec")
                processed_metric.metric("Processed", f"{batch_end:,}")
                errors_metric.metric("Errors", f"{error_count:,}")
        
        # Store results
        st.session_state.results_df = pd.DataFrame(results)
        
        # Final summary
        total_time = time.time() - start_time
        
        if error_count == 0:
            st.success(f"✅ Processing completed successfully! {total_rows:,} rows processed in {total_time:.1f}s with no errors.")
        elif error_count < total_rows * 0.1:  # Less than 10% errors
            st.warning(f"⚠️ Processing completed with {error_count:,} errors out of {total_rows:,} rows ({(error_count/total_rows)*100:.1f}%)")
        else:
            st.error(f"❌ Processing completed with {error_count:,} errors out of {total_rows:,} rows ({(error_count/total_rows)*100:.1f}%)")

# =========================
# Results Section
# =========================
if st.session_state.results_df is not None:
    st.header("📊 Results")
    
    df_results = st.session_state.results_df
    
    # Summary metrics
    total_processed = len(df_results)
    error_count = (df_results['processing_error'] != "").sum() if 'processing_error' in df_results.columns else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processed", f"{total_processed:,}")
    col2.metric("Processing Errors", f"{error_count:,}")
    col3.metric("Clean Records", f"{total_processed - error_count:,}")
    
    # Field extraction statistics
    st.subheader("Field Extraction Statistics")
    
    desired_fields = ['reporter_name', 'person_involved', 'incident_date', 'incident_time',
                     'department', 'location', 'was_injured', 'label', 'incident_description', 'injury_description']
    
    field_stats = []
    for field in desired_fields:
        if field in df_results.columns:
            # More robust check for meaningful content
            non_empty = (df_results[field].notna() & 
                        (df_results[field] != "") & 
                        (df_results[field] != "None") &
                        (df_results[field].astype(str).str.strip() != ""))
            extracted_count = non_empty.sum()
            blank_count = len(df_results) - extracted_count
            extraction_rate = (extracted_count / len(df_results)) * 100
            blank_rate = (blank_count / len(df_results)) * 100
            
            field_stats.append({
                'Field': field.replace('_', ' ').title(),
                'Extracted': f"{extracted_count:,}",
                'Blank': f"{blank_count:,}",
                'Extraction Rate': f"{extraction_rate:.1f}%",
                'Blank Rate': f"{blank_rate:.1f}%"
            })
    
    if field_stats:
        stats_df = pd.DataFrame(field_stats)
        st.dataframe(stats_df, use_container_width=True)
    
    # Sample results
    st.subheader("Sample Results")
    
    # Show clean results
    if 'processing_error' in df_results.columns:
        clean_results = df_results[df_results['processing_error'] == ""]
    else:
        clean_results = df_results
        
    if len(clean_results) > 0:
        display_cols = [col for col in clean_results.columns 
                       if col not in ['original_index', 'processing_error']]
        st.dataframe(clean_results[display_cols].head(10), use_container_width=True)
    
    # Show processing errors if any
    if error_count > 0:
        with st.expander(f"⚠️ View Processing Errors ({error_count} records)", expanded=False):
            error_df = df_results[df_results['processing_error'] != ""]
            st.dataframe(error_df[['original_index', 'processing_error']].head(10), use_container_width=True)

# =========================
# Error Analysis Section
# =========================
if st.session_state.error_log and len(st.session_state.error_log) > 0:
    st.header("🔍 Error Analysis")
    
    error_df = pd.DataFrame(st.session_state.error_log)
    
    # Error type breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Error Types")
        error_type_counts = error_df['error_type'].value_counts()
        st.bar_chart(error_type_counts)
    
    with col2:
        st.subheader("Error Summary")
        for error_type, count in error_type_counts.items():
            st.metric(error_type, f"{count:,}")
    
    # Detailed error log
    with st.expander("📋 Detailed Error Log", expanded=False):
        st.dataframe(
            error_df[['row_index', 'error_type', 'error_message', 'text_preview']], 
            use_container_width=True
        )
    
    # Download error log
    st.subheader("📥 Download Error Log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_buffer = io.StringIO()
    error_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📊 Download Error Log as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"error_log_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================
# Download Section
# =========================
if st.session_state.results_df is not None:
    st.header("📥 Download Results")
    
    # Prepare download data
    download_df = st.session_state.results_df.copy()
    
    # Remove internal columns but keep processing_error for transparency
    columns_to_remove = ['original_index']
    for col in columns_to_remove:
        if col in download_df.columns:
            download_df = download_df.drop(columns=[col])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_buffer = io.StringIO()
        download_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"incident_extraction_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON download
        json_data = download_df.to_json(orient="records", indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"incident_extraction_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )


# =========================
# Reset Section
# =========================
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("💡 Created by AUT R&D Students 2025")

with col2:
    if st.button("🔄 Reset App", type="secondary", use_container_width=True, help="Clear all data and start fresh"):
        reset_app()

