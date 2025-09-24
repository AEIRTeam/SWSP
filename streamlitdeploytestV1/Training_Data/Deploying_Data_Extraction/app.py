# app.py
from sheets import display_google_sheets_section, fetch_google_sheets
import streamlit as st
import pandas as pd
import time
import io
from datetime import datetime
import spacy
import logging

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
    keys_to_reset = ["results_df", "data_loaded", "current_df", "data_source"]
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
            "Select Processing Method:",
            # options=["Enhanced Pattern Matching", "Ensemble + Pattern Hybrid"],
            options=["Enhanced Pattern Matching"],
            index=0,
            help="Enhanced Pattern Matching is recommended for most use cases"
        )
    
    with col2:
        batch_size = st.slider("Batch Size", min_value=10, max_value=500, value=100)
        show_progress = st.checkbox("Show detailed progress", value=True)

# =========================
# Processing Section
# =========================
if df is not None:
    st.header("🚀 Start Processing")
    
    if st.button("Process Data", type="primary", use_container_width=True):
        
        # Initialize extractors
        pattern_extractor = RobustPatternExtractor()
        
        if processing_method == "Ensemble + Pattern Hybrid":
            nlp_model = load_spacy_model()
            ensemble_extractor = EnsembleVotingExtractor(nlp_model)
        
        # Processing setup
        results = []
        total_rows = len(df)
        
        # Progress tracking
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()
            with metrics_container:
                col_m1, col_m2, col_m3 = st.columns(3)
                processed_metric = col_m1.empty()
                success_metric = col_m2.empty()
                rate_metric = col_m3.empty()
        
        start_time = time.time()
        successful_extractions = 0
        
        # Process data
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:batch_end]
            
            batch_results = []
            batch_successes = 0
            
            for idx, row in batch_df.iterrows():
                try:
                    text = str(row["text"]) if pd.notna(row["text"]) else ""
                    
                    if not text.strip():
                        result_row = {
                            "original_index": idx,
                            **{field: "" for field in ['reporter_name', 'person_involved', 'incident_date', 
                                                     'incident_time', 'department', 'incident_description', 
                                                     'location', 'label', 'was_injured', 'injury_description']},
                            "error": "Empty text"
                        }
                    else:
                        if processing_method == "Enhanced Pattern Matching":
                            final_result = pattern_extractor.extract_comprehensive(text)
                        else:  # Ensemble + Pattern Hybrid
                            ensemble_result, _ = ensemble_extractor.extract_with_voting(text)
                            final_result = ensemble_result
                        
                        # Count successful extractions
                        filled_fields = sum(1 for v in final_result.values() if v and str(v).strip())
                        if filled_fields >= 3:
                            batch_successes += 1
                        
                        result_row = {
                            "original_index": idx,
                            **final_result
                        }
                
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    result_row = {
                        "original_index": idx,
                        **{field: "" for field in ['reporter_name', 'person_involved', 'incident_date', 
                                                 'incident_time', 'department', 'incident_description', 
                                                 'location', 'label', 'was_injured', 'injury_description']},
                        "error": str(e)
                    }
                
                batch_results.append(result_row)
            
            results.extend(batch_results)
            successful_extractions += batch_successes
            
            # Update progress
            if show_progress:
                progress = batch_end / total_rows
                progress_bar.progress(progress)
                
                elapsed_time = time.time() - start_time
                processing_rate = batch_end / elapsed_time if elapsed_time > 0 else 0
                
                status_text.text(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}")
                processed_metric.metric("Processed", f"{batch_end:,}")
                success_metric.metric("Success Rate", f"{(successful_extractions/batch_end)*100:.1f}%")
                rate_metric.metric("Rate (rows/sec)", f"{processing_rate:.1f}")
        
        # Store results
        st.session_state.results_df = pd.DataFrame(results)
        
        # Final summary
        total_time = time.time() - start_time
        final_success_rate = (successful_extractions / total_rows) * 100
        
        if final_success_rate > 70:
            st.success(f"Processing completed! {total_rows:,} rows processed in {total_time:.1f}s with {final_success_rate:.1f}% success rate")
        elif final_success_rate > 40:
            st.warning(f"Processing completed with {final_success_rate:.1f}% success rate")
        else:
            st.error(f"Processing completed with low success rate: {final_success_rate:.1f}%")

# =========================
# Results Section
# =========================
if st.session_state.results_df is not None:
    st.header("📊 Results")
    
    df_results = st.session_state.results_df
    
    # Summary metrics
    total_processed = len(df_results)
    error_count = df_results['error'].notna().sum() if 'error' in df_results.columns else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processed", f"{total_processed:,}")
    col2.metric("Errors", f"{error_count:,}")
    col3.metric("Clean Records", f"{total_processed - error_count:,}")
    
    # Field success rates
    st.subheader("Field Extraction Success Rates")
    
    desired_fields = ['reporter_name', 'person_involved', 'incident_date', 'incident_time',
                     'department', 'location', 'was_injured', 'label']
    
    field_stats = []
    for field in desired_fields:
        if field in df_results.columns:
            # More robust check for meaningful content
            non_empty = (df_results[field].notna() & 
                        (df_results[field] != "") & 
                        (df_results[field] != "None") &
                        (df_results[field].astype(str).str.strip() != ""))
            success_count = non_empty.sum()
            success_rate = (success_count / len(df_results)) * 100
            field_stats.append({
                'Field': field.replace('_', ' ').title(),
                'Success Rate': f"{success_rate:.1f}%",
                'Count': f"{success_count:,}"
            })
    
    if field_stats:
        stats_df = pd.DataFrame(field_stats)
        st.dataframe(stats_df, use_container_width=True)
    
    # Sample results
    st.subheader("Sample Results")
    
    # Show clean results
    if 'error' in df_results.columns:
        clean_results = df_results[df_results['error'].isna()]
    else:
        clean_results = df_results
        
    if len(clean_results) > 0:
        display_cols = [col for col in clean_results.columns 
                       if col not in ['original_index', 'error']]
        st.dataframe(clean_results[display_cols].head(10), use_container_width=True)
    
    # Show errors if any
    if error_count > 0:
        with st.expander(f"View Errors ({error_count} records)"):
            error_df = df_results[df_results['error'].notna()]
            st.dataframe(error_df[['original_index', 'error']].head(10), use_container_width=True)

# =========================
# Download Section
# =========================
if st.session_state.results_df is not None:
    st.header("📥 Download Results")
    
    # Prepare download data
    download_df = st.session_state.results_df.copy()
    
    # Remove internal columns
    columns_to_remove = ['original_index', 'error']
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
# Testing Section
# =========================
with st.expander("🧪 Test Pattern Extraction"):
    test_text = st.text_area(
        "Test your extraction on sample text:",
        value="John Smith reported that Mary Johnson was injured at the warehouse on March 15, 2024 at 2:30 PM when she slipped and cut her hand.",
        height=100
    )
    
    if st.button("Test Extraction"):
        if test_text.strip():
            pattern_extractor = RobustPatternExtractor()
            result = pattern_extractor.extract_comprehensive(test_text)
            
            st.subheader("Extraction Results:")
            
            for field, value in result.items():
                if value:
                    st.write(f"**{field.replace('_', ' ').title()}:** {value}")
            
            # Success rate
            extracted_fields = [field for field, value in result.items() if value]
            success_rate = len(extracted_fields) / len(result) * 100
            st.metric("Extraction Success", f"{success_rate:.1f}%")
        else:
            st.warning("Enter some test text first")

# =========================
# Reset Section
# =========================
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("**💡 Tip:** Enhanced Pattern Matching works best for consistent incident report formats.")

with col2:
    if st.button("🔄 Reset App", type="secondary", use_container_width=True, help="Clear all data and start fresh"):

        reset_app()
