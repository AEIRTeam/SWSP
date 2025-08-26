import streamlit as st
import subprocess
import sys
import pandas as pd
import time
from extractors import EnsembleVotingExtractor
import json
import io
from datetime import datetime

def download_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Page configuration
st.set_page_config(
    page_title="ML Entity Extraction Pipeline",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'ensemble' not in st.session_state:
    st.session_state.ensemble = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

st.title("🤖 Real-time ML Entity Extraction Pipeline")
st.markdown("Multi-Model Ensemble with Voting for unstructured data classification")

# Sidebar for configuration
st.sidebar.header("Configuration")
batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=500, value=100)
show_intermediate = st.sidebar.checkbox("Show Intermediate Results", value=True)
show_model_breakdown = st.sidebar.checkbox("Show Model Breakdown", value=False)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Data Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file with unstructured data",
        type=['csv'],
        help="CSV should have a 'text' column containing unstructured data to classify"
    )
    
    if uploaded_file is not None:
        # Load and preview data
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Show data info
        st.info(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Training section
        st.subheader("Model Training")
        
        if st.button("🎯 Train Ensemble Models", type="primary"):
            if 'text' not in df.columns:
                st.error("❌ CSV must contain a 'text' column")
            else:
                with st.spinner("Training ensemble models... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize ensemble
                    st.session_state.ensemble = EnsembleVotingExtractor()
                    
                    # Simulate training progress (adapt to your actual training)
                    training_steps = ["SpaCy NER", "Hybrid Extractor", "Template ML", "Advanced Ensemble"]
                    
                    for i, step in enumerate(training_steps):
                        status_text.text(f"Training {step}...")
                        progress_bar.progress((i + 1) / len(training_steps))
                        
                        # Your actual training would go here
                        # For demo, we'll use a subset of data
                        train_texts = df['text'].tolist()[:min(1000, len(df))]  # Limit for demo
                        train_labels = [{}] * len(train_texts)  # Adapt to your labels
                        
                        if step == "SpaCy NER":
                            st.session_state.ensemble.spacy_extractor.train(train_texts, train_labels)
                        elif step == "Hybrid Extractor":
                            st.session_state.ensemble.hybrid_extractor.train_ml_components(train_texts, train_labels)
                        elif step == "Template ML":
                            st.session_state.ensemble.template_extractor.train_classifiers(train_texts, train_labels)
                        elif step == "Advanced Ensemble":
                            st.session_state.ensemble.advanced_extractor.train(train_texts, train_labels)
                    
                    st.session_state.is_trained = True
                    status_text.text("✅ All models trained successfully!")
                    st.success("🎉 Training completed!")
        
        # Processing section
        if st.session_state.is_trained and st.session_state.ensemble is not None:
            st.subheader("Real-time Processing")
            
            if st.button("🚀 Start Processing", type="primary"):
                # Initialize results storage
                results = []
                total_rows = len(df)
                
                # Create containers for real-time updates
                main_progress = st.progress(0)
                status_container = st.empty()
                metrics_container = st.container()
                results_container = st.empty()
                
                # Metrics display
                with metrics_container:
                    col_metrics = st.columns(4)
                    processed_metric = col_metrics[0].metric("Processed", "0")
                    remaining_metric = col_metrics[1].metric("Remaining", str(total_rows))
                    rate_metric = col_metrics[2].metric("Rate (rows/sec)", "0")
                    eta_metric = col_metrics[3].metric("ETA", "Calculating...")
                
                start_time = time.time()
                
                # Process in batches
                for i in range(0, total_rows, batch_size):
                    batch_start = time.time()
                    batch_end = min(i + batch_size, total_rows)
                    batch_df = df.iloc[i:batch_end]
                    
                    # Process batch
                    batch_results = []
                    for idx, row in batch_df.iterrows():
                        try:
                            final_result, model_predictions = st.session_state.ensemble.extract_with_voting(row['text'])
                            
                            # Create result row
                            result_row = {
                                'original_index': idx,
                                'text_preview': row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
                                **final_result
                            }
                            
                            if show_model_breakdown:
                                result_row['model_breakdown'] = json.dumps(model_predictions)
                            
                            batch_results.append(result_row)
                            
                        except Exception as e:
                            st.error(f"Error processing row {idx}: {str(e)}")
                            batch_results.append({
                                'original_index': idx,
                                'text_preview': row['text'][:100] + '...',
                                'error': str(e)
                            })
                    
                    results.extend(batch_results)
                    
                    # Update progress and metrics
                    progress = batch_end / total_rows
                    main_progress.progress(progress)
                    
                    elapsed_time = time.time() - start_time
                    processing_rate = batch_end / elapsed_time if elapsed_time > 0 else 0
                    remaining_rows = total_rows - batch_end
                    eta_seconds = remaining_rows / processing_rate if processing_rate > 0 else 0
                    
                    # Update metrics
                    status_container.info(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}")
                    
                    # Update metric displays (you'd need to recreate these)
                    with metrics_container:
                        col_metrics = st.columns(4)
                        col_metrics[0].metric("Processed", f"{batch_end:,}")
                        col_metrics[1].metric("Remaining", f"{remaining_rows:,}")
                        col_metrics[2].metric("Rate (rows/sec)", f"{processing_rate:.1f}")
                        col_metrics[3].metric("ETA", f"{eta_seconds/60:.1f} min" if eta_seconds > 60 else f"{eta_seconds:.0f} sec")
                    
                    # Show intermediate results
                    if show_intermediate and results:
                        current_results_df = pd.DataFrame(results)
                        results_container.dataframe(current_results_df.tail(50), use_container_width=True)
                    
                    # Small delay to make progress visible
                    time.sleep(0.1)
                
                # Final results
                st.session_state.results_df = pd.DataFrame(results)
                st.success(f"✅ Processing completed! Extracted data from {total_rows} rows in {elapsed_time:.1f} seconds")

with col2:
    st.header("Real-time Stats")
    
    if st.session_state.results_df is not None:
        df_results = st.session_state.results_df
        
        # Summary statistics
        st.metric("Total Rows Processed", len(df_results))
        
        # Show field extraction success rates
        if len(df_results) > 0:
            st.subheader("Field Extraction Success")
            
            # Count non-null extractions for each field
            field_counts = {}
            for col in df_results.columns:
                if col not in ['original_index', 'text_preview', 'error', 'model_breakdown']:
                    non_null_count = df_results[col].notna().sum()
                    success_rate = (non_null_count / len(df_results)) * 100
                    field_counts[col] = success_rate
            
            # Display as metrics
            for field, rate in field_counts.items():
                st.metric(f"{field.title()} Success", f"{rate:.1f}%")

# Results download section
if st.session_state.results_df is not None:
    st.header("📥 Download Results")
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        # CSV download
        csv_buffer = io.StringIO()
        st.session_state.results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📊 Download as CSV",
            data=csv_data,
            file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_download2:
        # JSON download
        json_data = st.session_state.results_df.to_json(orient='records', indent=2)
        
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Multi-Model Ensemble Entity Extraction")