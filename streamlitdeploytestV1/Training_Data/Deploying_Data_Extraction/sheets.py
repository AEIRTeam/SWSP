# sheets.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


def make_unique_headers(headers):
    """Ensure headers are unique and non-empty."""
    seen = {}
    new_headers = []
    for i, h in enumerate(headers):
        h = h.strip() if h else ""  # remove whitespace
        if h == "":
            h = f"col_{i+1}"  # replace empty with col_#
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"  # rename duplicate
        else:
            seen[h] = 0
        new_headers.append(h)
    return new_headers


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard preprocessing to a DataFrame.
    
    Steps:
    1. Drop completely empty rows
    2. Fill NaN values with "Missing"
    3. Clean column headers (lowercase, replace spaces with underscores)
    4. Strip whitespace from string columns
    """
    # Store original info for reporting
    original_shape = df.shape
    
    # Drop completely empty rows
    df = df.dropna(how="all")
    
    # Fill NaN values
    df = df.fillna("Missing")
    
    # Clean column headers
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") 
        for c in df.columns
    ]
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Report changes
    new_shape = df.shape
    rows_removed = original_shape[0] - new_shape[0]
    
    if rows_removed > 0:
        st.info(f"🧹 Removed {rows_removed} empty rows during preprocessing")
    
    return df


def fetch_google_sheets(spreadsheet_name: str, worksheet_name: str = "Sheet2"):
    """Fetch data from a specific Google Sheets worksheet and return DataFrame."""
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ]

        # Check if secrets are available
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Google Service Account credentials not found in secrets.")
            st.write("Please add your Google Service Account credentials to Streamlit secrets.")
            return None

        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes,
        )
        gc = gspread.authorize(creds)

        st.info(f"🔑 Connected with service account: {st.secrets['gcp_service_account']['client_email']}")

        # Open spreadsheet
        try:
            sh = gc.open(spreadsheet_name)
        except gspread.SpreadsheetNotFound:
            st.error(f"❌ Spreadsheet '{spreadsheet_name}' not found. Please check the name and permissions.")
            return None
        
        # Open worksheet
        try:
            worksheet = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            st.error(f"❌ Worksheet '{worksheet_name}' not found in '{spreadsheet_name}'.")
            available_sheets = [ws.title for ws in sh.worksheets()]
            st.write(f"Available worksheets: {', '.join(available_sheets)}")
            return None

        # Get all values
        values = worksheet.get_all_values()
        if not values:
            st.warning("⚠️ The worksheet is empty.")
            return None

        if len(values) == 1:
            st.warning("⚠️ The worksheet only contains headers (no data rows).")
            return None

        # Process headers and data
        raw_headers = values[0]
        headers = make_unique_headers(raw_headers)
        
        # Create DataFrame
        df = pd.DataFrame(values[1:], columns=headers)

        st.success(f"✅ Fetched {len(df)} rows from **{spreadsheet_name}** → **{worksheet_name}**")
        
        # Show basic info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        
        # Preview the data
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(5), use_container_width=True)
        
        return df

    except Exception as e:
        st.error(f"❌ Failed to fetch Google Sheets data: {e}")
        st.write("Common issues:")
        st.write("- Check if the service account has access to the spreadsheet")
        st.write("- Verify the spreadsheet name is correct")
        st.write("- Ensure the worksheet name exists")
        return None


def display_data_section(spreadsheet_name: str, worksheet_name: str = "Sheet2"):
    """
    Streamlit section to let users choose between Google Sheets or file upload.
    Includes optional preprocessing toggle.
    Returns a DataFrame (preprocessed or raw based on user choice).
    """
    st.markdown("---")
    st.header("📊 Import Data")

    # Data source selection
    option = st.radio(
        "Choose a data source:",
        ("Google Sheets", "Upload a file"),
        horizontal=True,
    )
    
    # Preprocessing toggle
    apply_preprocessing = st.checkbox(
        "🛠️ Apply preprocessing",
        value=True,
        help="Clean headers, remove empty rows, fill NaN values"
    )

    df = None

    if option == "Google Sheets":
        if st.button("Fetch Google Sheets Data", type="primary", use_container_width=True):
            df = fetch_google_sheets(spreadsheet_name, worksheet_name)

    elif option == "Upload a file":
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"✅ Uploaded: **{uploaded_file.name}**")
                
                # Show basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                
                # Preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(5), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                return None

    # Apply preprocessing if requested and data is available
    if df is not None:
        if apply_preprocessing:
            with st.spinner("Applying preprocessing..."):
                df_original = df.copy()
                df = preprocess_dataframe(df)
                
                # Show before/after comparison
                with st.expander("🔍 Preprocessing Comparison"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Before")
                        st.write(f"Shape: {df_original.shape}")
                        st.write("Columns:")
                        for i, col in enumerate(df_original.columns[:5]):
                            st.write(f"  {i+1}. `{col}`")
                        if len(df_original.columns) > 5:
                            st.write(f"  ... and {len(df_original.columns) - 5} more")
                    
                    with col2:
                        st.subheader("After")
                        st.write(f"Shape: {df.shape}")
                        st.write("Columns:")
                        for i, col in enumerate(df.columns[:5]):
                            st.write(f"  {i+1}. `{col}`")
                        if len(df.columns) > 5:
                            st.write(f"  ... and {len(df.columns) - 5} more")
                
                st.success("✨ Preprocessing completed successfully!")
        
        # Final data display
        st.markdown("### 📊 Final Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        return df

    return None


# 🔄 Backward compatibility wrapper so existing app.py keeps working
def display_google_sheets_section(spreadsheet_name: str, worksheet_name: str = "Sheet2"):
    """
    Wrapper to maintain compatibility with existing app.py code.
    Returns raw Google Sheets data without preprocessing options.
    """
    return fetch_google_sheets(spreadsheet_name, worksheet_name)