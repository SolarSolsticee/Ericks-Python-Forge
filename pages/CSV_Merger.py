import streamlit as st
import pandas as pd
import io
import os
import csv # Import the csv module for delimiter sniffing

def merge_csv_to_excel_buffer(uploaded_files):
    """
    Merges multiple uploaded CSV files into a single Excel file within an in-memory buffer.
    Each CSV is copied verbatim into a separate sheet. It automatically detects the
    delimiter for each file.

    Args:
        uploaded_files (list): A list of Streamlit UploadedFile objects.

    Returns:
        io.BytesIO: A BytesIO buffer containing the generated Excel file data.
    """
    # Create an in-memory buffer to hold the Excel file
    output_buffer = io.BytesIO()

    # Use a progress bar to show the merging process
    progress_bar = st.progress(0)
    progress_status = st.empty()

    # Create an Excel writer object targeting the in-memory buffer
    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
        total_files = len(uploaded_files)
        for i, file in enumerate(uploaded_files):
            # Update progress status for the user
            progress_status.text(f"Processing file {i+1}/{total_files}: {file.name}...")
            
            try:
                # Before reading the file, reset its internal pointer to the beginning
                file.seek(0)
                
                # --- NEW: Auto-detect the delimiter ---
                # Read the first few lines to get a sample for the sniffer
                sample = file.read(2048).decode('utf-8', errors='ignore')
                file.seek(0) # Reset pointer after reading sample
                
                # Use the CSV Sniffer to find the correct separator
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
                delimiter = dialect.delimiter
                
                # MODIFIED: Read the CSV using the auto-detected delimiter.
                # Added engine='python' for better flexibility with sniffing.
                df = pd.read_csv(file, header=None, sep=delimiter, engine='python')

                if not df.empty:
                    # --- Sheet Name Generation ---
                    sheet_base_name, _ = os.path.splitext(file.name)
                    # Sanitize the sheet name to comply with Excel's rules (e.g., max 31 chars)
                    sheet_name = sheet_base_name[:31].replace(':', '_').replace('\\', '_').replace('/', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_')

                    # Write the DataFrame to the Excel sheet without adding an
                    # index or a header, ensuring a true verbatim copy.
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                else:
                    st.warning(f"File '{file.name}' is empty and will be skipped.")

            except csv.Error:
                 st.error(f"Could not determine the delimiter for '{file.name}'. Please ensure it is a standard CSV file. Skipping.")
            except pd.errors.EmptyDataError:
                st.warning(f"File '{file.name}' contains no data and will be skipped.")
            except Exception as e:
                st.error(f"An error occurred while processing '{file.name}': {e}")
            
            # Update the progress bar
            progress_bar.progress((i + 1) / total_files)

    progress_status.text("Merge complete!")
    
    # After writing is done, reset the buffer's pointer to the beginning
    output_buffer.seek(0)
    return output_buffer

# --- Streamlit Application UI ---

st.set_page_config(layout="centered", page_title="CSV to Excel Merger")

# --- Header ---
st.title("📊 CSV to Excel Merger & Converter")
st.markdown("""
This tool helps you combine multiple CSV files into a single Excel workbook.
Each CSV file will be placed in its own sheet, preserving the original layout.
""")

# --- Step 1: File Upload ---
st.header("1. Upload Your CSV Files")
uploaded_files = st.file_uploader(
    "Drag and drop or click to select multiple CSV files.",
    accept_multiple_files=True,
    type="csv",
    help="You can upload one or more CSV files."
)

# --- Step 2: Naming the Output ---
st.header("2. Set Your Output File Name")
base_name = st.text_input(
    "Enter a name for the final Excel file (without extension):",
    "merged_output",
    help="This name will be used for the downloaded .xlsx file."
)

# --- Step 3: Merging and Downloading ---
if uploaded_files and base_name:
    st.header("3. Merge and Download")
    st.info(f"Ready to merge **{len(uploaded_files)}** file(s) into **`{base_name}-refined.xlsx`**.")

    if st.button(f"🚀 Merge Files Now", type="primary"):
        with st.spinner('Working our magic... Copying files...'):
            try:
                # Call the main function to process the files
                excel_buffer = merge_csv_to_excel_buffer(uploaded_files)
                
                st.success("✅ Success! Your file is ready for download.")

                # Provide the download button
                st.download_button(
                    label="📥 Download Excel File",
                    data=excel_buffer,
                    file_name=f"{base_name}-refined.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"A critical error occurred: {e}")
                st.error("Please check your files and try again.")
else:
    st.info("Please upload at least one CSV file and provide an output name to proceed.")

