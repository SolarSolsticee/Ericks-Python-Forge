import streamlit as st
import pandas as pd
import io
import os
import csv # Import the csv module for delimiter sniffing

def merge_csv_to_excel_buffer(uploaded_files):
    """
    Merges multiple uploaded CSV files into a single Excel file within an in-memory buffer.
    Each CSV is copied verbatim into a separate sheet. It uses a robust method to
    handle non-standard CSV formats with title or empty rows.

    Args:
        uploaded_files (list): A list of Streamlit UploadedFile objects.

    Returns:
        io.BytesIO: A BytesIO buffer containing the generated Excel file data.
    """
    output_buffer = io.BytesIO()
    progress_bar = st.progress(0)
    progress_status = st.empty()

    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
        total_files = len(uploaded_files)
        for i, file in enumerate(uploaded_files):
            progress_status.text(f"Processing file {i+1}/{total_files}: {file.name}...")
            
            try:
                file.seek(0)
                
                # --- A MORE ROBUST PARSING STRATEGY ---
                # First, decode the file from bytes to a string stream
                # This is necessary for the csv module to work with it.
                string_io = io.StringIO(file.getvalue().decode('utf-8', errors='ignore'))
                
                delimiter = ',' # Default delimiter
                try:
                    # Try to sniff the delimiter from a sample.
                    # Reading the first line might not be enough if it's a title.
                    sample = string_io.read(2048)
                    string_io.seek(0) # Reset after reading sample
                    dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
                    delimiter = dialect.delimiter
                except csv.Error:
                    st.warning(f"Could not auto-detect delimiter for '{file.name}'. Defaulting to comma (',').")

                # Use the more lenient, built-in csv.reader to parse the file
                # This handles rows with varying numbers of columns gracefully.
                reader = csv.reader(string_io, delimiter=delimiter)
                data = list(reader)

                if data:
                    # Convert the list of lists directly into a DataFrame
                    df = pd.DataFrame(data)

                    # --- Sheet Name Generation ---
                    sheet_base_name, _ = os.path.splitext(file.name)
                    sheet_name = sheet_base_name[:31].replace(':', '_').replace('\\', '_').replace('/', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_')

                    # Write the DataFrame verbatim to the Excel sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                else:
                    st.warning(f"File '{file.name}' is empty and will be skipped.")

            except Exception as e:
                st.error(f"An error occurred while processing '{file.name}': {e}")
            
            progress_bar.progress((i + 1) / total_files)

    progress_status.text("Merge complete!")
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

