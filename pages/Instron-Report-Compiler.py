import streamlit as st
import pandas as pd
import pdfplumber
import re
from pathlib import Path
import io

# Set the page configuration for the Streamlit app
st.set_page_config(layout="wide", page_title="PDF Report Compiler")


def clean_value(value):
    """
    Converts extracted string values to floats or None if not possible.
    Handles '----', '---', and removes commas/special characters.
    """
    if isinstance(value, str):
        value = value.strip().replace('$', '').replace('~', '')
        if value in ['----', '---']:
            return None
        value = value.replace(',', '')
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def clean_header(header_text):
    """
    Cleans up header text from PDF tables that might be split across lines.
    """
    if not header_text:
        return ""
    # Join lines, consolidate whitespace, and handle specific known issues
    text = re.sub(r'\s+', ' ', header_text.replace('\n', ' ')).strip()
    corrections = {
        'Strengt h': 'Strength', 'Thickne ss': 'Thickness',
        'Elongatio n at': 'Elongation at', 'Elongatio nat': 'Elongation at',
        'Modulu S': 'Modulus', 'Modulu s': 'Modulus', 'Secan t': 'Secant',
        'Tange nt': 'Tangent', '(Secan 13%)': '(Secant 3%)', ' %': '%'
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


def process_pdf_report(uploaded_file, status_placeholder):
    """
    Extracts tabular data from a single uploaded PDF report file.
    Takes a Streamlit UploadedFile object.
    """
    status_placeholder.info(f"Processing: {uploaded_file.name}...")

    # Use the name of the uploaded file to derive a sheet name
    sheet_name = Path(uploaded_file.name).stem
    common_params = {}
    specimen_data_list = []

    # Mapping from PDF header variations to final Excel column names
    col_keys = {
        'Breaking Factor [N/m]': 'Breaking Factor [N/m]',
        'Tensile Strength [MPa]': 'Tensile Strength [MPa]',
        'Tensile Strength at Break [MPa]': 'Tensile Strength at Break [MPa]',
        '% Elongation at Break [%]': 'Maximum recorded Elongation [%]',
        'Yield Strength (Zero slope) [MPa]': 'Yield Strength (Zero slope) [MPa]',
        '% Elongation at Yield [%]': '% Elongation at Yield [%]',
        'Modulus (Tangent 3%) [MPa]': 'Modulus (Tangent 3%) [MPa]',
        'Modulus (Secant 3%) [MPa]': 'Modulus (Secant 3%) [MPa]',
        'Tensile Energy to Break [J]': 'Tensile Energy to Break [J]',
        'Break behavior': 'Break Behavior'
    }

    # pdfplumber can open file-like objects directly
    with pdfplumber.open(uploaded_file) as pdf:
        # --- Extract parameters from the first page using regex ---
        try:
            first_page_text = pdf.pages[0].extract_text()

            # Use regex to find various parameters, with fallbacks
            patterns = {
                'Run_Date': r'(\d{1,2}/\d{1,2}/\d{4})',
                'Width [mm]': r'Width\s+([\d.]+)\s+mm',
                'Thickness [mm]': r'Thickness\s+([\d.]+)\s+mm',
                'Gauge length (mm)': r'Gauge length\s+([\d.]+)\s+mm',
                'Sample length (mm)': r'Specimen length \(mm\)\s+([\d.]+)',
                'Strain rate (mm/mm/min)': r'Initial strain rate\s+([$\d.~]+)\s*mm/mm~min',
                'Rate (mm/min)': r'Rate 1\s+([$\d.~]+)\s*mm/min'
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, first_page_text)
                if match:
                    common_params[key] = clean_value(match.group(1)) if key != 'Run_Date' else match.group(1)
                else:
                    common_params[key] = None
        except Exception as e:
            st.warning(f"Could not extract all parameters from page 1 of {uploaded_file.name}: {e}")

        # --- Extract main results table from all pages ---
        header_map = None
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table: continue

                if header_map is None:
                    first_row = [clean_header(cell) for cell in table[0]]
                    if 'Breaking Factor [N/m]' in first_row:
                        status_placeholder.info(
                            f"Found main data header in {uploaded_file.name} on page {page.page_number}.")
                        header_map = {header: i for i, header in enumerate(first_row)}
                        rows_to_process = table[1:]
                    else:
                        continue  # Not the header table, skip
                else:
                    rows_to_process = table

                # Process rows identified for data extraction
                for row in rows_to_process:
                    if row and row[0] and str(row[0]).isdigit():
                        specimen_id = int(row[0])
                        specimen_dict = {}
                        for key_pdf, key_excel in col_keys.items():
                            if key_pdf in header_map:
                                idx = header_map[key_pdf]
                                if idx < len(row):
                                    raw_val = row[idx]
                                    if key_excel != 'Break Behavior':
                                        specimen_dict[key_excel] = clean_value(raw_val)
                                    else:
                                        specimen_dict[key_excel] = clean_header(raw_val) if raw_val else ''
                        specimen_dict['Identifier'] = f"{sheet_name}-{specimen_id}"
                        specimen_data_list.append(specimen_dict)

    # De-duplicate any re-processed rows and combine with common parameters
    unique_specimens = []
    seen_identifiers = set()
    for specimen in specimen_data_list:
        if specimen['Identifier'] not in seen_identifiers:
            specimen.update(common_params)
            specimen['Sheet_identifier'] = sheet_name
            unique_specimens.append(specimen)
            seen_identifiers.add(specimen['Identifier'])

    return unique_specimens


def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("📄 Instron BlueHill 2 PDF Report Compiler")
    st.markdown("""
    This application is specifically designed to extract key parameters and tabular data from **Instron BlueHill 2** PDF reports and compile them into a single Excel file.

    **Instructions:**
    1.  **Upload** one or more PDF report files generated by BlueHill 2.
    2.  **Enter** a base name for your output Excel file.
    3.  **Click** the process button to generate and download the compiled data.
    """)

    # --- Step 1: File Uploader ---
    uploaded_files = st.file_uploader(
        "**Step 1: Select PDF Files**",
        type="pdf",
        accept_multiple_files=True
    )

    # --- Step 2: Output File Name ---
    output_base_name = st.text_input(
        "**Step 2: Enter Base Name for Output Excel File**",
        value="compiled-report-data"
    )

    # --- Step 3: Process Button ---
    if st.button("🚀 Process Files and Generate Excel", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file to continue.")
            return
        if not output_base_name:
            st.warning("Please provide a base name for the output file.")
            return

        all_data = []
        has_errors = False

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            try:
                data_from_pdf = process_pdf_report(file, status_placeholder)
                if data_from_pdf:
                    all_data.extend(data_from_pdf)
                else:
                    st.warning(f"No valid data was extracted from {file.name}.")
            except Exception as e:
                st.error(f"An error occurred while processing {file.name}: {e}")
                has_errors = True
            progress_bar.progress((i + 1) / len(uploaded_files))

        status_placeholder.empty()  # Clear the status message

        if not all_data:
            message = "Processing complete, but no data could be extracted from the selected files."
            if not has_errors:
                message += "\nPlease check that the PDFs have the expected format and table structure."
            st.info(message)
            return

        st.success(f"Successfully extracted data from {len(uploaded_files)} PDF(s).")

        df = pd.DataFrame(all_data)

        # Define the desired final column order
        final_column_order = [
            'Run_Date', 'Sheet_identifier', 'Strain rate (mm/mm/min)', 'Rate (mm/min)',
            'Sample length (mm)', 'Gauge length (mm)', 'Temperature (C)',
            'Relative Humidity (%)', 'Sheet Direction', 'Width [mm]', 'Thickness [mm]',
            'Breaking Factor [N/m]', 'Tensile Strength [MPa]',
            'Tensile Strength at Break [MPa]', 'Maximum recorded Elongation [%]',
            'Yield Strength (Zero slope) [MPa]', '% Elongation at Yield [%]',
            'Modulus (Tangent 3%) [MPa]', 'Modulus (Secant 3%) [MPa]',
            'Tensile Energy to Break [J]', 'Break Behavior'
        ]

        # Add any missing columns with empty values and reorder
        for col in final_column_order:
            if col not in df.columns:
                df[col] = ''
        df = df[final_column_order]

        st.dataframe(df)

        # Convert DataFrame to Excel in-memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='CompiledData')

        # Prepare data for download
        excel_data = output.getvalue()
        output_filename = f"{output_base_name}.xlsx"

        st.download_button(
            label="📥 Download Compiled Excel File",
            data=excel_data,
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()

