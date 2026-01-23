import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Import logic from the local module
from excel_modulus import (
    load_workbook_all_sheets,
    find_header_row,
    extract_extension_load,
    split_samples_by_smart_markers,
    compute_stress_strain,
    fit_modulus,
    sanitize_sheet_name,
    update_database,
    check_sheet_exists,
    DEFAULT_WIDTH_MM,
)

st.set_page_config(page_title="Modulus DB Entry", layout='wide')

# --- SIDEBAR: Configuration ---
st.sidebar.title("Configuration")
uploaded = st.sidebar.file_uploader('1. Upload .xlsx file', type=['xlsx'])
output_csv = st.sidebar.text_input('2. Output Database (CSV)', value='modulus_db.csv')
width_mm = st.sidebar.number_input('3. Sample Width (mm)', value=DEFAULT_WIDTH_MM)

if not uploaded:
    st.info("Please upload an Excel file to begin analysis.")
    st.stop()


# --- HELPER: Get Tags from CSV ---
def get_existing_tags(csv_path):
    """Scans the existing database to find unique tags for autocomplete."""
    path = Path(csv_path)
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
        if 'tags' not in df.columns:
            return []

        # Split "tag1, tag2" strings into individual items
        all_tags = df['tags'].dropna().astype(str).str.split(',')
        # Flatten the list and strip whitespace
        flat_tags = [t.strip() for sublist in all_tags for t in sublist]
        return sorted(list(set(flat_tags)))
    except Exception:
        return []


# --- Load Data ---
@st.cache_data
def get_sheets(file_bytes):
    temp_path = Path("temp_uploaded.xlsx")
    with open(temp_path, 'wb') as f:
        f.write(file_bytes.getbuffer())
    return load_workbook_all_sheets(temp_path), temp_path.name


try:
    sheets, filename = get_sheets(uploaded)
    st.sidebar.success(f"Loaded {len(sheets)} sheets.")
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# --- Main Selection ---
selected_sheet = st.sidebar.selectbox("Select Sheet to Analyze", options=list(sheets.keys()))

# --- STATE MANAGEMENT ---
if 'current_sheet' not in st.session_state:
    st.session_state['current_sheet'] = None

if selected_sheet != st.session_state['current_sheet']:
    st.session_state['current_sheet'] = selected_sheet
    st.session_state['input_L0'] = 100.0
    st.session_state['input_thick'] = 0.89
    st.session_state['input_s_min'] = 0.0
    st.session_state['input_s_max'] = 2.0
    st.session_state['input_notes'] = ""
    # Reset tags when switching sheets?
    # Usually safer to reset so you don't accidentally tag PLA as ABS
    st.session_state['selected_tags'] = []
    st.session_state['new_tags'] = ""

# --- Main Layout ---
st.title(f'Analysis: {selected_sheet}')

if selected_sheet:
    df_raw = sheets[selected_sheet]
    sanitized_name = sanitize_sheet_name(selected_sheet)

    try:
        header_row = find_header_row(df_raw)
        data = extract_extension_load(df_raw, header_row)
        blocks = split_samples_by_smart_markers(data)
    except Exception as e:
        st.error(f"Could not parse sheet '{selected_sheet}'. Reason: {e}")
        st.stop()

    if not blocks:
        st.warning("Header found, but no data blocks detected.")
        st.stop()

    st.markdown(f"**Found {len(blocks)} samples.**")

    # 2. Geometry Inputs
    c1, c2 = st.columns(2)
    with c1:
        initial_length_mm = st.number_input('Initial Length ($L_0$) [mm]', step=1.0, key='input_L0')
    with c2:
        thickness_mm = st.number_input('Thickness [mm]', step=0.01, format="%.2f", key='input_thick')

    # 1. Parse Sheet
    try:
        header_row = find_header_row(df_raw)
        data = extract_extension_load(df_raw, header_row)
        blocks = split_samples_by_smart_markers(data)
    except Exception as e:
        st.error(f"Could not parse sheet. {e}")
        st.stop()

    # --- CONTROLS ---
    st.subheader("Physics Settings")
    c_phys1, c_phys2 = st.columns(2)
    with c_phys1:
        # Check if DIC column has data
        dic_avail = data['dic_strain'].notna().sum() > 0
        use_dic = st.checkbox("Use DIC Strain (Column E)", value=False, disabled=not dic_avail, 
                              help="If checked, uses Column E directly as strain. Ignores Initial Length.")
        if use_dic and not dic_avail:
            st.warning("Column E appears empty. DIC mode may fail.")
            
    with c_phys2:
        save_curves = st.checkbox("Log Stress/Strain Curves to DB", value=True,
                                  help="Saves the full x,y data points to the database for future graphing.")

    # 3. Interactive Plotting & Window Selection
st.subheader("Strain Window Selection")
    
plot_data = []
for i, blk in enumerate(blocks, start=1):
    ext = blk['extension'].values
    load = blk['primary_load'].values
    
    # Safe get for DIC column (handles if it's missing)
    dic = blk.get('dic_strain', pd.Series([np.nan]*len(blk))).values
    
    # PASS THE NEW 'use_dic' FLAG HERE
    strain, stress = compute_stress_strain(ext, load, dic, initial_length_mm, thickness_mm, width_mm, use_dic=use_dic)
    
    plot_data.append({
        'id': i, 
        'strain_raw': strain, 
        'stress_raw': stress,
        'strain_pct': strain * 100, 
        'stress_mpa': stress / 1e6
    })

    col_controls, col_plot = st.columns([1, 2])

    with col_controls:
        st.markdown("### Settings")
        strain_min_pct = st.number_input('Strain Min (%)', step=0.1, key='input_s_min')
        strain_max_pct = st.number_input('Strain Max (%)', step=0.1, key='input_s_max')
        preview = st.checkbox("Show Window Lines", value=True)

        st.divider()
        st.markdown("### Metadata")

        # --- NEW TAGGING SYSTEM ---
        # 1. Fetch existing tags for autocomplete
        existing_opts = get_existing_tags(output_csv)

        # 2. Multiselect for known tags
        selected_tags = st.multiselect(
            "Select existing tags",
            options=existing_opts,
            key='selected_tags'
        )

        # 3. Text input for new tags
        new_tags_input = st.text_input(
            "Add new tags (comma separated)",
            placeholder="e.g. annealed, supplier-B",
            key='new_tags'
        )

        # Combine them
        final_tag_list = selected_tags.copy()
        if new_tags_input:
            # clean up manual input
            manual_tags = [t.strip() for t in new_tags_input.split(',') if t.strip()]
            final_tag_list.extend(manual_tags)

        # Deduplicate and join
        final_tag_string = ", ".join(sorted(list(set(final_tag_list))))

        if final_tag_string:
            st.caption(f"Final Tags: *{final_tag_string}*")

        notes = st.text_area("Notes", placeholder="e.g. 50% infill", key='input_notes')
        # --- NEW SAVE LOGIC ---
        # Initialize session state for confirmation if not present
        if "confirm_save" not in st.session_state:
            st.session_state["confirm_save"] = False

        submit = st.button("Calculate & Save to DB", type="primary")
        # Logic: User Clicks Submit -> Check DB -> Either Save or Ask Confirm
        if submit:
            exists = check_sheet_exists(Path(output_csv), sanitized_name)
            if exists:
                st.session_state["confirm_save"] = True
            else:
                # No conflict, save immediately (trigger via a flag to run the block below)
                st.session_state["do_save"] = True

        # Render Confirmation Warning if flag is set
        if st.session_state["confirm_save"]:
            st.warning(f"⚠️ Data for **{sanitized_name}** already exists in the database.")
            col_conf_1, col_conf_2 = st.columns(2)
            with col_conf_1:
                if st.button("✅ Yes, Overwrite"):
                    st.session_state["do_save"] = True
                    st.session_state["confirm_save"] = False  # Reset
            with col_conf_2:
                if st.button("❌ Cancel"):
                    st.session_state["confirm_save"] = False
                    st.info("Save cancelled.")

    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for p in plot_data:
            sns.lineplot(x=p['strain_pct'], y=p['stress_mpa'], ax=ax, label=f"S{p['id']}")

        if preview:
            ax.axvline(strain_min_pct, color='r', linestyle='--', alpha=0.5)
            ax.axvline(strain_max_pct, color='r', linestyle='--', alpha=0.5)
            ax.axvspan(strain_min_pct, strain_max_pct, color='r', alpha=0.1)

        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(f"Stress-Strain Curves: {selected_sheet}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 4. Actual Saving Block (Runs if safe or if confirmed)
        if st.session_state.get("do_save", False):
            rows = []
            summary_metrics = []

            for p in plot_data:
                strain = p['strain_raw']
                stress = p['stress_raw']

                limit_min = strain_min_pct / 100.0
                limit_max = strain_max_pct / 100.0

                mask = (strain >= limit_min) & (strain <= limit_max)
                fit = fit_modulus(strain[mask], stress[mask])
                modulus_pa = fit['slope']
                modulus_gpa = modulus_pa / 1e9 if not np.isnan(modulus_pa) else 0.0

                curve_strain_str = ""
                curve_stress_str = ""
                if save_curves:
                    # Save as JSON string "[0.01, 0.02, ...]"
                    # Rounding to 6 decimals saves huge amounts of space
                    curve_strain_str = json.dumps(np.round(p['strain_raw'], 6).tolist())
                    curve_stress_str = json.dumps(np.round(p['stress_raw'], 0).tolist())

                rows.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'workbook_filename': filename,
                    'sheet_name_raw': selected_sheet,
                    'sheet_name_sanitized': sanitized_name,
                    'sample_name': f'{sanitized_name}-{p["id"]}',
                    'sample_index': p['id'],
                    'initial_length_mm': initial_length_mm,
                    'thickness_mm': thickness_mm,
                    'width_mm': width_mm,
                    'area_mm2': width_mm * thickness_mm,
                    'modulus_pa': modulus_pa,
                    'modulus_gpa': modulus_gpa,
                    'r_value': fit['rvalue'],
                    'n_points': fit['n'],
                    'strain_fit_min': limit_min,
                    'strain_fit_max': limit_max,
                    'notes': notes,
                    'tags': final_tag_string,
                    'curve_strain': curve_strain_str,  # <--- Add this
                    'curve_stress': curve_stress_str,  # <--- Add this
                })
                summary_metrics.append(modulus_gpa)

            update_database(Path(output_csv), rows)

            # Reset the trigger
            st.session_state["do_save"] = False

            avg_mod = np.nanmean(summary_metrics)
            st.success(f"Saved {len(rows)} samples to `{output_csv}`")
            st.metric("Average Modulus", f"{avg_mod:.2f} GPa")
            st.dataframe(pd.DataFrame(rows)[['sample_name', 'modulus_gpa', 'tags']])





