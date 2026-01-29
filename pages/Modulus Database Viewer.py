import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from excel_modulus import get_current_db # Import the new function
import json

# Import the update logic
from excel_modulus import update_tags_for_sheet, update_iv_for_sheet

st.set_page_config(page_title="Modulus Database Viewer", layout="wide")

# --- 1. Load & Clean Data ---
CSV_PATH = Path('modulus_db.csv')

def load_data():
    # 1. Load from Cloud or Local
    df = get_current_db(CSV_PATH)
    
    # 2. Check if empty
    if df.empty:
        return pd.DataFrame(columns=['display_name', 'tags', 'notes', 'thickness_mm', 'modulus_gpa', 'sheet_name_sanitized'])

    # 3. Ensure string types for safety
    if 'tags' not in df.columns: df['tags'] = ''
    if 'notes' not in df.columns: df['notes'] = ''
    
    df['tags'] = df['tags'].fillna('').astype(str)
    df['notes'] = df['notes'].fillna('').astype(str)
    
    # 4. Create the 'display_name' column (The missing link!)
    def clean_name(name):
        name = str(name)
        # Remove "ASTM_D882_results_" prefix case-insensitive
        name = re.sub(r'(?i)astm_d882_results_', '', name)
        # Remove trailing separators like "_", "-"
        name = re.sub(r'[-_]+$', '', name)
        return name

    if 'sheet_name_sanitized' in df.columns:
        df['display_name'] = df['sheet_name_sanitized'].apply(clean_name)
    else:
        # Fallback if the CSV is malformed
        df['display_name'] = "Unknown"

    return df

df_full = load_data()

# --- ADD THIS BLOCK ---
if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()
# ----------------------

# --- 2. Sidebar Filters ---
st.sidebar.header("Filter Database")
if df_full.empty:
    st.warning("No database file found. Run 'app.py' to generate data.")
    st.stop()

# Get Global Tag List (for autocomplete)
all_tags = set()
for tag_str in df_full['tags'].unique():
    if tag_str:
        cleaned = [t.strip() for t in tag_str.split(',')]
        all_tags.update(cleaned)
sorted_tags = sorted(list(all_tags))

# Filters
selected_tags = st.sidebar.multiselect("Include Tags", options=sorted_tags)
excluded_tags = st.sidebar.multiselect("Exclude Tags", options=sorted_tags)

df_full['thickness_rounded'] = df_full['thickness_mm'].round(2)
available_thicknesses = sorted(df_full['thickness_rounded'].unique())
selected_thicknesses = st.sidebar.multiselect("Filter by Thickness (mm)", options=available_thicknesses)

available_sheets = sorted(df_full['display_name'].unique())
selected_sheets = st.sidebar.multiselect("Filter by Material Name", options=available_sheets)

# Apply Filters
df_filtered = df_full.copy()
if selected_tags:
    mask = df_filtered['tags'].apply(lambda x: any(t in x for t in selected_tags))
    df_filtered = df_filtered[mask]
if excluded_tags:
    exclude_mask = df_filtered['tags'].apply(lambda x: any(t in x for t in excluded_tags))
    df_filtered = df_filtered[~exclude_mask]
if selected_thicknesses:
    df_filtered = df_filtered[df_filtered['thickness_rounded'].isin(selected_thicknesses)]
if selected_sheets:
    df_filtered = df_filtered[df_filtered['display_name'].isin(selected_sheets)]

if df_filtered.empty:
    st.warning("No data matches your filters.")
    st.stop()

# --- 4. Main Dashboard ---
st.title("Material Database Viewer")

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(df_filtered))
col2.metric("Unique Materials", df_filtered['display_name'].nunique())
col3.metric("Global Avg Modulus", f"{df_filtered['modulus_gpa'].mean():.2f} GPa")

# ADDED NEW TAB: "Edit Tags"
tab_overview, tab_compare, tab_calculator, tab_edit, tab_curves, tab_manage = st.tabs(
    ["📊 Overview", "📋 Raw Data", "🧮 Calculator", "✏️ Edit Tags", "📈 Curve Compare", "🗑️ Manage Data"]
)

# ... [Keep Tab 1, 2, and 3 contents EXACTLY as they were in the previous version] ...
# (I will only write out Tab 4 here to save space, assume Tabs 1-3 are unchanged)

# === TAB 1: OVERVIEW ===
with tab_overview:
    # ... [Same code as previous turn] ...
    c_head, c_tog = st.columns([5, 1])
    with c_head: st.subheader("Modulus Comparison")
    with c_tog: use_notes_label = st.toggle("Use Notes as Labels", value=False)
    
    agg_df = df_filtered.groupby('display_name').agg(
        notes=('notes', 'first'),
        mean=('modulus_gpa', 'mean'),
        std=('modulus_gpa', 'std'),
        count=('modulus_gpa', 'count')
    ).reset_index()
    agg_df = agg_df.sort_values('mean', ascending=False)
    agg_df['notes'] = agg_df['notes'].replace('', '(No Note)')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_filtered, x='display_name', y='modulus_gpa', order=agg_df['display_name'], errorbar='sd', capsize=.1, palette="viridis", ax=ax)
    
    if use_notes_label:
        ax.set_xticklabels(agg_df['notes'], rotation=45, ha='right')
        ax.set_xlabel("Material Note")
    else:
        ax.set_xticklabels(agg_df['display_name'], rotation=45, ha='right')
        ax.set_xlabel("Sheet Name")

    ax.set_ylabel("Modulus (GPa)")
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

# === TAB 2: RAW DATA ===
with tab_compare:
    # ... [Same code as previous turn] ...
    st.subheader("Database Entries")
    show_avgs_only = st.checkbox("Show Sheet Averages Only", value=True)
    if show_avgs_only:
        disp_df = df_filtered.groupby('display_name').agg({'modulus_gpa':'mean', 'tags':'first', 'notes':'first', 'timestamp':'max'}).reset_index()
    else:
        disp_df = df_filtered
    st.dataframe(disp_df, use_container_width=True)

# === TAB 3: CALCULATOR ===
with tab_calculator:
    # ... [Same code as previous turn] ...
    st.subheader("Comparison")
    avail = sorted(df_filtered['display_name'].unique())
    if len(avail) > 0:
        b = st.selectbox("Baseline", avail)
        t = st.selectbox("Target", avail)
        # ... calculation logic ...

# === TAB 4: EDIT METADATA ===
with tab_edit:
    st.subheader("Update Material Metadata")
    
    # 1. Select Material
    edit_opts = df_full[['sheet_name_sanitized', 'display_name']].drop_duplicates()
    name_map = {f"{row['display_name']} ({row['sheet_name_sanitized']})": row['sheet_name_sanitized'] 
                for i, row in edit_opts.iterrows()}
    
    label = st.selectbox("Select Material to Edit", options=sorted(name_map.keys()))
    
    if label:
        sid = name_map[label]
        rows = df_full[df_full['sheet_name_sanitized'] == sid]
        
        if not rows.empty:
            st.divider()
            
            # --- SECTION A: TAGS ---
            st.markdown("#### 🏷️ Tags")
            
            curr_tags_str = rows.iloc[0]['tags']
            curr_tag_list = [t.strip() for t in curr_tags_str.split(',') if t.strip()]
            
            # Filter valid defaults
            valid_defaults = [t for t in curr_tag_list if t in sorted_tags]
            
            c_tag1, c_tag2 = st.columns([3, 1])
            
            with c_tag1:
                # FIX: DYNAMIC KEYS using {sid} ensure the widget resets when sample changes
                updated_list = st.multiselect(
                    "Manage Tags (Select from existing)", 
                    options=sorted_tags, 
                    default=valid_defaults,
                    key=f"edit_multiselect_{sid}" 
                )
                
                new_custom = st.text_input(
                    "Create new tag (comma separated)", 
                    placeholder="e.g. specialized-process",
                    key=f"edit_text_{sid}"
                )

            with c_tag2:
                st.write("") 
                st.write("") 
                st.write("") 
                if st.button("Save Tags", type="primary", key=f"save_btn_{sid}"):
                    final_combined = updated_list.copy()
                    if new_custom:
                        extras = [t.strip() for t in new_custom.split(',') if t.strip()]
                        final_combined.extend(extras)
                    
                    final_str = ", ".join(sorted(list(set(final_combined))))
                    
                    update_tags_for_sheet(CSV_PATH, sid, final_str)
                    st.success("Tags Saved!")
                    st.cache_data.clear()
            
            st.divider()
            
            # --- SECTION B: INTRINSIC VISCOSITY (IV) ---
            st.markdown("#### 🧪 Intrinsic Viscosity (IV)")
            
            curr_iv = 0.0
            if 'iv' in rows.columns:
                val = rows.iloc[0]['iv']
                if pd.notna(val):
                    try:
                        curr_iv = float(val)
                    except:
                        curr_iv = 0.0
            
            c_iv1, c_iv2 = st.columns([3, 1])
            with c_iv1:
                new_iv_val = st.number_input(
                    "Update IV [dL/g]", 
                    value=curr_iv, 
                    step=0.01, 
                    format="%.2f",
                    help="Set to 0.00 to remove IV.",
                    key=f"edit_iv_{sid}" # Dynamic Key
                )
            with c_iv2:
                st.write("") 
                st.write("") 
                if st.button("Save IV", key=f"save_iv_{sid}"):
                    update_iv_for_sheet(CSV_PATH, sid, new_iv_val)
                    st.success(f"IV updated to {new_iv_val}")
                    st.cache_data.clear()
# === TAB 5: CURVE COMPARE ===
with tab_curves:
    st.subheader("Compare Stress-Strain Curves")
    
    # 1. Check for data
    if 'curve_strain' not in df_filtered.columns:
        st.warning("No curve data found. Enable 'Log Stress/Strain Curves' in app.py and save new data.")
    else:
        # Filter for rows that actually have curve data
        has_curve = df_filtered[df_filtered['curve_strain'].str.len() > 5].copy()
        
        if has_curve.empty:
            st.info("No saved curves found in the current filtered selection.")
        else:
            # --- 2. CONTROLS ---
            
            # Row 1: Graph Customization
            c_cust1, c_cust2 = st.columns([1, 2])
            
            # Get tags
            curve_tags = set()
            for t_str in has_curve['tags'].unique():
                if t_str:
                    curve_tags.update([t.strip() for t in t_str.split(',') if t.strip()])
            
            with c_cust1:
                graph_title = st.text_input("Graph Title", value="Comparative Stress-Strain Curves")
                
                # Visual Tag Filter
                hidden_legend_tags = st.multiselect(
                    "👁️ Hide Tags from Legend text:", 
                    options=sorted(list(curve_tags)),
                    help="Curves remain visible, but these tags are removed from the legend text."
                )

            with c_cust2:
                has_curve['select_label'] = has_curve['display_name'] + " | " + has_curve['sample_name']
                
                selected_curves = st.multiselect(
                    "Select Samples to Graph", 
                    options=has_curve['select_label'].unique(),
                    default=has_curve['select_label'].head(5).tolist()
                )
            
            st.write("") 
            st.caption("⚙️ Graph Settings")
            
            # Row 2: Toggles
            c_tog1, c_tog2, c_tog3, c_tog4 = st.columns(4)
            with c_tog1:
                show_raw = st.toggle("Show Raw Force (N)", value=False)
            with c_tog2:
                show_rep_only = st.toggle("Representative Only (Avg)", value=False)
            with c_tog3:
                show_legend_tags = st.toggle("Show Tags in Legend", value=True)
            with c_tog4:
                # NEW: ALIGNMENT TOGGLE
                align_origin = st.toggle("📐 Align Linear Region to (0,0)", value=False, 
                                         help="Corrects for slack/toe. Projects the modulus slope back to zero and shifts the curve.")

            # --- 3. PLOTTING LOGIC ---
            if selected_curves:
                final_labels_to_plot = selected_curves
                
                if show_rep_only:
                    final_labels_to_plot = []
                    subset = has_curve[has_curve['select_label'].isin(selected_curves)]
                    for material_name, group in subset.groupby('display_name'):
                        if len(group) == 0: continue
                        avg_mod = group['modulus_gpa'].mean()
                        best_idx = (group['modulus_gpa'] - avg_mod).abs().idxmin()
                        winner_label = group.loc[best_idx, 'select_label']
                        final_labels_to_plot.append(winner_label)
                    
                    st.info(f"Filtered to {len(final_labels_to_plot)} representative curves.")

                # Plot
                fig, ax = plt.subplots(figsize=(12, 7))
                
                for label in final_labels_to_plot:
                    row = has_curve[has_curve['select_label'] == label].iloc[0]
                    
                    try:
                        # Parse Data
                        strain_arr = np.array(json.loads(row['curve_strain']), dtype=float)
                        stress_arr = np.array(json.loads(row['curve_stress']), dtype=float)
                        
                        # --- 1. TOE COMPENSATION (Align & Clean) ---
                        if align_origin and not show_raw: 
                            # A. Vertical Tare (Fix floating starts)
                            # Shift stress so the minimum value is 0 (removes sensor drift)
                            stress_arr = stress_arr - np.nanmin(stress_arr)

                            # B. Horizontal Align (The Projection)
                            mod_mpa = row['modulus_gpa'] * 1000.0
                            fit_max = row.get('strain_fit_max', 0.02)
                            
                            if mod_mpa > 0:
                                # Anchor to the fitted region
                                idx_anchor = (np.abs(strain_arr - fit_max)).argmin()
                                y_anchor = stress_arr[idx_anchor] / 1e6 # MPa
                                x_anchor = strain_arr[idx_anchor]       # Unitless
                                
                                # Shift X
                                x_offset = x_anchor - (y_anchor / mod_mpa)
                                strain_arr = strain_arr - x_offset
                                
                                # C. Smart Toe Clipping (The Red Line Fix)
                                # Instead of cutting just negative strain, we cut low stress.
                                # Calculate 1% of the max stress for this sample
                                stress_threshold = np.nanmax(stress_arr) * 0.01 
                                
                                # Create mask: Keep data that is ABOVE 1% stress OR POSITIVE strain
                                # This removes the "tail" dragging on the floor, but keeps the elastic rise
                                clean_mask = (stress_arr > stress_threshold) & (strain_arr > -0.002)
                                
                                strain_arr = strain_arr[clean_mask]
                                stress_arr = stress_arr[clean_mask]

                        # --- 2. UNITS & AXES ---
                        if show_raw:
                            area_m2 = row.get('area_mm2', 1.0) * 1e-6
                            y_data = stress_arr * area_m2 # N
                            L0_mm = row.get('initial_length_mm', 100.0)
                            x_data = strain_arr * L0_mm   # mm
                        else:
                            y_data = stress_arr / 1e6     # MPa
                            x_data = strain_arr * 100     # %
                        
                        # --- LEGEND BUILDER (Robust IV Fix) ---
                        if show_rep_only:
                            base_name = row['display_name']
                        else:
                            base_name = label.split(" | ")[-1] 
                        
                        legend_parts = []
                        
                        # 1. Add IV (Safe Casting)
                        if 'iv' in row:
                            val = row['iv']
                            try:
                                # Force convert to float to handle strings like "0.82"
                                float_iv = float(val) 
                                if pd.notna(float_iv) and float_iv > 0:
                                    legend_parts.append(f"IV: {float_iv:.2f}")
                            except (ValueError, TypeError):
                                pass # Ignore if it's not a number

                        # 2. Add Tags (Visual Filter)
                        if show_legend_tags:
                            tag_val = str(row['tags']).strip()
                            if tag_val and tag_val.lower() != 'nan':
                                cur_tags = [t.strip() for t in tag_val.split(',') if t.strip()]
                                vis_tags = [t for t in cur_tags if t not in hidden_legend_tags]
                                if vis_tags:
                                    legend_parts.extend(vis_tags)
                        
                        # 3. Assemble
                        if legend_parts:
                            legend_label = f"{base_name} [{', '.join(legend_parts)}]"
                        else:
                            legend_label = base_name

                        # Plot
                        ax.plot(x_data, y_data, label=legend_label)
                        
                    except Exception as e:
                        # st.warning(f"Skipping {label}: {e}")
                        pass

                # Graph Styling
                if show_raw:
                    ax.set_xlabel("Extension (mm)")
                    ax.set_ylabel("Load Force (N)")
                else:
                    ax.set_xlabel("Strain (%)")
                    ax.set_ylabel("Stress (MPa)")
                
                ax.set_title(graph_title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            else:
                st.info("Select samples above to generate the plot.")
# === TAB 6: MANAGE DATA (DELETE) ===
from excel_modulus import delete_samples_from_db # Import the function

with tab_manage:
    st.subheader("Delete Data Entries")
    st.warning("⚠️ Actions here are permanent. If connected to GitHub, this will push a new commit removing the data.")
    
    # 1. Select Material (Sheet)
    # We use the raw sanitized names to ensure we target the right data
    unique_sheets = sorted(df_full['sheet_name_sanitized'].unique())
    target_sheet = st.selectbox("Select Material / Sheet", options=unique_sheets)
    
    if target_sheet:
        # Get samples for this sheet
        sheet_rows = df_full[df_full['sheet_name_sanitized'] == target_sheet]
        
        # Create a list of sample names: "Sheet-1", "Sheet-2"
        available_samples = sorted(sheet_rows['sample_name'].unique())
        
        # 2. Select Samples to Delete
        selected_to_delete = st.multiselect(
            "Select Samples to Delete", 
            options=available_samples,
            default=[] # Default to none for safety
        )
        
        # "Select All" helper
        if st.checkbox("Select All Samples in this Sheet"):
            selected_to_delete = available_samples
            st.info(f"Selected all {len(selected_to_delete)} samples.")
        
        st.divider()
        
        # 3. Confirmation Button
        if selected_to_delete:
            st.write(f"Ready to delete **{len(selected_to_delete)}** samples.")
            
            # Use Session State for double-confirmation
            if "confirm_delete" not in st.session_state:
                st.session_state["confirm_delete"] = False
                
            delete_btn = st.button("🗑️ Delete Selected Samples", type="primary")
            
            if delete_btn:
                st.session_state["confirm_delete"] = True
                
            if st.session_state["confirm_delete"]:
                st.error("Are you sure? This cannot be undone.")
                col_d1, col_d2 = st.columns(2)
                
                if col_d1.button("Yes, Delete Permanently"):
                    # PERFORM DELETION
                    delete_samples_from_db(CSV_PATH, selected_to_delete)
                    
                    st.success("Deleted!")
                    st.session_state["confirm_delete"] = False
                    st.cache_data.clear() # Clear cache to refresh data
                    # Optional: st.rerun()
                    
                if col_d2.button("Cancel"):
                    st.session_state["confirm_delete"] = False
                    st.info("Cancelled.")
















