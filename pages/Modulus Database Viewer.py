import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from excel_modulus import get_current_db # Import the new function

# Import the update logic
from excel_modulus import update_tags_for_sheet

st.set_page_config(page_title="Modulus Database Viewer", layout="wide")

# --- 1. Load & Clean Data ---
CSV_PATH = Path('modulus_db.csv')

def load_data():
    # This now works on Cloud (fetching from GitHub) AND Local
    df = get_current_db(CSV_PATH)
    
    if df.empty:
        return pd.DataFrame()
        
    df['tags'] = df['tags'].fillna('').astype(str)
    df['notes'] = df['notes'].fillna('').astype(str)
    # ... rest of your cleaning logic ...
    return df

df_full = load_data()

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
tab_overview, tab_compare, tab_calculator, tab_edit = st.tabs(
    ["📊 Overview & Charts", "📋 Raw Data", "🧮 % Difference Calculator", "✏️ Edit Tags"]
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

# === TAB 4: EDIT TAGS (NEW) ===
with tab_edit:
    st.subheader("Update Tags for Existing Entries")
    st.info("Select a material below to modify its tags. This updates the CSV immediately.")
    
    # 1. Select Material (Use full list, not filtered, so you can edit anything)
    # We use the raw sanitized names for ID, but display the clean name
    edit_opts = df_full[['sheet_name_sanitized', 'display_name']].drop_duplicates()
    
    # Create a dictionary for the dropdown: "Display Name (ID)" -> "ID"
    # This ensures uniqueness if two sheets map to the same display name
    name_map = {f"{row['display_name']} ({row['sheet_name_sanitized']})": row['sheet_name_sanitized'] 
                for i, row in edit_opts.iterrows()}
    
    selected_label = st.selectbox("Select Material to Edit", options=sorted(name_map.keys()))
    
    if selected_label:
        target_sheet_id = name_map[selected_label]
        
        # Get current tags for this sheet
        # We take the tags from the first sample of this sheet
        current_rows = df_full[df_full['sheet_name_sanitized'] == target_sheet_id]
        if not current_rows.empty:
            current_tag_str = current_rows.iloc[0]['tags']
            # Convert string "a, b" to list ['a', 'b']
            current_tag_list = [t.strip() for t in current_tag_str.split(',') if t.strip()]
            
            st.write(f"**Current Tags:** {current_tag_str if current_tag_str else '(None)'}")
            
            # Form to edit
            with st.form("edit_tags_form"):
                # Multiselect with existing global tags
                updated_selection = st.multiselect(
                    "Select tags", 
                    options=sorted_tags, 
                    default=[t for t in current_tag_list if t in sorted_tags]
                )
                
                # Text input for BRAND NEW tags
                # Find tags in current_tag_list that are NOT in sorted_tags (custom ones)
                custom_existing = [t for t in current_tag_list if t not in sorted_tags]
                custom_existing_str = ", ".join(custom_existing)
                
                new_custom_tags = st.text_input("Add new/custom tags (comma separated)", value=custom_existing_str)
                
                submitted = st.form_submit_button("Update Database")
                
                if submitted:
                    # Combine lists
                    final_list = updated_selection.copy()
                    if new_custom_tags:
                        extras = [t.strip() for t in new_custom_tags.split(',') if t.strip()]
                        final_list.extend(extras)
                    
                    # Deduplicate and Stringify
                    final_tag_str = ", ".join(sorted(list(set(final_list))))
                    
                    # WRITE TO DB
                    update_tags_for_sheet(CSV_PATH, target_sheet_id, final_tag_str)
                    
                    st.success(f"Updated tags for **{target_sheet_id}** to: {final_tag_str}")
                    st.cache_data.clear() # Clear cache so viewer reloads new data
                    # st.rerun() # Optional: force reload page

