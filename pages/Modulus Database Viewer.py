import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

st.set_page_config(page_title="Modulus Database Viewer", layout="wide")

# --- 1. Load & Clean Data ---
CSV_PATH = Path('modulus_db.csv')


def load_data():
    if not CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)

    # Ensure tags/notes are string for searching
    df['tags'] = df['tags'].fillna('').astype(str)
    df['notes'] = df['notes'].fillna('').astype(str)

    # --- CLEANUP NAME LOGIC ---
    def clean_name(name):
        name = str(name)
        name = re.sub(r'(?i)astm_d882_results_', '', name)
        name = re.sub(r'[-_]+$', '', name)
        return name

    df['display_name'] = df['sheet_name_sanitized'].apply(clean_name)
    return df


df_full = load_data()

# --- 2. Sidebar Filters ---
st.sidebar.header("Filter Database")

if df_full.empty:
    st.warning("No database file found. Run 'Modulus DB Entry App.py' to generate data.")
    st.stop()

# Get all unique tags
all_tags = set()
for tag_str in df_full['tags'].unique():
    if tag_str:
        cleaned = [t.strip() for t in tag_str.split(',')]
        all_tags.update(cleaned)
sorted_tags = sorted(list(all_tags))

# A. INCLUDE Tags
selected_tags = st.sidebar.multiselect(
    "Include Tags (Any)",
    options=sorted_tags,
    help="Show samples that have at least one of these tags."
)

# B. EXCLUDE Tags (New Feature)
excluded_tags = st.sidebar.multiselect(
    "Exclude Tags (None)",
    options=sorted_tags,
    help="Hide samples that contain any of these tags."
)

# C. Thickness Filter
df_full['thickness_rounded'] = df_full['thickness_mm'].round(2)
available_thicknesses = sorted(df_full['thickness_rounded'].unique())
selected_thicknesses = st.sidebar.multiselect("Filter by Thickness (mm)", options=available_thicknesses)

# D. Sheet Filter
available_sheets = sorted(df_full['display_name'].unique())
selected_sheets = st.sidebar.multiselect("Filter by Material Name", options=available_sheets)

# --- 3. Apply Filters ---
df_filtered = df_full.copy()

# 1. Apply Inclusion
if selected_tags:
    mask = df_filtered['tags'].apply(lambda x: any(t in x for t in selected_tags))
    df_filtered = df_filtered[mask]

# 2. Apply Exclusion (Logic: Remove row if it shares ANY tag with excluded_list)
if excluded_tags:
    # lambda returns True if the row contains a forbidden tag
    exclude_mask = df_filtered['tags'].apply(lambda x: any(t in x for t in excluded_tags))
    # We keep rows where exclude_mask is False (using ~)
    df_filtered = df_filtered[~exclude_mask]

# 3. Apply Thickness
if selected_thicknesses:
    df_filtered = df_filtered[df_filtered['thickness_rounded'].isin(selected_thicknesses)]

# 4. Apply Sheet Name
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

tab_overview, tab_compare, tab_calculator = st.tabs(["📊 Overview & Charts", "📋 Raw Data", "🧮 % Difference Calculator"])

# === TAB 1: OVERVIEW & CHARTS ===
with tab_overview:
    # Header & Toggle Layout
    c_head, c_tog = st.columns([5, 1])
    with c_head:
        st.subheader("Modulus Comparison")
    with c_tog:
        # VISUAL TOGGLE
        use_notes_label = st.toggle("Use Notes as Labels", value=False)

    # 1. ALWAYS Aggregate by Display Name (Sheet) to keep bars distinct
    agg_df = df_filtered.groupby('display_name').agg(
        notes=('notes', 'first'),
        mean=('modulus_gpa', 'mean'),
        std=('modulus_gpa', 'std'),
        count=('modulus_gpa', 'count')
    ).reset_index()

    # 2. Sort by Mean Modulus for the chart
    agg_df = agg_df.sort_values('mean', ascending=False)

    # 3. Clean up empty notes for display if needed
    agg_df['notes'] = agg_df['notes'].replace('', '(No Note)')

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df_filtered,
        x='display_name',
        y='modulus_gpa',
        order=agg_df['display_name'],
        errorbar='sd',
        capsize=.1,
        palette="viridis",
        ax=ax
    )

    # 5. OVERRIDE LABELS (Purely Visual)
    if use_notes_label:
        ax.set_xticklabels(agg_df['notes'], rotation=45, ha='right')
        ax.set_xlabel("Material Note")
    else:
        ax.set_xticklabels(agg_df['display_name'], rotation=45, ha='right')
        ax.set_xlabel("Sheet Name")

    ax.set_ylabel("Modulus (GPa)")
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

    st.divider()

    # 2. Summary Table
    st.subheader("Summary Statistics")

    display_cols = ['notes', 'display_name', 'mean', 'std', 'count']

    st.dataframe(
        agg_df[display_cols].style
        .format({'mean': '{:.2f} GPa', 'std': '{:.2f} GPa'})
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ]),
        use_container_width=True,
        hide_index=True
    )

# === TAB 2: RAW DATA ===
with tab_compare:
    st.subheader("Database Entries")

    show_avgs_only = st.checkbox("Show Sheet Averages Only (Hide Samples)")

    cols_order_samples = ['notes', 'display_name', 'sample_index', 'thickness_mm', 'modulus_gpa', 'r_value', 'tags',
                          'timestamp']
    cols_order_agg = ['notes', 'display_name', 'thickness_mm', 'Avg Modulus (GPa)', 'tags', 'timestamp']

    if show_avgs_only:
        disp_df = df_filtered.groupby('display_name').agg({
            'modulus_gpa': 'mean',
            'thickness_mm': 'mean',
            'tags': 'first',
            'notes': 'first',
            'timestamp': 'max'
        }).reset_index()
        disp_df = disp_df.rename(columns={'modulus_gpa': 'Avg Modulus (GPa)'})
        disp_df = disp_df[cols_order_agg]
        format_dict = {'Avg Modulus (GPa)': '{:.2f}'}
    else:
        disp_df = df_filtered[cols_order_samples]
        format_dict = {'modulus_gpa': '{:.2f}', 'r_value': '{:.4f}'}

    st.dataframe(
        disp_df.style
        .format(format_dict)
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ]),
        use_container_width=True,
        hide_index=True
    )

# === TAB 3: COMPARISON CALCULATOR ===
with tab_calculator:
    st.subheader("Percentage Difference Calculator")

    # Use filtered data names if available, but fallback to all if needed?
    # Usually better to stick to filtered scope in calculator so you don't pick hidden items.
    available_names = sorted(df_filtered['display_name'].unique())

    if len(available_names) < 1:
        st.error("Not enough data to compare.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            baseline_name = st.selectbox("Select Baseline (Control)", options=available_names, index=0)
            base_df = df_filtered[df_filtered['display_name'] == baseline_name]
            base_avg = base_df['modulus_gpa'].mean()
            base_std = base_df['modulus_gpa'].std()
            st.info(f"**{baseline_name}**\n\nAvg: {base_avg:.2f} GPa (±{base_std:.2f})")

        with c2:
            target_name = st.selectbox("Select Target (Experimental)", options=available_names,
                                       index=min(1, len(available_names) - 1))
            target_df = df_filtered[df_filtered['display_name'] == target_name]
            target_avg = target_df['modulus_gpa'].mean()
            target_std = target_df['modulus_gpa'].std()
            st.info(f"**{target_name}**\n\nAvg: {target_avg:.2f} GPa (±{target_std:.2f})")

        st.divider()

        if base_avg > 0:
            diff = target_avg - base_avg
            pct_diff = (diff / base_avg) * 100

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric("Percentage Difference", f"{pct_diff:.2f}%", f"{diff:.2f} GPa")

            with col_res2:
                if pct_diff > 0:
                    st.success(f"**{target_name}** is **{pct_diff:.1f}% stiffer**.")
                elif pct_diff < 0:
                    st.error(f"**{target_name}** is **{abs(pct_diff):.1f}% more flexible**.")
                else:
                    st.warning("Identical stiffness.")

                if len(base_df) > 1 and len(target_df) > 1:
                    from scipy.stats import ttest_ind

                    t_stat, p_val = ttest_ind(base_df['modulus_gpa'].dropna(), target_df['modulus_gpa'].dropna(),
                                              equal_var=False)
                    if p_val < 0.05:
                        st.markdown(f"✅ Statistically Significant (p={p_val:.4f})")
                    else:
                        st.markdown(f"❌ Not Statistically Significant (p={p_val:.4f})")