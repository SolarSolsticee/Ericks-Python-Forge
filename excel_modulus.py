import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress
import streamlit as st
from github import Github, GithubException
from io import StringIO

DEFAULT_WIDTH_MM = 25.4

# --- HELPER: DETECT ENVIRONMENT ---
def is_running_on_streamlit_cloud():
    # Streamlit Cloud usually sets specific environment variables, 
    # but checking for our specific secret is a reliable proxy.
    return "github" in st.secrets

# --- GITHUB CONNECTION ---
def get_github_repo():
    """Connects to the repo using the token in secrets."""
    if not is_running_on_streamlit_cloud():
        return None
    g = Github(st.secrets["github"]["token"])
    return g.get_repo(st.secrets["github"]["repo_name"])

def load_csv_from_github(filename):
    """Downloads the CSV content from GitHub main branch."""
    repo = get_github_repo()
    try:
        contents = repo.get_contents(filename, ref=st.secrets["github"]["branch"])
        return pd.read_csv(StringIO(contents.decoded_content.decode()))
    except Exception:
        # File might not exist yet
        return pd.DataFrame()

def push_csv_to_github(filename, df, commit_message):
    """Updates (or creates) the CSV file in the GitHub repo."""
    repo = get_github_repo()
    branch = st.secrets["github"]["branch"]
    csv_content = df.to_csv(index=False)
    
    try:
        # Try to get existing file to update it
        contents = repo.get_contents(filename, ref=branch)
        repo.update_file(contents.path, commit_message, csv_content, contents.sha, branch=branch)
    except GithubException:
        # File doesn't exist, create it
        repo.create_file(filename, commit_message, csv_content, branch=branch)

# --- CORE LOGIC ---

def sanitize_sheet_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^0-9A-Za-z_-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name

def load_workbook_all_sheets(path: Path) -> dict:
    return pd.read_excel(path, sheet_name=None, engine='openpyxl')

def find_header_row(df: pd.DataFrame) -> int:
    search_limit = min(100, len(df))
    for i in range(search_limit):
        row = df.iloc[i]
        try:
            c = str(row.iloc[2]).strip().lower()
            d = str(row.iloc[3]).strip().lower()
            if 'extension' in c and 'primary load' in d:
                return i
        except Exception:
            continue
    raise ValueError('Header row with "Extension" and "Primary Load" not found in first 100 rows.')

def extract_extension_load(df: pd.DataFrame, header_row: int) -> pd.DataFrame:
    data = df.iloc[header_row+1:, [2, 3]].copy()
    data.columns = ['extension', 'primary_load']
    data['extension'] = pd.to_numeric(data['extension'], errors='coerce')
    data['primary_load'] = pd.to_numeric(data['primary_load'], errors='coerce')
    return data.reset_index(drop=True)

def split_samples_by_smart_markers(data: pd.DataFrame) -> list:
    is_data = data['extension'].notna() & data['primary_load'].notna()
    blocks = []
    current_block_indices = []
    for idx, valid in is_data.items():
        if valid:
            current_block_indices.append(idx)
        else:
            if current_block_indices:
                blocks.append(data.loc[current_block_indices].reset_index(drop=True))
                current_block_indices = []
    if current_block_indices:
        blocks.append(data.loc[current_block_indices].reset_index(drop=True))
    return blocks

def compute_stress_strain(extension_mm, load_n, initial_length_mm, thickness_mm, width_mm=DEFAULT_WIDTH_MM):
    area_mm2 = width_mm * thickness_mm
    area_m2 = area_mm2 * 1e-6 
    if area_m2 == 0 or initial_length_mm == 0:
        return np.zeros_like(extension_mm), np.zeros_like(load_n)
    strain = np.array(extension_mm, dtype=float) / float(initial_length_mm)
    stress = np.array(load_n, dtype=float) / area_m2
    return strain, stress

def fit_modulus(strain, stress):
    mask = np.isfinite(strain) & np.isfinite(stress)
    if mask.sum() < 2:
        return dict(slope=np.nan, intercept=np.nan, rvalue=np.nan, n=0)
    res = linregress(strain[mask], stress[mask])
    return dict(slope=res.slope, intercept=res.intercept, rvalue=res.rvalue, n=mask.sum())

# --- DATABASE OPERATIONS (Hybrid Local/Cloud) ---

def get_current_db(path: Path):
    """Reads DB from GitHub if on Cloud, else local disk."""
    if is_running_on_streamlit_cloud():
        return load_csv_from_github(path.name)
    else:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

def save_db(path: Path, df: pd.DataFrame, commit_msg: str):
    """Saves DB to GitHub if on Cloud, else local disk."""
    if is_running_on_streamlit_cloud():
        push_csv_to_github(path.name, df, commit_msg)
    else:
        df.to_csv(path, index=False)

def check_sheet_exists(path: Path, sheet_name_sanitized: str) -> bool:
    df = get_current_db(path)
    if df.empty or 'sheet_name_sanitized' not in df.columns:
        return False
    return sheet_name_sanitized in df['sheet_name_sanitized'].values

def update_database(path: Path, new_rows: list):
    columns = [
        'timestamp', 'workbook_filename', 'sheet_name_raw', 'sheet_name_sanitized',
        'sample_name', 'sample_index', 'initial_length_mm', 'thickness_mm', 
        'width_mm', 'area_mm2', 'modulus_pa', 'modulus_gpa', 
        'r_value', 'n_points', 'strain_fit_min', 'strain_fit_max', 
        'notes', 'tags'
    ]
    
    new_df = pd.DataFrame(new_rows)
    for c in columns:
        if c not in new_df.columns:
            new_df[c] = np.nan
    new_df = new_df[columns]

    existing_df = get_current_db(path)
    
    if not existing_df.empty:
        # Upsert Logic: Remove old entries for this sheet
        sheets_being_updated = new_df['sheet_name_sanitized'].unique()
        # Ensure existing_df has the right columns to avoid errors
        existing_df = existing_df.reindex(columns=columns)
        existing_df = existing_df[~existing_df['sheet_name_sanitized'].isin(sheets_being_updated)]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    # SAVE (Local or Cloud)
    sheet_id = new_df['sheet_name_sanitized'].iloc[0] if not new_df.empty else "batch"
    save_db(path, final_df, commit_msg=f"Update results for {sheet_id}")

def update_tags_for_sheet(path: Path, sheet_name_sanitized: str, new_tag_string: str):
    df = get_current_db(path)
    if df.empty:
        return
    
    mask = df['sheet_name_sanitized'] == sheet_name_sanitized
    if mask.any():
        df.loc[mask, 'tags'] = new_tag_string
        save_db(path, df, commit_msg=f"Update tags for {sheet_name_sanitized}")
