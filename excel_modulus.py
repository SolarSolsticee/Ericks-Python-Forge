import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress
import streamlit as st
from github import Github, GithubException
from io import StringIO
import json

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
    """Downloads the CSV content using the raw URL to bypass the 1MB API limit."""
    repo = get_github_repo()
    if not repo:
        return pd.DataFrame()
        
    try:
        # 1. Get the file metadata (lightweight) to find the download URL
        contents = repo.get_contents(filename, ref=st.secrets["github"]["branch"])
        
        # 2. Use pandas to read directly from the raw URL
        # We must pass the token in the storage_options for private repos
        token = st.secrets["github"]["token"]
        
        return pd.read_csv(
            contents.download_url, 
            storage_options={'Authorization': f'token {token}'}
        )
        
    except Exception as e:
        # File doesn't exist or connection failed
        print(f"Error loading DB: {e}")
        return pd.DataFrame()

def push_csv_to_github(filename, df, commit_message):
    """Updates (or creates) the CSV file in the GitHub repo."""
    repo = get_github_repo()
    branch = st.secrets["github"]["branch"]
    csv_content = df.to_csv(index=False)
    
    # Clean filename (API dislikes leading ./)
    filename = str(filename).lstrip("./")

    try:
        # 1. Try to fetch the existing file (we NEED the SHA to update)
        contents = repo.get_contents(filename, ref=branch)
        
        # PyGithub can return a list if the path implies a directory, handle that safety case
        if isinstance(contents, list):
            contents = contents[0]

        # 2. File exists -> Update it using the fetched SHA
        repo.update_file(
            path=contents.path,
            message=commit_message,
            content=csv_content,
            sha=contents.sha, # <--- The critical missing piece in the error
            branch=branch
        )
        
    except GithubException as e:
        if e.status == 404:
            # 3. File not found (404) -> Create new file (no SHA needed)
            repo.create_file(
                path=filename,
                message=commit_message,
                content=csv_content,
                branch=branch
            )
        else:
            # 4. Some other error (Permissions, etc.) -> Raise it so we see it
            raise e

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
    """
    Extracts Extension (Col C), Load (Col D), and optionally DIC Strain (Col E).
    """
    # We now try to read 3 columns: C (idx 2), D (idx 3), E (idx 4)
    # We slice up to column 5 to ensure we get E if it exists
    data = df.iloc[header_row+1:, 2:5].copy()
    
    # Standardize column names based on how many columns we actually found
    if data.shape[1] >= 3:
        data.columns = ['extension', 'primary_load', 'dic_strain']
    else:
        data.columns = ['extension', 'primary_load']
        data['dic_strain'] = np.nan # Placeholder if missing

    # Convert to numeric
    data['extension'] = pd.to_numeric(data['extension'], errors='coerce')
    data['primary_load'] = pd.to_numeric(data['primary_load'], errors='coerce')
    data['dic_strain'] = pd.to_numeric(data['dic_strain'], errors='coerce')
    
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

def compute_stress_strain(extension_mm, load_n, dic_strain_raw, initial_length_mm, thickness_mm, width_mm, use_dic=False):
    """
    Computes Stress and Strain.
    If use_dic is True, uses the raw 'dic_strain' column directly (no L0 division).
    """
    area_mm2 = width_mm * thickness_mm
    area_m2 = area_mm2 * 1e-6 
    
    if area_m2 == 0:
        return np.zeros_like(load_n), np.zeros_like(load_n)

    # STRESS IS ALWAYS LOAD / AREA
    stress = np.array(load_n, dtype=float) / area_m2
    
    # STRAIN LOGIC
    if use_dic:
        # DIC data is already Strain (unitless or %)
        # Assumption: DIC data is usually standard strain (mm/mm). 
        # If it's percent, the user might need to divide by 100, but usually raw DIC is unitless.
        strain = np.array(dic_strain_raw, dtype=float)
        
        # Check for NaNs in DIC column which would break the plot
        if np.all(np.isnan(strain)):
            # Fallback if user checked box but col is empty
            return np.zeros_like(stress), stress
    else:
        # Standard Machine Extension / L0
        if initial_length_mm == 0:
            return np.zeros_like(extension_mm), stress
        strain = np.array(extension_mm, dtype=float) / float(initial_length_mm)

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
    """
    Updates the CSV database.
    """
    columns = [
        'timestamp', 'workbook_filename', 'sheet_name_raw', 'sheet_name_sanitized',
        'sample_name', 'sample_index', 'initial_length_mm', 'thickness_mm', 
        'width_mm', 'area_mm2', 'modulus_pa', 'modulus_gpa', 
        'r_value', 'n_points', 'strain_fit_min', 'strain_fit_max', 
        'notes', 'tags',
        'curve_strain', 'curve_stress' # <--- NEW COLUMNS FOR CURVE DATA
    ]
    
    new_df = pd.DataFrame(new_rows)
    for c in columns:
        if c not in new_df.columns:
            new_df[c] = np.nan
    new_df = new_df[columns]

    existing_df = get_current_db(path)
    
    if not existing_df.empty:
        sheets_being_updated = new_df['sheet_name_sanitized'].unique()
        existing_df = existing_df.reindex(columns=columns)
        existing_df = existing_df[~existing_df['sheet_name_sanitized'].isin(sheets_being_updated)]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    sheet_id = new_df['sheet_name_sanitized'].iloc[0] if not new_df.empty else "batch"
    save_db(path, final_df, commit_msg=f"Update results for {sheet_id}")

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

def delete_samples_from_db(path: Path, sample_names_to_delete: list):
    """
    Removes rows where 'sample_name' matches the provided list.
    Works for both Local file and GitHub Cloud.
    """
    df = get_current_db(path)
    if df.empty:
        return
    
    # Check if 'sample_name' exists (it should)
    if 'sample_name' not in df.columns:
        return

    # Filter out the bad rows
    # We keep rows where sample_name is NOT in the deletion list
    initial_count = len(df)
    df_new = df[~df['sample_name'].isin(sample_names_to_delete)]
    final_count = len(df_new)
    
    deleted_count = initial_count - final_count
    
    if deleted_count > 0:
        save_db(path, df_new, commit_msg=f"Deleted {deleted_count} samples")




