import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress

DEFAULT_WIDTH_MM = 25.4


def sanitize_sheet_name(name: str) -> str:
    """Creates a filename-safe version of the sheet name."""
    name = str(name).strip()
    name = re.sub(r"[^0-9A-Za-z_-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name


def load_workbook_all_sheets(path: Path) -> dict:
    """Loads all sheets from the Excel file."""
    return pd.read_excel(path, sheet_name=None, engine='openpyxl')


def find_header_row(df: pd.DataFrame) -> int:
    """
    Scans the dataframe to find the row where:
    Column C (index 2) is 'Extension' AND Column D (index 3) is 'Primary Load'.
    """
    # Limit search to first 100 rows to save time
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
    """Extracts the relevant columns below the detected header."""
    data = df.iloc[header_row + 1:, [2, 3]].copy()
    data.columns = ['extension', 'primary_load']
    data['extension'] = pd.to_numeric(data['extension'], errors='coerce')
    data['primary_load'] = pd.to_numeric(data['primary_load'], errors='coerce')
    return data.reset_index(drop=True)


def split_samples_by_smart_markers(data: pd.DataFrame) -> list:
    """
    Splits data into samples.
    Assumption: A valid data row has numeric Extension AND numeric Load.
    Separator: Rows where Load is NaN OR Extension is an integer marker (1, 2, 3) with no load.
    """
    # Mark rows that are valid data (both are numbers)
    is_data = data['extension'].notna() & data['primary_load'].notna()

    blocks = []
    current_block_indices = []

    for idx, valid in is_data.items():
        if valid:
            current_block_indices.append(idx)
        else:
            # We hit a separator (NaN or text or marker)
            if current_block_indices:
                # If we have accumulated data, save it as a block
                blocks.append(data.loc[current_block_indices].reset_index(drop=True))
                current_block_indices = []

    # Append the final block if exists
    if current_block_indices:
        blocks.append(data.loc[current_block_indices].reset_index(drop=True))

    return blocks


def compute_stress_strain(extension_mm, load_n, initial_length_mm, thickness_mm, width_mm=DEFAULT_WIDTH_MM):
    """
    Computes Engineering Stress and Strain.
    Stress = Force / Area (Pa)
    Strain = Delta_L / L_0 (unitless)
    """
    area_mm2 = width_mm * thickness_mm
    # Convert mm^2 to m^2 for Pascals
    area_m2 = area_mm2 * 1e-6

    # Avoid division by zero
    if area_m2 == 0 or initial_length_mm == 0:
        return np.zeros_like(extension_mm), np.zeros_like(load_n)

    strain = np.array(extension_mm, dtype=float) / float(initial_length_mm)
    stress = np.array(load_n, dtype=float) / area_m2
    return strain, stress


def fit_modulus(strain, stress):
    """
    Performs linear regression on the provided arrays.
    """
    mask = np.isfinite(strain) & np.isfinite(stress)
    if mask.sum() < 2:
        return dict(slope=np.nan, intercept=np.nan, rvalue=np.nan, n=0)

    res = linregress(strain[mask], stress[mask])
    return dict(
        slope=res.slope,  # Young's Modulus in Pa
        intercept=res.intercept,
        rvalue=res.rvalue,
        n=mask.sum()
    )


# In excel_modulus.py

def update_database(path: Path, new_rows: list):
    """
    Updates the CSV database.
    If entries for the current sheet already exist, they are REMOVED
    and replaced with the new values (Upsert logic).
    """
    # Added 'tags' to this list
    columns = [
        'timestamp', 'workbook_filename', 'sheet_name_raw', 'sheet_name_sanitized',
        'sample_name', 'sample_index', 'initial_length_mm', 'thickness_mm',
        'width_mm', 'area_mm2', 'modulus_pa', 'modulus_gpa',
        'r_value', 'n_points', 'strain_fit_min', 'strain_fit_max',
        'notes', 'tags'  # <--- NEW COLUMN
    ]

    new_df = pd.DataFrame(new_rows)

    # Ensure columns match schema
    for c in columns:
        if c not in new_df.columns:
            new_df[c] = np.nan
    new_df = new_df[columns]

    if path.exists():
        try:
            existing_df = pd.read_csv(path)

            # Filter: Keep rows that are NOT in the list of sheets we are updating
            sheets_being_updated = new_df['sheet_name_sanitized'].unique()
            existing_df = existing_df[~existing_df['sheet_name_sanitized'].isin(sheets_being_updated)]

            # Combine
            final_df = pd.concat([existing_df, new_df], ignore_index=True)

        except pd.errors.EmptyDataError:
            final_df = new_df
    else:
        final_df = new_df


    final_df.to_csv(path, index=False)

def check_sheet_exists(path: Path, sheet_name_sanitized: str) -> bool:
    """Checks if a sheet name already exists in the CSV."""
    if not path.exists():
        return False
    try:
        # Read only the sheet_name column to be fast
        df = pd.read_csv(path, usecols=['sheet_name_sanitized'])
        return sheet_name_sanitized in df['sheet_name_sanitized'].values
    except Exception:
        return False

def update_tags_for_sheet(path: Path, sheet_name_sanitized: str, new_tag_string: str):
    """Updates the tags for all samples belonging to a specific sheet."""
    if not path.exists():
        return
    
    df = pd.read_csv(path)
    
    # Update tags where the sheet name matches
    mask = df['sheet_name_sanitized'] == sheet_name_sanitized
    if mask.any():
        df.loc[mask, 'tags'] = new_tag_string
        df.to_csv(path, index=False)
