import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import MaxNLocator
import io

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Excel Data Visualizer")
st.title("📊 Excel Data Visualization Tool")

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'figure_buffer' not in st.session_state:
    st.session_state.figure_buffer = None
if 'ci_text' not in st.session_state:
    st.session_state.ci_text = ""

# --- Helper Functions ---
def calculate_ci_text(df, y_col, group_col):
    """Calculates and returns a formatted string for the 95% CI of bar chart means."""
    ci_text = []
    try:
        if group_col is None or group_col == "None":
            data_series = df[y_col].dropna()
            if len(data_series) > 1:
                mean = np.mean(data_series)
                sem = stats.sem(data_series)
                ci = stats.t.interval(0.95, len(data_series) - 1, loc=mean, scale=sem)
                ci_text.append(f"Overall Mean: {mean:.3f}")
                ci_text.append(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
        else:
            grouped = df.groupby(group_col)[y_col]
            for name, group in grouped:
                group = group.dropna()
                if len(group) > 1:
                    mean = np.mean(group)
                    sem = stats.sem(group)
                    ci = stats.t.interval(0.95, len(group) - 1, loc=mean, scale=sem)
                    ci_text.append(f"Group '{name}':")
                    ci_text.append(f"  Mean={mean:.3f}, 95% CI=({ci[0]:.3f}, {ci[1]:.3f})")
    except Exception as e:
        ci_text.append(f"Error calculating CI: {e}")
    return "\n".join(ci_text)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("1. Data Loading")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    sheet_index = st.number_input("Sheet Index", min_value=0, value=0, step=1,
                                  help="The index of the Excel sheet to use (0 for the first sheet).")

    if uploaded_file:
        try:
            st.session_state.df = pd.read_excel(uploaded_file, sheet_name=sheet_index)
            st.session_state.figure_buffer = None
            st.session_state.ci_text = ""
            st.success(f"Loaded '{uploaded_file.name}'.")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.session_state.df = None

# --- Main Application Area ---
if st.session_state.df is None:
    st.info("👋 Welcome! Upload an Excel file using the sidebar to begin.")
    st.stop()

# --- Display Controls if DataFrame is loaded ---
df = st.session_state.df
columns = df.columns.tolist()
columns_with_none = ["None"] + columns

with st.sidebar:
    st.header("2. Plot Configuration")
    
    with st.expander("📊 Plot Type & Axes", expanded=True):
        plot_type = st.selectbox("Plot Type",
                                 ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart",
                                  "Plot of Averages", "Line Plot (of Averages)"])
        x_col = st.selectbox("X-Axis Column", columns)
        y_col_disabled = plot_type == "Histogram"
        y_col = st.selectbox("Y-Axis Column", columns_with_none,
                             index=0 if y_col_disabled else 1, disabled=y_col_disabled)
        group1_col = st.selectbox("Group By Column 1 (Hue)", columns_with_none)
        group2_col = st.selectbox("Group By Column 2 (Style)", columns_with_none)

    with st.expander("⚙️ Plot Specifics", expanded=False):
        if plot_type == "Histogram":
            overlay = st.selectbox("Overlay Plot", ["None", "KDE", "Rug"])
            num_bins = st.slider("Number of Bins", 5, 200, 30)
        
        if plot_type in ["Bar Chart", "Scatter Plot", "Plot of Averages", "Line Plot (of Averages)"]:
            ci_checkbox = st.checkbox("Display 95% CI", help="Shows a 95% confidence interval.")
        
        if plot_type == "Line Plot (of Averages)":
            show_line = st.checkbox("Show Connecting Line", value=True)

        if plot_type == "Scatter Plot":
            st.markdown("---")
            st.write("**Error Bar Values (+/-)**")
            err_bars_disabled = 'ci_checkbox' in locals() and ci_checkbox
            x_err = st.text_input("X-Error", disabled=err_bars_disabled, help="Enter a single number.")
            y_err = st.text_input("Y-Error", disabled=err_bars_disabled, help="Enter a single number.")

    with st.expander("🔍 Data Filtering", expanded=False):
        filtered_df = df.copy()
        if group1_col != "None":
            unique_groups1 = sorted(df[group1_col].dropna().unique())
            selected_groups1 = st.multiselect(f"Filter '{group1_col}' (Hue)",
                                              options=unique_groups1, default=unique_groups1)
            filtered_df = filtered_df[filtered_df[group1_col].isin(selected_groups1)]

        if group2_col != "None":
            unique_groups2 = sorted(df[group2_col].dropna().unique())
            selected_groups2 = st.multiselect(f"Filter '{group2_col}' (Style)",
                                              options=unique_groups2, default=unique_groups2)
            filtered_df = filtered_df[filtered_df[group2_col].isin(selected_groups2)]

    with st.expander("🎨 Formatting & Style", expanded=False):
        title = st.text_input("Plot Title", "My Plot Title")
        xlabel = st.text_input("X-Axis Label", x_col)
        ylabel = st.text_input("Y-Axis Label", y_col if not y_col_disabled else "Count")
        title_fontsize = st.slider("Title Font", 8, 40, 16)
        label_fontsize = st.slider("Label Font", 6, 30, 12)
        tick_fontsize = st.slider("Tick Font", 6, 24, 10)
        st.divider()
        palette = st.selectbox("Color Palette", ["deep", "muted", "pastel", "bright", "dark", "colorblind", "viridis", "rocket"])
        bg_color = st.selectbox("Background Color", ["white", "lightgrey", "beige", "ivory", "whitesmoke"])
        alpha = st.slider("Plot Transparency (Alpha)", 0.0, 1.0, 0.7)

    with st.expander("✨ Advanced Axes & Lines", expanded=False):
        col1, col2 = st.columns(2)
        xlim_min = col1.number_input("X-Axis Min", value=None, format="%f")
        xlim_max = col1.number_input("X-Axis Max", value=None, format="%f")
        xticks = col1.number_input("X-Axis Ticks (#)", min_value=0, value=0, help="Approx. # of ticks. 0 for auto.")
        ylim_min = col2.number_input("Y-Axis Min", value=None, format="%f")
        ylim_max = col2.number_input("Y-Axis Max", value=None, format="%f")
        yticks = col2.number_input("Y-Axis Ticks (#)", min_value=0, value=0, help="Approx. # of ticks. 0 for auto.")
        st.divider()
        vlines_str = st.text_input("Vertical Lines (X-Values, comma-separated)", help="e.g., 10, 25.5, 40")
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        vline_width = vcol1.number_input("Width", 0.5, 10.0, 1.5, 0.5)
        vline_style = vcol2.selectbox("Style", ["solid", "dashed", "dotted", "dashdot"])
        vline_color = vcol3.selectbox("Color", ["red", "black", "blue", "green", "gray"])
        vline_alpha = vcol4.slider("Alpha", 0.0, 1.0, 1.0, 0.1)

    # --- NEW: Export Options Section ---
    with st.expander("📥 Export Options", expanded=True):
        aspect_ratio_text = st.selectbox("Aspect Ratio (w x h inches)",
                                         ["Auto (10x6)", "4:3 (8x6)", "16:9 (12x6.75)", "1:1 (8x8)"])
        dpi = st.slider("DPI (Dots Per Inch)", 75, 600, 300, 25)

st.divider()

if st.button("🚀 Generate Plot", type="primary", use_container_width=True):
    # --- Map aspect ratio text to figsize tuple ---
    aspect_map = {
        "Auto (10x6)": (10, 6),
        "4:3 (8x6)": (8, 6),
        "16:9 (12x6.75)": (12, 6.75),
        "1:1 (8x8)": (8, 8)
    }
    figsize = aspect_map[aspect_ratio_text]

    fig, ax = plt.subplots(figsize=figsize)
    try:
        # --- Plotting Logic (same as before) ---
        if plot_type == "Histogram":
            sns.histplot(data=filtered_df, x=x_col, hue=group1_col if group1_col != "None" else None, 
                         palette=palette, alpha=alpha, ax=ax, kde=(overlay == "KDE"), bins=num_bins)
            if overlay == "Rug":
                sns.rugplot(data=filtered_df, x=x_col, hue=group1_col if group1_col != "None" else None, palette=palette, ax=ax)

        elif plot_type == "Bar Chart":
            if y_col == "None": raise ValueError("Please select a Y-Axis column.")
            sns.barplot(data=filtered_df, x=x_col, y=y_col, hue=group1_col if group1_col != "None" else None,
                        palette=palette, ax=ax, errorbar=('ci', 95) if ci_checkbox else None, alpha=alpha)
            if ci_checkbox:
                st.session_state.ci_text = calculate_ci_text(filtered_df, y_col, group1_col if group1_col != "None" else None)
        # ... (Other plot types remain the same) ...
        elif plot_type == "Box Plot":
            if y_col == "None": raise ValueError("Please select a Y-Axis column.")
            sns.boxplot(data=filtered_df, x=x_col, y=y_col, hue=group1_col if group1_col != "None" else None, palette=palette, ax=ax)
        elif plot_type == "Plot of Averages":
            if y_col == "None": raise ValueError("Please select a Y-Axis column.")
            sns.pointplot(data=filtered_df, x=x_col, y=y_col, hue=group1_col if group1_col != "None" else None,
                          errorbar=('ci', 95) if ci_checkbox else None, palette=palette, ax=ax, join=False, dodge=True)
        elif plot_type == "Line Plot (of Averages)":
            if y_col == "None": raise ValueError("Please select a Y-Axis column.")
            sns.lineplot(data=filtered_df, x=x_col, y=y_col, hue=group1_col if group1_col != "None" else None,
                         style=group2_col if group2_col != "None" else None,
                         errorbar=('ci', 95) if ci_checkbox else None, palette=palette, ax=ax, marker='o', 
                         linestyle='-' if show_line else '')
        elif plot_type == "Scatter Plot":
            if y_col == "None": raise ValueError("Please select a Y-Axis column.")
            if ci_checkbox:
                sns.regplot(data=filtered_df, x=x_col, y=y_col, ax=ax, scatter_kws={'alpha': alpha})
            elif x_err or y_err:
                x_err_val = float(x_err) if x_err else None
                y_err_val = float(y_err) if y_err else None
                ax.errorbar(x=filtered_df[x_col], y=filtered_df[y_col], yerr=y_err_val, xerr=x_err_val, fmt='o', alpha=alpha, capsize=5)
            else:
                sns.scatterplot(data=filtered_df, x=x_col, y=y_col, hue=group1_col if group1_col != "None" else None,
                                style=group2_col if group2_col != "None" else None, palette=palette, alpha=alpha, ax=ax)

        # --- Apply Formatting (same as before) ---
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        if xticks > 0: ax.xaxis.set_major_locator(MaxNLocator(nbins=xticks))
        if yticks > 0: ax.yaxis.set_major_locator(MaxNLocator(nbins=yticks))
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        ax.set_xlim(xlim_min if xlim_min is not None else current_xlim[0], 
                    xlim_max if xlim_max is not None else current_xlim[1])
        ax.set_ylim(ylim_min if ylim_min is not None else current_ylim[0],
                    ylim_max if ylim_max is not None else current_ylim[1])
        if vlines_str:
            line_vals = [float(v.strip()) for v in vlines_str.split(',')]
            for val in line_vals:
                ax.axvline(x=val, color=vline_color, linestyle=vline_style, linewidth=vline_width, alpha=vline_alpha)

        # --- Save Figure to Buffer with selected DPI ---
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
        st.session_state.figure_buffer = buf
        plt.close(fig) 

    except (ValueError, TypeError) as e:
        st.error(f"⚠️ An error occurred: {e}")
        st.session_state.figure_buffer = None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state.figure_buffer = None

if st.session_state.figure_buffer:
    plot_col, ci_col = st.columns([0.75, 0.25])
    with plot_col:
        st.image(st.session_state.figure_buffer)
        st.download_button(
            label="📥 Download Plot as PNG",
            data=st.session_state.figure_buffer,
            file_name="plot.png",
            mime="image/png",
            use_container_width=True
        )
    with ci_col:
        if st.session_state.ci_text:
            st.subheader("95% CI")
            st.code(st.session_state.ci_text)
