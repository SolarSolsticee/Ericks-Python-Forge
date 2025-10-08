import sys
import os
import json
import re
from functools import partial
from io import BytesIO
import tempfile

import numpy as np
from skimage import io as skio, filters, draw, util

import pyvista as pv
import streamlit as st

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms


# --- Helper Functions ---

def get_pitch(options):
    selected_text = st.sidebar.selectbox("Select voxel size (pitch) for the STL", list(options.keys()) + ["Custom"])
    pitch = None
    custom_val_um = None
    if selected_text == "Custom":
        custom_str = st.sidebar.text_input("Enter custom size in µm", "")
        try:
            custom_val_um = float(custom_str)
            if custom_val_um > 0:
                pitch = custom_val_um / 1000.0
            else:
                st.error("Please enter a valid, positive number for the custom size.")
        except ValueError:
            if custom_str:
                st.error("Please enter a valid, positive number for the custom size.")
    else:
        pitch = options[selected_text]
    return pitch, selected_text, str(custom_val_um) if custom_val_um else ""


def fit_line_to_edge_in_roi(roi_index, best_slice, data_mode, scharr_threshold, roi_extents):
    x1, x2, y1, y2 = roi_extents[roi_index]
    roi_img = best_slice[y1:y2, x1:x2]
    if roi_img.size < 2 or np.all(roi_img == roi_img.min()): return None

    if data_mode == 'RADIOGRAPH':
        roi_float = util.img_as_float(roi_img)
        edges = filters.scharr(roi_float)
        max_edge_val = np.max(edges)
        if max_edge_val == 0: return None
        threshold = max_edge_val * scharr_threshold
        coords = np.argwhere(edges >= threshold)
    else:
        is_voxelized = (data_mode == 'STL')
        thresh = roi_img.mean() if is_voxelized else filters.threshold_otsu(roi_img)
        coords = np.argwhere(roi_img > thresh)

    if coords.shape[0] < 2: return None

    abs_coords_y = coords[:, 0] + y1
    abs_coords_x = coords[:, 1] + x1

    x_range = np.ptp(abs_coords_x)
    y_range = np.ptp(abs_coords_y)

    if x_range < y_range:
        m, _ = np.polyfit(abs_coords_y, abs_coords_x, 1)
        return 1 / m if m != 0 else np.inf
    else:
        slope, _ = np.polyfit(abs_coords_x, abs_coords_y, 1)
        return slope


def find_edge_in_roi(roi_index, axis, edge_type, best_slice, data_mode, scharr_threshold, roi_extents):
    x1, x2, y1, y2 = roi_extents[roi_index]
    roi_img = best_slice[y1:y2, x1:x2]
    if roi_img.size == 0 or np.all(roi_img == roi_img.min()): return None
    if data_mode == 'RADIOGRAPH':
        roi_float = util.img_as_float(roi_img)
        edges = filters.scharr(roi_float)
        max_edge_val = np.max(edges)
        if max_edge_val == 0: return None
        threshold = max_edge_val * scharr_threshold
        coords = np.argwhere(edges >= threshold)
    else:
        is_voxelized = (data_mode == 'STL')
        thresh = roi_img.mean() if is_voxelized else filters.threshold_otsu(roi_img)
        coords = np.argwhere(roi_img > thresh)
    if coords.size == 0: return None
    if axis == 'y':
        return y1 + (np.min(coords[:, 0]) if edge_type == 'min' else np.max(coords[:, 0]))
    else:
        return x1 + (np.min(coords[:, 1]) if edge_type == 'min' else np.max(coords[:, 1]))


def calculate_dimension_data(results_px, results_mm, direction_is_vertical, edge_is_min, roi_extents, best_slice,
                             data_mode, tilt_comp, scharr_threshold, conversion_factor, roi_colors):
    dimension_data = []
    pair_configs = [{"name": "Height"}, {"name": "ID1"}, {"name": "ID2"}]

    for i, config in enumerate(pair_configs):
        r1_idx, r2_idx = 2 * i, 2 * i + 1
        axis = 'y' if direction_is_vertical[i] else 'x'
        edge1_type = "min" if edge_is_min[r1_idx] else "max"
        edge2_type = "min" if edge_is_min[r2_idx] else "max"

        edge1_val = find_edge_in_roi(r1_idx, axis, edge1_type, best_slice, data_mode, scharr_threshold, roi_extents)
        edge2_val = find_edge_in_roi(r2_idx, axis, edge2_type, best_slice, data_mode, scharr_threshold, roi_extents)

        data = {"name": config["name"], "axis": axis, "edge1_val": edge1_val, "edge2_val": edge2_val,
                "r1_idx": r1_idx, "r2_idx": r2_idx, "color": roi_colors[r1_idx]}

        if edge1_val is not None and edge2_val is not None:
            dist_px = abs(edge2_val - edge1_val)
            correction_applied = False
            tilt_angle_deg = 0.0

            if tilt_comp:
                slope1 = fit_line_to_edge_in_roi(r1_idx, best_slice, data_mode, scharr_threshold, roi_extents)
                slope2 = fit_line_to_edge_in_roi(r2_idx, best_slice, data_mode, scharr_threshold, roi_extents)

                if slope1 is not None and slope2 is not None and np.isfinite(slope1) and np.isfinite(slope2):
                    avg_slope = (slope1 + slope2) / 2.0
                    theta = np.arctan(avg_slope)
                    dist_px *= np.cos(theta)
                    tilt_angle_deg = np.rad2deg(theta)
                    correction_applied = True

            data['tilt_angle_deg'] = tilt_angle_deg
            results_px[config["name"]] = f"{dist_px:.1f}{'*' if correction_applied else ''}"
            measurement_mm = dist_px * conversion_factor
            results_mm[config["name"]] = f"{measurement_mm:.4g}"
            data["mm_text"] = results_mm[config["name"]]

        dimension_data.append(data)
    return dimension_data


def resolve_overlaps_and_draw_dimensions(ax, dimension_data, roi_extents, tilt_comp, roi_colors):
    DIM_OFFSET = 20
    MIN_SEPARATION = 25
    horiz_dims = [d for d in dimension_data if d["axis"] == 'x' and d.get("mm_text")]
    vert_dims = [d for d in dimension_data if d["axis"] == 'y' and d.get("mm_text")]

    is_tilted = tilt_comp

    # --- Horizontal Dimensions ---
    for d in horiz_dims:
        _, _, y1_end = roi_extents[d["r1_idx"]][3]
        _, _, y2_end = roi_extents[d["r2_idx"]][3]
        d["natural_pos"] = max(y1_end, y2_end) + DIM_OFFSET
    horiz_dims.sort(key=lambda d: d["natural_pos"])
    for i in range(1, len(horiz_dims)):
        if horiz_dims[i]["natural_pos"] < horiz_dims[i - 1]["natural_pos"] + MIN_SEPARATION:
            horiz_dims[i]["natural_pos"] = horiz_dims[i - 1]["natural_pos"] + MIN_SEPARATION

    for d in horiz_dims:
        y_dim_line = d["natural_pos"]
        _, _, y1_start, y1_end = roi_extents[d["r1_idx"]]
        _, _, y2_start, y2_end = roi_extents[d["r2_idx"]]

        angle_deg = d.get('tilt_angle_deg', 0.0) if is_tilted else 0.0

        ext1 = Line2D([d["edge1_val"], d["edge1_val"]], [max(y1_start, y1_end), y_dim_line], color=d["color"], lw=1)
        ext2 = Line2D([d["edge2_val"], d["edge2_val"]], [max(y2_start, y2_end), y_dim_line], color=d["color"], lw=1)
        dim = Line2D([d["edge1_val"], d["edge2_val"]], [y_dim_line, y_dim_line], color=d["color"], lw=1, marker='|',
                     markersize=8)
        text = ax.text((d["edge1_val"] + d["edge2_val"]) / 2, y_dim_line + 5, d["mm_text"] + ' mm',
                       color=d["color"], va='bottom', ha='center', fontsize=10, weight='bold')

        if angle_deg != 0.0:
            cx = (d["edge1_val"] + d["edge2_val"]) / 2
            cy = y_dim_line
            t = transforms.Affine2D().rotate_deg_around(cx, cy, angle_deg) + ax.transData
            dim.set_transform(t)
            text.set_rotation(angle_deg)

        ax.add_line(ext1)
        ax.add_line(ext2)
        ax.add_line(dim)
        ax.texts.append(text)

    # --- Vertical Dimensions ---
    for d in vert_dims:
        x1_start, x1_end, _, _ = roi_extents[d["r1_idx"]]
        x2_start, x2_end, _, _ = roi_extents[d["r2_idx"]]
        x_dim_line = min(x1_start, x2_start) - DIM_OFFSET

        angle_deg = d.get('tilt_angle_deg', 0.0) if is_tilted else 0.0

        ext1 = Line2D([min(x1_start, x1_end), x_dim_line], [d["edge1_val"], d["edge1_val"]], color=d["color"], lw=1)
        ext2 = Line2D([min(x2_start, x2_end), x_dim_line], [d["edge2_val"], d["edge2_val"]], color=d["color"], lw=1)
        dim = Line2D([x_dim_line, x_dim_line], [d["edge1_val"], d["edge2_val"]], color=d["color"], lw=1, marker='|',
                     markersize=8)
        text = ax.text(x_dim_line - 8, (d["edge1_val"] + d["edge2_val"]) / 2, d["mm_text"] + ' mm',
                       color=d["color"], va='center', ha='right', fontsize=10, weight='bold', rotation=90)

        if angle_deg != 0.0:
            cx = x_dim_line
            cy = (d["edge1_val"] + d["edge2_val"]) / 2
            t = transforms.Affine2D().rotate_deg_around(cx, cy, angle_deg) + ax.transData
            dim.set_transform(t)
            text.set_rotation(90 + angle_deg)

        ax.add_line(ext1)
        ax.add_line(ext2)
        ax.add_line(dim)
        ax.texts.append(text)


def process_3d_data():
    image_stack = st.session_state.image_stack
    data_mode = st.session_state.data_mode
    is_stl = (data_mode == 'STL')
    threshold = 0 if is_stl else filters.threshold_otsu(image_stack)
    areas = np.sum(image_stack > threshold, axis=(1, 2))
    best_axial_slice = image_stack[np.argmax(areas), :, :]
    horiz_proj = np.sum(best_axial_slice > threshold, axis=0)
    obj_x_coords = np.where(horiz_proj > 0)[0]
    st.session_state.center_x = int(np.median(obj_x_coords)) if obj_x_coords.size > 0 else image_stack.shape[2] // 2
    vert_proj = np.sum(best_axial_slice > threshold, axis=1)
    obj_y_coords = np.where(vert_proj > 0)[0]
    st.session_state.center_y = int(np.median(obj_y_coords)) if obj_y_coords.size > 0 else image_stack.shape[1] // 2
    sagittal_raw = image_stack[:, :, st.session_state.center_x]
    coronal_raw = image_stack[:, st.session_state.center_y, :]
    max_width = max(sagittal_raw.shape[1], coronal_raw.shape[1])

    def pad_slice(slice_data, target_width):
        h, w = slice_data.shape
        pad_width = target_width - w
        pad_left = pad_width // 2
        return np.pad(slice_data, ((0, 0), (pad_left, pad_width - pad_left)), 'constant')

    st.session_state.sagittal_slice = np.rot90(pad_slice(sagittal_raw, max_width), 2)
    st.session_state.coronal_slice = np.rot90(pad_slice(coronal_raw, max_width), 2)
    st.session_state.current_view = 'sagittal'
    st.session_state.best_slice = st.session_state.sagittal_slice
    update_display_and_rois()


def update_display_and_rois():
    current_view = st.session_state.current_view
    data_mode = st.session_state.data_mode
    if current_view == 'sagittal':
        title = f"Sagittal View (at X={st.session_state.center_x})"
        st.session_state.best_slice = st.session_state.sagittal_slice
    else:
        title = f"Coronal View (at Y={st.session_state.center_y})"
        st.session_state.best_slice = st.session_state.coronal_slice
    if st.session_state.first_load:
        st.session_state.custom_title = title
        st.session_state.first_load = False
    if data_mode == 'STL':
        title = "Voxelized " + title


def recalculate_and_draw(best_slice, data_mode, show_rois, roi_extents, active_roi_index, roi_colors,
                         direction_is_vertical, edge_is_min, tilt_comp, scharr_threshold, conversion_factor):
    if best_slice is None: return None, {}
    fig = Figure(tight_layout=True)
    ax = fig.add_subplot(111)
    if data_mode == 'STL':
        ax.imshow(best_slice, cmap='gray')
    elif data_mode == 'TIFF':
        ax.imshow(best_slice, cmap='gray', vmin=0, vmax=65535)
    else:
        ax.imshow(best_slice, cmap='gray')
    ax.set_title(st.session_state.custom_title, fontsize=16)
    ax.axis('off')

    detected_edge_artists = []
    for i in range(6):
        x1, x2, y1, y2 = roi_extents[i]
        if show_rois:
            patch = ax.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=roi_colors[i], linewidth=2))
            label_text = f"ROI {i + 1}" + (" *" if i == active_roi_index else "")
            label = ax.text(x1, y1 - 5, label_text, color=roi_colors[i], fontweight='bold')

        pair_idx = i // 2
        axis = 'y' if direction_is_vertical[pair_idx] else 'x'
        edge_type = "min" if edge_is_min[i] else "max"
        coord = find_edge_in_roi(i, axis, edge_type, best_slice, data_mode, scharr_threshold, roi_extents)
        if coord is not None:
            line_data = ([x1, x2], [coord, coord]) if axis == 'y' else ([coord, coord], [y1, y2])
            line = Line2D(*line_data, color='white', linewidth=1, linestyle='--')
            ax.add_line(line)
            detected_edge_artists.append(line)

    results_px = {}
    results_mm = {}
    dimension_data = calculate_dimension_data(results_px, results_mm, direction_is_vertical, edge_is_min, roi_extents,
                                              best_slice, data_mode, tilt_comp, scharr_threshold, conversion_factor,
                                              roi_colors)
    resolve_overlaps_and_draw_dimensions(ax, dimension_data, roi_extents, tilt_comp, roi_colors)

    return fig, results_mm


# --- Streamlit App ---

st.title("Measurement Tool")

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['image_stack'] = None
    st.session_state['best_slice'] = None
    st.session_state['sagittal_slice'] = None
    st.session_state['coronal_slice'] = None
    st.session_state['current_view'] = 'sagittal'
    st.session_state['center_x'] = 0
    st.session_state['center_y'] = 0
    st.session_state['data_mode'] = None
    st.session_state['roi_extents'] = [[100, 200, 100, 200] for _ in range(6)]  # Default extents
    st.session_state['active_roi_index'] = 0
    st.session_state['CONVERSION_FACTORS'] = {'1k Large (35.6 µm)': 0.0356, '2k Med (17.8 µm)': 0.0178,
                                              '4k Small (8.9 µm)': 0.0089}
    st.session_state['conversion_factor'] = 0.0356
    st.session_state['first_load'] = True
    st.session_state['measurements_mm'] = {}
    st.session_state['show_rois'] = True
    st.session_state['tilt_comp'] = False
    st.session_state['scharr_threshold'] = 0.10
    st.session_state['custom_title'] = "Load an image to begin"
    st.session_state['direction_is_vertical'] = [True, False, False]
    st.session_state['edge_is_min'] = [True, True, True, True, True, True]
    st.session_state['roi_colors'] = ['cyan', 'cyan', 'lime', 'lime', 'magenta', 'magenta']
    st.session_state['pixel_select'] = list(st.session_state['CONVERSION_FACTORS'].keys())[0]
    st.session_state['custom_um'] = ""

sidebar = st.sidebar

sidebar.header("File Operations")

tiffs = sidebar.file_uploader("Load TIFFs", type=['tif', 'tiff'], accept_multiple_files=True)
if tiffs:
    try:
        st.session_state.image_stack = np.stack([skio.imread(t.getvalue()) for t in tiffs])
        st.session_state.data_mode = 'TIFF'
        process_3d_data()
    except Exception as e:
        st.error(f"Error loading TIFFs: {e}")

stl = sidebar.file_uploader("Load STL", type=['stl'])
if stl:
    pitch, selected_key, custom_val_str = get_pitch(st.session_state.CONVERSION_FACTORS)
    if pitch and sidebar.button("Confirm Load STL"):
        try:
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as temp:
                temp.write(stl.getvalue())
                temp_path = temp.name
            mesh = pv.read(temp_path)
            voxels = mesh.voxelize(spacing=pitch)
            dims = voxels.dimensions
            image_stack_xyz = voxels.active_scalars.reshape(dims, order='F')
            st.session_state.image_stack = util.img_as_uint(image_stack_xyz.transpose(2, 1, 0))
            st.session_state.data_mode = 'STL'
            st.session_state.pixel_select = selected_key
            if selected_key == "Custom":
                st.session_state.custom_um = custom_val_str
            st.session_state.conversion_factor = pitch
            process_3d_data()
            os.unlink(temp_path)
        except Exception as e:
            st.error(f"Error during STL processing: {e}")

radiograph = sidebar.file_uploader("Load Radiograph", type=['bmp'])
if radiograph:
    try:
        st.session_state.best_slice = skio.imread(radiograph.getvalue())
        st.session_state.data_mode = 'RADIOGRAPH'
        if st.session_state.first_load:
            st.session_state.custom_title = "Radiograph"
            st.session_state.first_load = False
    except Exception as e:
        st.error(f"Error loading Radiograph: {e}")

sidebar.header("Image Options")
st.session_state.custom_title = sidebar.text_input("Custom Title", st.session_state.custom_title)

options = list(st.session_state.CONVERSION_FACTORS.keys()) + ["Custom"]
st.session_state.pixel_select = sidebar.selectbox("Pixel Size", options,
                                                  index=options.index(st.session_state.pixel_select))
if st.session_state.pixel_select == "Custom":
    st.session_state.custom_um = sidebar.text_input("Enter µm", st.session_state.custom_um)
    try:
        custom_val_um = float(st.session_state.custom_um)
        if custom_val_um > 0:
            st.session_state.conversion_factor = custom_val_um / 1000.0
    except ValueError:
        if st.session_state.custom_um:
            sidebar.error("Invalid custom pixel size.")
else:
    st.session_state.conversion_factor = st.session_state.CONVERSION_FACTORS[st.session_state.pixel_select]

st.session_state.tilt_comp = sidebar.checkbox("Compensate for Tilt", st.session_state.tilt_comp)

if st.session_state.data_mode == 'RADIOGRAPH':
    st.session_state.scharr_threshold = sidebar.slider("Scharr Threshold", 0.01, 1.0, st.session_state.scharr_threshold,
                                                       0.01)

st.session_state.show_rois = sidebar.checkbox("Show ROI Boxes", st.session_state.show_rois)

sidebar.header("ROI Selection")
roi_options = [f"ROI {i + 1}" for i in range(6)]
selected_roi = sidebar.selectbox("Adjust ROI", roi_options, index=st.session_state.active_roi_index)
st.session_state.active_roi_index = roi_options.index(selected_roi)

if st.session_state.best_slice is not None:
    img_height, img_width = st.session_state.best_slice.shape
    idx = st.session_state.active_roi_index
    col1, col2 = sidebar.columns(2)
    with col1:
        x_min = st.number_input("x_min", min_value=0, max_value=img_width, value=st.session_state.roi_extents[idx][0],
                                step=1)
    with col2:
        y_min = st.number_input("y_min", min_value=0, max_value=img_height, value=st.session_state.roi_extents[idx][2],
                                step=1)
    x_max = st.number_input("x_max", min_value=x_min, max_value=img_width, value=st.session_state.roi_extents[idx][1],
                            step=1)
    y_max = st.number_input("y_max", min_value=y_min, max_value=img_height, value=st.session_state.roi_extents[idx][3],
                            step=1)
    st.session_state.roi_extents[idx] = [x_min, x_max, y_min, y_max]

pair_configs = {"Pair 1: Height": (0, 1), "Pair 2: ID1": (2, 3), "Pair 3: ID2": (4, 5)}
for title, (r1_idx, r2_idx) in pair_configs.items():
    pair_idx = r1_idx // 2
    with sidebar.expander(title):
        dir_label = "Measurement Direction"
        selected_dir = st.radio(dir_label, ("Vertical", "Horizontal"),
                                index=0 if st.session_state.direction_is_vertical[pair_idx] else 1,
                                key=f"dir_radio_{pair_idx}")
        st.session_state.direction_is_vertical[pair_idx] = selected_dir == "Vertical"

        edge_label1 = f"ROI {r1_idx + 1} Edge"
        selected_edge1 = st.radio(edge_label1, ("Min", "Max"), index=0 if st.session_state.edge_is_min[r1_idx] else 1,
                                  key=f"edge_radio_{r1_idx}")
        st.session_state.edge_is_min[r1_idx] = selected_edge1 == "Min"

        edge_label2 = f"ROI {r2_idx + 1} Edge"
        selected_edge2 = st.radio(edge_label2, ("Min", "Max"), index=0 if st.session_state.edge_is_min[r2_idx] else 1,
                                  key=f"edge_radio_{r2_idx}")
        st.session_state.edge_is_min[r2_idx] = selected_edge2 == "Min"

sidebar.header("Copy Measurements")
col_h, col_id1, col_id2 = sidebar.columns(3)
with col_h:
    st.text_input("Height", value=st.session_state.measurements_mm.get('Height', 'N/A'), disabled=True)
with col_id1:
    st.text_input("ID1", value=st.session_state.measurements_mm.get('ID1', 'N/A'), disabled=True)
with col_id2:
    st.text_input("ID2", value=st.session_state.measurements_mm.get('ID2', 'N/A'), disabled=True)

if st.session_state.data_mode in ['TIFF', 'STL']:
    switch_text = "Switch to Coronal View" if st.session_state.current_view == 'sagittal' else "Switch to Sagittal View"
    if sidebar.button(switch_text):
        st.session_state.current_view = 'coronal' if st.session_state.current_view == 'sagittal' else 'sagittal'
        update_display_and_rois()

sidebar.header("ROI Save/Load")
if sidebar.button("Save ROIs"):
    roi_data = [{"roi_index": i, "extents": st.session_state.roi_extents[i]} for i in range(6)]
    json_bytes = json.dumps(roi_data, indent=4).encode('utf-8')
    st.download_button("Download ROIs.json", json_bytes, file_name="rois.json", mime="application/json")

rois_json = sidebar.file_uploader("Load ROIs", type=['json'])
if rois_json:
    try:
        roi_data = json.load(rois_json)
        for data in roi_data:
            if data["roi_index"] < 6:
                st.session_state.roi_extents[data["roi_index"]] = data["extents"]
    except Exception as e:
        st.error(f"Error loading ROIs: {e}")

sidebar.header("Export Images")
if sidebar.button("Export Raw Image"):
    if st.session_state.best_slice is not None:
        bio = BytesIO()
        skio.imsave(bio, st.session_state.best_slice, format='tiff')
        st.download_button("Download raw.tif", bio.getvalue(), file_name="raw.tif", mime="image/tiff")

if sidebar.button("Export Image w/ Results"):
    if st.session_state.best_slice is not None:
        fig, _ = recalculate_and_draw(st.session_state.best_slice, st.session_state.data_mode,
                                      st.session_state.show_rois, st.session_state.roi_extents,
                                      st.session_state.active_roi_index, st.session_state.roi_colors,
                                      st.session_state.direction_is_vertical, st.session_state.edge_is_min,
                                      st.session_state.tilt_comp, st.session_state.scharr_threshold,
                                      st.session_state.conversion_factor)

        # Add results text to the figure for export
        results_parts = []
        for name in ['Height', 'ID1', 'ID2']:
            px = st.session_state.results_px.get(name, 'N/A')
            mm = st.session_state.measurements_mm.get(name, 'N/A')
            dim_info = next((d for d in st.session_state.dimension_data if d['name'] == name), None)
            tilt_str = ""
            if dim_info and 'tilt_angle_deg' in dim_info and dim_info['tilt_angle_deg'] != 0.0:
                tilt_str = f" (tilt: {dim_info['tilt_angle_deg']:.1f}°)"
            results_parts.append(f"{name}: {px} px ({mm} mm){tilt_str}")
        full_str = " | ".join(results_parts)
        fig.text(0.01, 0.01, full_str, ha='left', va='bottom', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9))

        if st.session_state.data_mode == 'RADIOGRAPH':
            fig.text(0.01, 0.05, "Caution: Using radiographs greatly reduces data quality.", ha='left', va='bottom',
                     fontsize=10, color='red')

        bio = BytesIO()
        fig.savefig(bio, format='png', dpi=300, bbox_inches='tight')
        st.download_button("Download processed.png", bio.getvalue(), file_name="processed.png", mime="image/png")

# --- Main Display ---
if st.session_state.best_slice is not None:
    st.session_state.fig, st.session_state.measurements_mm = recalculate_and_draw(st.session_state.best_slice,
                                                                                  st.session_state.data_mode,
                                                                                  st.session_state.show_rois,
                                                                                  st.session_state.roi_extents,
                                                                                  st.session_state.active_roi_index,
                                                                                  st.session_state.roi_colors,
                                                                                  st.session_state.direction_is_vertical,
                                                                                  st.session_state.edge_is_min,
                                                                                  st.session_state.tilt_comp,
                                                                                  st.session_state.scharr_threshold,
                                                                                  st.session_state.conversion_factor)

    # Temporary storage for results_px and dimension_data since not in original session
    st.session_state.results_px = {}
    st.session_state.dimension_data = calculate_dimension_data(st.session_state.results_px,
                                                               st.session_state.measurements_mm,
                                                               st.session_state.direction_is_vertical,
                                                               st.session_state.edge_is_min,
                                                               st.session_state.roi_extents,
                                                               st.session_state.best_slice, st.session_state.data_mode,
                                                               st.session_state.tilt_comp,
                                                               st.session_state.scharr_threshold,
                                                               st.session_state.conversion_factor,
                                                               st.session_state.roi_colors)

    st.pyplot(st.session_state.fig)

    results_parts = []
    for name in ['Height', 'ID1', 'ID2']:
        px = st.session_state.results_px.get(name, 'N/A')
        mm = st.session_state.measurements_mm.get(name, 'N/A')
        dim_info = next((d for d in st.session_state.dimension_data if d['name'] == name), None)
        tilt_str = ""
        if dim_info and 'tilt_angle_deg' in dim_info and dim_info['tilt_angle_deg'] != 0.0:
            tilt_str = f" (tilt: {dim_info['tilt_angle_deg']:.1f}°)"
        results_parts.append(f"<b>{name}:</b> {px} px ({mm} mm){tilt_str}")
    results_html = "   |   ".join(results_parts)
    if st.session_state.data_mode == 'RADIOGRAPH':
        results_html += "<br><font color='red'>Caution: Using radiographs greatly reduces data quality.</font>"
    st.markdown(results_html, unsafe_allow_html=True)
else:
    st.write("Load a file to begin.")