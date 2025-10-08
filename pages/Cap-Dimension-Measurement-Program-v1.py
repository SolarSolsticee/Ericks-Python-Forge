# MODIFIED: Added streamlit import
import streamlit as st

import sys
import os
import json
import re
from functools import partial

import numpy as np
from skimage import io, filters, draw, util

import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QRadioButton, QFileDialog, QLabel, QGroupBox, QGridLayout, QLineEdit,
    QDialog, QDialogButtonBox, QSlider, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms


# MODIFIED: Re-introduced an enhanced ResolutionDialog for STL imports
class ResolutionDialog(QDialog):
    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Voxel Resolution for STL")
        self.pitch = None  # This will store the final pitch in mm
        self.options = options

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Select the voxel size (pitch) for the STL.\nThis choice will also update the main window's setting."))

        self.combo_box = QComboBox()
        self.combo_box.addItems(list(options.keys()) + ["Custom"])
        self.combo_box.currentTextChanged.connect(self.on_combo_change)
        layout.addWidget(self.combo_box)

        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText("Enter µm")
        self.custom_edit.setVisible(False)
        layout.addWidget(self.custom_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_combo_change(self, text):
        self.custom_edit.setVisible(text == "Custom")

    def validate_and_accept(self):
        selected_text = self.combo_box.currentText()
        if selected_text == "Custom":
            try:
                custom_val_um = float(self.custom_edit.text())
                if custom_val_um <= 0: raise ValueError
                self.pitch = custom_val_um / 1000.0
                self.accept()
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid, positive number for the custom size.")
        else:
            self.pitch = self.options[selected_text]
            self.accept()

    @staticmethod
    def get_pitch(options, parent=None):
        dialog = ResolutionDialog(options, parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.pitch, dialog.combo_box.currentText(), dialog.custom_edit.text()
        return None, None, None


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- Data Storage ---
        self.image_stack = None;
        self.best_slice = None;
        self.sagittal_slice = None
        self.coronal_slice = None;
        self.current_view = 'sagittal';
        self.center_x, self.center_y = 0, 0
        self.data_mode = None
        self.roi_selectors = []
        self.roi_patches = [];
        self.roi_text_labels = [];
        self.detected_edge_artists = []
        self.measurement_dimension_artists = []
        self.roi_colors = ['cyan', 'cyan', 'lime', 'lime', 'magenta', 'magenta']
        self.active_roi_index = 0

        self.CONVERSION_FACTORS = {'1k Large (35.6 µm)': 0.0356, '2k Med (17.8 µm)': 0.0178,
                                   '4k Small (8.9 µm)': 0.0089}
        self.conversion_factor = list(self.CONVERSION_FACTORS.values())[0]

        self.first_load = True
        self.measurements_mm = {}

        self.setWindowTitle("PyQt5 Measurement Tool");
        self.setGeometry(100, 100, 1600, 950)
        main_widget = QWidget();
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        controls_layout = self.setup_controls_panel()
        main_layout.addLayout(controls_layout, 1)
        image_layout = self.setup_image_panel()
        main_layout.addLayout(image_layout, 3)

    def setup_controls_panel(self):
        layout = QVBoxLayout();
        layout.setAlignment(Qt.AlignTop)
        file_box = QGroupBox("File Operations");
        file_layout = QGridLayout()
        self.btn_load_tiffs = QPushButton("Load TIFFs");
        self.btn_load_stl = QPushButton("Load STL")
        self.btn_load_radiograph = QPushButton("Load Radiograph")
        self.btn_save_rois = QPushButton("Save ROIs");
        self.btn_load_rois = QPushButton("Load ROIs")
        self.btn_export_raw = QPushButton("Export Raw Image");
        self.btn_export_processed = QPushButton("Export Image w/ Results")

        file_layout.addWidget(self.btn_load_tiffs, 0, 0);
        file_layout.addWidget(self.btn_load_stl, 0, 1)
        file_layout.addWidget(self.btn_load_radiograph, 0, 2)
        file_layout.addWidget(self.btn_save_rois, 1, 0);
        file_layout.addWidget(self.btn_load_rois, 1, 1)
        file_layout.addWidget(self.btn_export_raw, 2, 0);
        file_layout.addWidget(self.btn_export_processed, 2, 1, 1, 2)
        file_box.setLayout(file_layout);
        layout.addWidget(file_box)

        options_box = QGroupBox("Image Options");
        options_layout = QGridLayout()
        options_layout.addWidget(QLabel("Custom Title:"), 0, 0)
        self.title_edit = QLineEdit("Load an image to begin")
        options_layout.addWidget(self.title_edit, 0, 1, 1, 2)
        self.btn_switch_view = QPushButton("Switch to Coronal View")
        options_layout.addWidget(self.btn_switch_view, 1, 0, 1, 3)

        options_layout.addWidget(QLabel("Pixel Size:"), 2, 0)
        pixel_size_layout = QHBoxLayout()
        self.combo_conversion = QComboBox()
        self.combo_conversion.addItems(list(self.CONVERSION_FACTORS.keys()) + ["Custom"])
        pixel_size_layout.addWidget(self.combo_conversion, 2)

        self.custom_pixel_size_edit = QLineEdit()
        self.custom_pixel_size_edit.setPlaceholderText("Enter µm")
        self.custom_pixel_size_edit.setVisible(False)
        pixel_size_layout.addWidget(self.custom_pixel_size_edit, 1)
        options_layout.addLayout(pixel_size_layout, 2, 1, 1, 2)

        # NEW: Tilt compensation checkbox
        self.tilt_comp_checkbox = QCheckBox("Compensate for Tilt")
        self.tilt_comp_checkbox.setChecked(False)
        options_layout.addWidget(self.tilt_comp_checkbox, 3, 0, 1, 3)

        options_box.setLayout(options_layout);
        layout.addWidget(options_box)

        self.radiograph_controls_box = QGroupBox("Radiograph Edge Detection")
        radiograph_layout = QGridLayout()
        radiograph_layout.addWidget(QLabel("Scharr Threshold:"), 0, 0)
        self.scharr_slider = QSlider(Qt.Horizontal)
        self.scharr_slider.setRange(1, 100);
        self.scharr_slider.setValue(10)
        self.scharr_label = QLabel("0.10")
        radiograph_layout.addWidget(self.scharr_slider, 0, 1);
        radiograph_layout.addWidget(self.scharr_label, 0, 2)
        self.radiograph_controls_box.setLayout(radiograph_layout)
        layout.addWidget(self.radiograph_controls_box)
        self.radiograph_controls_box.setVisible(False)

        roi_adjust_box = QGroupBox("ROI Selection");
        roi_adjust_layout = QGridLayout()
        self.adjust_buttons = []
        for i in range(6):
            btn = QPushButton(f"Adjust ROI {i + 1}");
            self.adjust_buttons.append(btn)
            roi_adjust_layout.addWidget(btn, i // 2, i % 2)

        self.show_rois_checkbox = QCheckBox("Show ROI Boxes")
        self.show_rois_checkbox.setChecked(True)
        roi_adjust_layout.addWidget(self.show_rois_checkbox, 3, 0, 1, 2)
        roi_adjust_box.setLayout(roi_adjust_layout);
        layout.addWidget(roi_adjust_box)

        self.direction_radios = [];
        self.edge_radios = []
        pair_configs = {"Pair 1: Height": (0, 1), "Pair 2: ID1": (2, 3), "Pair 3: ID2": (4, 5)}
        for title, (r1_idx, r2_idx) in pair_configs.items():
            pair_box = QGroupBox(title);
            pair_layout = QVBoxLayout()
            dir_group = QGroupBox("Measurement Direction");
            dir_layout = QHBoxLayout()
            rb_vert = QRadioButton("Vertical");
            rb_horiz = QRadioButton("Horizontal")
            rb_vert.setChecked("Height" in title);
            rb_horiz.setChecked("Height" not in title)
            dir_layout.addWidget(rb_vert);
            dir_layout.addWidget(rb_horiz)
            dir_group.setLayout(dir_layout);
            self.direction_radios.append((rb_vert, rb_horiz))
            pair_layout.addWidget(dir_group)
            edge_group_1 = QGroupBox(f"ROI {r1_idx + 1} Edge");
            edge_layout_1 = QHBoxLayout()
            rb_min1 = QRadioButton("Min");
            rb_max1 = QRadioButton("Max");
            rb_min1.setChecked(True)
            edge_layout_1.addWidget(rb_min1);
            edge_layout_1.addWidget(rb_max1)
            edge_group_1.setLayout(edge_layout_1);
            self.edge_radios.append((rb_min1, rb_max1))
            pair_layout.addWidget(edge_group_1)
            edge_group_2 = QGroupBox(f"ROI {r2_idx + 1} Edge");
            edge_layout_2 = QHBoxLayout()
            rb_min2 = QRadioButton("Min");
            rb_max2 = QRadioButton("Max");
            rb_min2.setChecked(True)
            edge_layout_2.addWidget(rb_min2);
            edge_layout_2.addWidget(rb_max2)
            edge_group_2.setLayout(edge_layout_2);
            self.edge_radios.append((rb_min2, rb_max2))
            pair_layout.addWidget(edge_group_2)
            pair_box.setLayout(pair_layout);
            layout.addWidget(pair_box)

        copy_box = QGroupBox("Copy Measurements")
        copy_layout = QHBoxLayout()
        self.btn_copy_h = QPushButton("Copy Height")
        self.btn_copy_id1 = QPushButton("Copy ID1")
        self.btn_copy_id2 = QPushButton("Copy ID2")
        copy_layout.addWidget(self.btn_copy_h)
        copy_layout.addWidget(self.btn_copy_id1)
        copy_layout.addWidget(self.btn_copy_id2)
        copy_box.setLayout(copy_layout)
        layout.addWidget(copy_box)
        self.btn_copy_h.setEnabled(False)
        self.btn_copy_id1.setEnabled(False)
        self.btn_copy_id2.setEnabled(False)

        for btn in [self.btn_save_rois, self.btn_load_rois, self.btn_export_raw, self.btn_export_processed,
                    self.btn_switch_view, self.tilt_comp_checkbox]: btn.setEnabled(False)
        self.title_edit.setEnabled(False)

        self.btn_load_tiffs.clicked.connect(self.load_tiff_files)
        self.btn_load_stl.clicked.connect(self.load_stl_file)
        self.btn_load_radiograph.clicked.connect(self.load_radiograph_file)
        self.btn_switch_view.clicked.connect(self.switch_view)
        self.scharr_slider.valueChanged.connect(self.on_scharr_slider_change)
        self.show_rois_checkbox.stateChanged.connect(self.recalculate_and_draw)
        self.btn_save_rois.clicked.connect(self.save_rois);
        self.btn_load_rois.clicked.connect(self.load_rois)
        self.btn_export_raw.clicked.connect(self.export_raw_image);
        self.btn_export_processed.clicked.connect(self.export_processed_image)
        self.title_edit.textChanged.connect(self.update_plot_title)

        self.combo_conversion.currentTextChanged.connect(self.update_conversion_factor)
        self.custom_pixel_size_edit.editingFinished.connect(self.on_custom_pixel_size_changed)

        # NEW: Connect the tilt compensation checkbox
        self.tilt_comp_checkbox.toggled.connect(self.recalculate_and_draw)

        for i, btn in enumerate(self.adjust_buttons): btn.clicked.connect(partial(self.activate_roi, i))
        for rb_pair in self.direction_radios + self.edge_radios:
            rb_pair[0].toggled.connect(self.recalculate_and_draw)
            rb_pair[1].toggled.connect(self.recalculate_and_draw)
        self.btn_copy_h.clicked.connect(lambda: self.copy_measurement('Height'))
        self.btn_copy_id1.clicked.connect(lambda: self.copy_measurement('ID1'))
        self.btn_copy_id2.clicked.connect(lambda: self.copy_measurement('ID2'))
        return layout

    def load_radiograph_file(self):
        self.reset_state_for_new_file('RADIOGRAPH')
        fname, _ = QFileDialog.getOpenFileName(self, "Select Radiograph", "", "BMP Images (*.bmp)")
        if not fname: return
        try:
            previous_extents = [selector.extents for selector in self.roi_selectors] if self.roi_selectors else None
            self.best_slice = io.imread(fname)
            self.fig.clear();
            self.ax = self.fig.add_subplot(111)
            self.ax.imshow(self.best_slice, cmap='gray')
            if self.first_load:
                self.title_edit.setText("Radiograph")
                self.first_load = False
            for btn in [self.btn_save_rois, self.btn_load_rois, self.btn_export_raw, self.btn_export_processed,
                        self.tilt_comp_checkbox]: btn.setEnabled(True)
            self.title_edit.setEnabled(True)
            self.setup_rois(previous_extents)
        except Exception as e:
            self.ax.set_title(f"Error loading Radiograph: {e}");
            self.canvas.draw_idle()

    # NEW: Function to fit a line to the detected edge in an ROI and return its slope.
    def fit_line_to_edge_in_roi(self, roi_index):
        """Finds all edge coordinates within an ROI and fits a line, returning the slope."""
        x1, x2, y1, y2 = [int(v) for v in self.roi_selectors[roi_index].extents]
        roi_img = self.best_slice[y1:y2, x1:x2]
        if roi_img.size < 2 or np.all(roi_img == roi_img.min()): return None

        # --- Edge detection logic (mirrors find_edge_in_roi) ---
        if self.data_mode == 'RADIOGRAPH':
            roi_float = util.img_as_float(roi_img);
            edges = filters.scharr(roi_float)
            max_edge_val = np.max(edges)
            if max_edge_val == 0: return None
            threshold = max_edge_val * (self.scharr_slider.value() / 100.0)
            coords = np.argwhere(edges >= threshold)
        else:
            is_voxelized = (self.data_mode == 'STL')
            thresh = roi_img.mean() if is_voxelized else filters.threshold_otsu(roi_img)
            coords = np.argwhere(roi_img > thresh)
        # --- End edge detection logic ---

        if coords.shape[0] < 2: return None  # Not enough points to fit a line

        # Coords are (row, col) i.e., (y, x) relative to ROI. Convert to absolute image coords.
        abs_coords_y = coords[:, 0] + y1
        abs_coords_x = coords[:, 1] + x1

        # To robustly handle near-vertical lines, fit x against y if the y-range is larger.
        x_range = np.ptp(abs_coords_x)  # ptp is peak-to-peak (max-min)
        y_range = np.ptp(abs_coords_y)

        if x_range < y_range:  # Line is more vertical than horizontal
            # Fit x = m*y + c, so slope is dx/dy
            m, _ = np.polyfit(abs_coords_y, abs_coords_x, 1)
            # Return dy/dx. If m is near 0 (horizontal fit), original slope is infinite.
            return 1 / m if m != 0 else np.inf
        else:  # Line is more horizontal
            # Fit y = m*x + c, slope is dy/dx
            slope, _ = np.polyfit(abs_coords_x, abs_coords_y, 1)
            return slope

    def find_edge_in_roi(self, roi_index, axis, edge_type):
        x1, x2, y1, y2 = [int(v) for v in self.roi_selectors[roi_index].extents]
        roi_img = self.best_slice[y1:y2, x1:x2]
        if roi_img.size == 0 or np.all(roi_img == roi_img.min()): return None
        if self.data_mode == 'RADIOGRAPH':
            roi_float = util.img_as_float(roi_img);
            edges = filters.scharr(roi_float)
            max_edge_val = np.max(edges)
            if max_edge_val == 0: return None
            threshold = max_edge_val * (self.scharr_slider.value() / 100.0)
            coords = np.argwhere(edges >= threshold)
        else:
            is_voxelized = (self.data_mode == 'STL')
            thresh = roi_img.mean() if is_voxelized else filters.threshold_otsu(roi_img)
            coords = np.argwhere(roi_img > thresh)
        if coords.size == 0: return None
        if axis == 'y':
            return y1 + (np.min(coords[:, 0]) if edge_type == 'min' else np.max(coords[:, 0]))
        else:
            return x1 + (np.min(coords[:, 1]) if edge_type == 'min' else np.max(coords[:, 1]))

    def recalculate_and_draw(self):
        if self.best_slice is None or not hasattr(self, 'ax') or not self.roi_selectors: return
        if self.roi_selectors:
            self.roi_selectors[self.active_roi_index].set_visible(self.show_rois_checkbox.isChecked())
        for artist_list in [self.roi_patches, self.roi_text_labels, self.detected_edge_artists,
                            self.measurement_dimension_artists]:
            for artist in artist_list: artist.remove()
            artist_list.clear()
        for i in range(6):
            if self.show_rois_checkbox.isChecked():
                x1, x2, y1, y2 = [int(v) for v in self.roi_selectors[i].extents]
                patch = self.ax.add_patch(
                    Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=self.roi_colors[i], linewidth=2))
                self.roi_patches.append(patch)
                label_text = f"ROI {i + 1}" + (" *" if i == self.active_roi_index else "")
                label = self.ax.text(x1, y1 - 5, label_text, color=self.roi_colors[i], fontweight='bold')
                self.roi_text_labels.append(label)
            x1, x2, y1, y2 = [int(v) for v in self.roi_selectors[i].extents]
            pair_idx = i // 2
            axis = 'y' if self.direction_radios[pair_idx][0].isChecked() else 'x'
            edge_type = "min" if self.edge_radios[i][0].isChecked() else "max"
            coord = self.find_edge_in_roi(i, axis, edge_type)
            if coord is not None:
                line_data = ([x1, x2], [coord, coord]) if axis == 'y' else ([coord, coord], [y1, y2])
                line = Line2D(*line_data, color='white', linewidth=1, linestyle='--');
                self.ax.add_line(line)
                self.detected_edge_artists.append(line)
        results_px = {}
        self.measurements_mm.clear()
        dimension_data = self.calculate_dimension_data(results_px, self.measurements_mm)
        self.resolve_overlaps_and_draw_dimensions(dimension_data)

        # MODIFIED: Build the results string including tilt information
        results_parts = []
        for name in ['Height', 'ID1', 'ID2']:
            px = results_px.get(name, 'N/A')
            mm = self.measurements_mm.get(name, 'N/A')
            dim_info = next((d for d in dimension_data if d['name'] == name), None)
            tilt_str = ""
            if dim_info and 'tilt_angle_deg' in dim_info and dim_info['tilt_angle_deg'] != 0.0:
                tilt_str = f" (tilt: {dim_info['tilt_angle_deg']:.1f}°)"

            results_parts.append(f"<b>{name}:</b> {px} px ({mm} mm){tilt_str}")

        results_html = "   |   ".join(results_parts)

        if self.data_mode == 'RADIOGRAPH':
            results_html += "<br><font color='red'>Caution: Using radiographs greatly reduces data quality.</font>"
        self.results_label.setText(results_html);
        self.canvas.draw_idle()
        self.btn_copy_h.setEnabled('Height' in self.measurements_mm)
        self.btn_copy_id1.setEnabled('ID1' in self.measurements_mm)
        self.btn_copy_id2.setEnabled('ID2' in self.measurements_mm)

    def setup_image_panel(self):
        layout = QVBoxLayout()
        self.fig = Figure(tight_layout=True);
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Load a file to begin");
        self.ax.axis('off')
        layout.addWidget(self.canvas)
        self.results_label = QLabel("Results will be shown here")
        self.results_label.setAlignment(Qt.AlignCenter);
        self.results_label.setFont(QFont("Arial", 14))
        self.results_label.setStyleSheet("background-color: wheat; border: 1px solid black; padding: 10px;")
        self.results_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        layout.setStretch(0, 1);
        layout.setStretch(1, 0)
        return layout

    def process_3d_data(self):
        is_stl = (self.data_mode == 'STL')
        threshold = 0 if is_stl else filters.threshold_otsu(self.image_stack)
        areas = np.sum(self.image_stack > threshold, axis=(1, 2))
        best_axial_slice = self.image_stack[np.argmax(areas), :, :]
        horiz_proj = np.sum(best_axial_slice > threshold, axis=0);
        obj_x_coords = np.where(horiz_proj > 0)[0]
        self.center_x = int(np.median(obj_x_coords)) if obj_x_coords.size > 0 else self.image_stack.shape[2] // 2
        vert_proj = np.sum(best_axial_slice > threshold, axis=1);
        obj_y_coords = np.where(vert_proj > 0)[0]
        self.center_y = int(np.median(obj_y_coords)) if obj_y_coords.size > 0 else self.image_stack.shape[1] // 2
        sagittal_raw = self.image_stack[:, :, self.center_x];
        coronal_raw = self.image_stack[:, self.center_y, :]
        max_width = max(sagittal_raw.shape[1], coronal_raw.shape[1])

        def pad_slice(slice_data, target_width):
            h, w = slice_data.shape;
            pad_width = target_width - w;
            pad_left = pad_width // 2
            return np.pad(slice_data, ((0, 0), (pad_left, pad_width - pad_left)), 'constant')

        self.sagittal_slice = np.rot90(pad_slice(sagittal_raw, max_width), 2)
        self.coronal_slice = np.rot90(pad_slice(coronal_raw, max_width), 2)
        self.current_view = 'sagittal';
        self.best_slice = self.sagittal_slice
        for btn in [self.btn_save_rois, self.btn_load_rois, self.btn_export_raw, self.btn_export_processed,
                    self.btn_switch_view, self.tilt_comp_checkbox]: btn.setEnabled(True)
        self.title_edit.setEnabled(True);
        self.update_display_and_rois()

    def update_display_and_rois(self):
        previous_extents = [selector.extents for selector in self.roi_selectors] if self.roi_selectors else None
        self.fig.clear();
        self.ax = self.fig.add_subplot(111)
        if self.current_view == 'sagittal':
            title = f"Sagittal View (at X={self.center_x})"
        else:
            title = f"Coronal View (at Y={self.center_y})"
        if self.data_mode == 'STL':
            self.ax.imshow(self.best_slice, cmap='gray');
            title = "Voxelized " + title
        else:
            self.ax.imshow(self.best_slice, cmap='gray', vmin=0, vmax=65535)
        if self.first_load:
            self.title_edit.setText(title)
            self.first_load = False
        self.ax.axis('off')
        self.setup_rois(previous_extents)

    def setup_rois(self, previous_extents=None):
        self.roi_selectors.clear();
        self.roi_patches.clear();
        self.roi_text_labels.clear()
        self.detected_edge_artists.clear();
        self.measurement_dimension_artists.clear()
        for i in range(6):
            selector = RectangleSelector(self.ax, self.on_select, useblit=True, button=[1], minspanx=5, minspany=5,
                                         spancoords='pixels', interactive=True,
                                         props=dict(facecolor='none', edgecolor=self.roi_colors[i], linewidth=2))
            if previous_extents and i < len(previous_extents):
                try:
                    selector.extents = previous_extents[i]
                except Exception as e:
                    print(f"Could not restore ROI {i + 1} position: {e}")
            selector.set_active(False);
            self.roi_selectors.append(selector)
        self.activate_roi(0)

    def export_processed_image(self):
        base_name = self.title_edit.text()
        sanitized_name = re.sub(r'[\\/*?:"<>|]', "", base_name)
        default_filepath = f"{sanitized_name}.png"
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", default_filepath,
                                                  "PNG Image (*.png);;JPEG Image (*.jpg)")
        if not filepath: return
        artists_to_remove = []

        def clean_html(raw_html):
            return re.sub(re.compile('<.*?>'), '', raw_html)

        full_text_html = self.results_label.text()
        full_str_cleaned = clean_html(full_text_html.replace("   |   ", "\n").replace("<br>", "\n"))
        base_text_artist = self.fig.text(0.01, 0.01, full_str_cleaned, ha='left', va='bottom', fontsize=10,
                                         bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9))
        artists_to_remove.append(base_text_artist)
        if self.data_mode == 'RADIOGRAPH' and '<br>' in full_text_html:
            caution_str = clean_html(full_text_html.split('<br>')[1])
            caution_artist = self.fig.text(0.01, 0.01, caution_str, ha='left', va='bottom', fontsize=10, color='red')
            artists_to_remove.append(caution_artist)
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        for artist in artists_to_remove: artist.remove()
        self.canvas.draw_idle()
        print(f"Processed image saved to {filepath}")

    # MODIFIED: Major update to incorporate tilt compensation logic
    def calculate_dimension_data(self, results_px, results_mm):
        dimension_data = []
        pair_configs = [{"name": "Height"}, {"name": "ID1"}, {"name": "ID2"}]

        for i, config in enumerate(pair_configs):
            r1_idx, r2_idx = 2 * i, 2 * i + 1
            axis = 'y' if self.direction_radios[i][0].isChecked() else 'x'
            edge1_type = "min" if self.edge_radios[r1_idx][0].isChecked() else "max"
            edge2_type = "min" if self.edge_radios[r2_idx][0].isChecked() else "max"

            edge1_val = self.find_edge_in_roi(r1_idx, axis, edge1_type)
            edge2_val = self.find_edge_in_roi(r2_idx, axis, edge2_type)

            data = {"name": config["name"], "axis": axis, "edge1_val": edge1_val, "edge2_val": edge2_val,
                    "r1_idx": r1_idx, "r2_idx": r2_idx, "color": self.roi_colors[r1_idx]}

            if edge1_val is not None and edge2_val is not None:
                dist_px = abs(edge2_val - edge1_val)
                correction_applied = False
                tilt_angle_deg = 0.0

                if self.tilt_comp_checkbox.isChecked():
                    slope1 = self.fit_line_to_edge_in_roi(r1_idx)
                    slope2 = self.fit_line_to_edge_in_roi(r2_idx)

                    if slope1 is not None and slope2 is not None and np.isfinite(slope1) and np.isfinite(slope2):
                        avg_slope = (slope1 + slope2) / 2.0
                        # Angle of the edge from the horizontal axis
                        theta = np.arctan(avg_slope)

                        # The correction factor is always cos(angle_between_measurement_and_perpendicular)
                        # For vertical measurement, this angle is theta.
                        # For horizontal measurement, this angle is also theta.
                        dist_px *= np.cos(theta)

                        tilt_angle_deg = np.rad2deg(theta)
                        correction_applied = True

                data['tilt_angle_deg'] = tilt_angle_deg
                results_px[config["name"]] = f"{dist_px:.1f}{'*' if correction_applied else ''}"
                measurement_mm = dist_px * self.conversion_factor
                results_mm[config["name"]] = f"{measurement_mm:.4g}"
                data["mm_text"] = results_mm[config["name"]]

            dimension_data.append(data)
        return dimension_data

    def copy_measurement(self, key):
        if key in self.measurements_mm:
            value = self.measurements_mm[key]
            QApplication.clipboard().setText(str(value))
            print(f"Copied '{value}' ({key}) to clipboard.")

    def on_custom_pixel_size_changed(self):
        try:
            custom_val_um = float(self.custom_pixel_size_edit.text())
            if custom_val_um <= 0: raise ValueError("Pixel size must be positive.")
            self.conversion_factor = custom_val_um / 1000.0
            self.custom_pixel_size_edit.setStyleSheet("")
            self.recalculate_and_draw()
        except (ValueError, TypeError):
            self.custom_pixel_size_edit.setStyleSheet("background-color: #ffcccc;")

    def update_conversion_factor(self, text):
        self.custom_pixel_size_edit.setStyleSheet("")
        if text == "Custom":
            self.custom_pixel_size_edit.setVisible(True)
            self.on_custom_pixel_size_changed()
        elif text in self.CONVERSION_FACTORS:
            self.custom_pixel_size_edit.setVisible(False)
            self.conversion_factor = self.CONVERSION_FACTORS[text]
            self.recalculate_and_draw()

    def reset_state_for_new_file(self, mode):
        self.data_mode = mode;
        self.radiograph_controls_box.setVisible(mode == 'RADIOGRAPH')
        self.btn_switch_view.setVisible(mode in ['TIFF', 'STL'])

    def on_scharr_slider_change(self, value):
        self.scharr_label.setText(f"{value / 100:.2f}");
        self.recalculate_and_draw()

    def load_tiff_files(self):
        self.reset_state_for_new_file('TIFF')
        fnames, _ = QFileDialog.getOpenFileNames(self, "Select TIFF Slices", "", "TIFF Images (*.tif *.tiff)")
        if not fnames: return
        try:
            self.image_stack = np.stack([io.imread(f) for f in fnames]);
            self.process_3d_data()
        except Exception as e:
            self.ax.set_title(f"Error loading TIFFs: {e}");
            self.canvas.draw_idle()

    # Replace the entire function in your script with this one
    def load_stl_file(self):
        self.reset_state_for_new_file('STL')

        pitch, selected_key, custom_val_str = ResolutionDialog.get_pitch(self.CONVERSION_FACTORS, self)
        if pitch is None: return

        fname, _ = QFileDialog.getOpenFileName(self, "Select STL File", "", "STL Files (*.stl)")
        if not fname: return

        try:
            print(f"Loading STL with PyVista engine...")
            QApplication.processEvents()

            # 1. Load the mesh using PyVista
            mesh = pv.read(fname)

            print(f"Voxelizing with pitch {pitch:.4f} mm... This may take a moment.")
            QApplication.processEvents()

            voxels = mesh.voxelize_binary_mask(spacing=pitch)

            # 3. Convert the PyVista ImageData to a NumPy array
            dims = voxels.dimensions

            # MODIFIED: Use `.active_scalars` to get the data array regardless of its name.
            image_stack_xyz = voxels.active_scalars.reshape(dims, order='F')

            # 4. Transpose to the (depth, height, width) -> (nz, ny, nx) format our app expects
            self.image_stack = util.img_as_uint(image_stack_xyz.transpose(2, 1, 0))

            print("Voxelization complete.")

            # --- UI Synchronization ---
            self.combo_conversion.blockSignals(True)
            self.custom_pixel_size_edit.blockSignals(True)
            self.combo_conversion.setCurrentText(selected_key)
            if selected_key == "Custom":
                self.custom_pixel_size_edit.setText(custom_val_str)
            self.combo_conversion.blockSignals(False)
            self.custom_pixel_size_edit.blockSignals(False)

            self.update_conversion_factor(selected_key)
            self.process_3d_data()

        except Exception as e:
            import traceback
            print(f"Error during STL processing: {e}")
            traceback.print_exc()
            self.ax.set_title(f"Error during STL processing: {e}");
            self.canvas.draw_idle();

    def switch_view(self):
        if self.sagittal_slice is None: return
        if self.current_view == 'sagittal':
            self.current_view = 'coronal';
            self.best_slice = self.coronal_slice
            self.btn_switch_view.setText("Switch to Sagittal View")
        else:
            self.current_view = 'sagittal';
            self.best_slice = self.sagittal_slice
            self.btn_switch_view.setText("Switch to Coronal View")
        self.update_display_and_rois()

    def update_plot_title(self):
        if self.best_slice is not None and hasattr(self, 'ax'):
            self.ax.set_title(self.title_edit.text(), fontsize=16);
            self.canvas.draw_idle()

    def export_raw_image(self):
        if self.best_slice is None: return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Raw TIFF Image", "", "TIFF Image (*.tif)")
        if not filepath: return
        io.imsave(filepath, self.best_slice, check_contrast=False)
        print(f"Raw image saved to {filepath}")

    # MODIFIED: Overhauled to draw tilted dimension lines when compensation is active
    def resolve_overlaps_and_draw_dimensions(self, dimension_data):
        DIM_OFFSET = 20;
        MIN_SEPARATION = 25
        horiz_dims = [d for d in dimension_data if d["axis"] == 'x' and d.get("mm_text")]
        vert_dims = [d for d in dimension_data if d["axis"] == 'y' and d.get("mm_text")]

        is_tilted = self.tilt_comp_checkbox.isChecked()

        # --- Horizontal Dimensions ---
        for d in horiz_dims:
            _, _, _, y1_end = [int(v) for v in self.roi_selectors[d["r1_idx"]].extents];
            _, _, _, y2_end = [int(v) for v in self.roi_selectors[d["r2_idx"]].extents]
            d["natural_pos"] = max(y1_end, y2_end) + DIM_OFFSET
        horiz_dims.sort(key=lambda d: d["natural_pos"])
        for i in range(1, len(horiz_dims)):
            if horiz_dims[i]["natural_pos"] < horiz_dims[i - 1]["natural_pos"] + MIN_SEPARATION:
                horiz_dims[i]["natural_pos"] = horiz_dims[i - 1]["natural_pos"] + MIN_SEPARATION

        for d in horiz_dims:
            y_dim_line = d["natural_pos"]
            _, _, y1_start, y1_end = [int(v) for v in self.roi_selectors[d["r1_idx"]].extents]
            _, _, y2_start, y2_end = [int(v) for v in self.roi_selectors[d["r2_idx"]].extents]

            angle_deg = d.get('tilt_angle_deg', 0.0) if is_tilted else 0.0

            # Standard non-tilted lines
            ext1 = Line2D([d["edge1_val"], d["edge1_val"]], [max(y1_start, y1_end), y_dim_line], color=d["color"], lw=1)
            ext2 = Line2D([d["edge2_val"], d["edge2_val"]], [max(y2_start, y2_end), y_dim_line], color=d["color"], lw=1)
            dim = Line2D([d["edge1_val"], d["edge2_val"]], [y_dim_line, y_dim_line], color=d["color"], lw=1, marker='|',
                         markersize=8)
            text = self.ax.text((d["edge1_val"] + d["edge2_val"]) / 2, y_dim_line + 5, d["mm_text"] + ' mm',
                                color=d["color"], va='bottom', ha='center', fontsize=10, weight='bold')

            # Apply rotation if tilted
            if angle_deg != 0.0:
                cx = (d["edge1_val"] + d["edge2_val"]) / 2
                cy = y_dim_line
                t = transforms.Affine2D().rotate_deg_around(cx, cy, angle_deg) + self.ax.transData
                dim.set_transform(t)
                text.set_rotation(angle_deg)

            self.ax.add_line(ext1);
            self.ax.add_line(ext2);
            self.ax.add_line(dim)
            self.measurement_dimension_artists.extend([ext1, ext2, dim, text])

        # --- Vertical Dimensions ---
        for d in vert_dims:
            x1_start, x1_end, _, _ = [int(v) for v in self.roi_selectors[d["r1_idx"]].extents];
            x2_start, x2_end, _, _ = [int(v) for v in self.roi_selectors[d["r2_idx"]].extents]
            x_dim_line = min(x1_start, x2_start) - DIM_OFFSET

            angle_deg = d.get('tilt_angle_deg', 0.0) if is_tilted else 0.0

            # Standard non-tilted lines
            ext1 = Line2D([min(x1_start, x1_end), x_dim_line], [d["edge1_val"], d["edge1_val"]], color=d["color"], lw=1)
            ext2 = Line2D([min(x2_start, x2_end), x_dim_line], [d["edge2_val"], d["edge2_val"]], color=d["color"], lw=1)
            dim = Line2D([x_dim_line, x_dim_line], [d["edge1_val"], d["edge2_val"]], color=d["color"], lw=1, marker='|',
                         markersize=8)
            text = self.ax.text(x_dim_line - 8, (d["edge1_val"] + d["edge2_val"]) / 2, d["mm_text"] + ' mm',
                                color=d["color"], va='center', ha='right', fontsize=10, weight='bold', rotation=90)

            # Apply rotation if tilted
            if angle_deg != 0.0:
                cx = x_dim_line
                cy = (d["edge1_val"] + d["edge2_val"]) / 2
                t = transforms.Affine2D().rotate_deg_around(cx, cy, angle_deg) + self.ax.transData
                dim.set_transform(t)
                text.set_rotation(90 + angle_deg)

            self.ax.add_line(ext1);
            self.ax.add_line(ext2);
            self.ax.add_line(dim)
            self.measurement_dimension_artists.extend([ext1, ext2, dim, text])

    def activate_roi(self, index):
        self.active_roi_index = index
        for i, selector in enumerate(self.roi_selectors):
            is_active = (i == index);
            selector.set_active(is_active);
            selector.set_visible(is_active)
            style = "background-color: lightgoldenrodyellow; border: 2px solid black;" if is_active else ""
            self.adjust_buttons[i].setStyleSheet(style)
        self.recalculate_and_draw()

    def on_select(self, eclick, erelease):
        self.recalculate_and_draw()

    def save_rois(self):
        if not self.roi_selectors: return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save ROI Positions", "", "JSON Files (*.json)")
        if not filepath: return
        roi_data = [{"roi_index": i, "extents": s.extents} for i, s in enumerate(self.roi_selectors)]
        with open(filepath, 'w') as f:
            json.dump(roi_data, f, indent=4)
        print(f"ROI positions saved to {filepath}")

    def load_rois(self):
        if self.best_slice is None: return
        filepath, _ = QFileDialog.getOpenFileName(self, "Load ROI Positions", "", "JSON Files (*.json)")
        if not filepath: return
        with open(filepath, 'r') as f:
            roi_data = json.load(f)
        for data in roi_data:
            if data["roi_index"] < len(self.roi_selectors): self.roi_selectors[data["roi_index"]].extents = tuple(
                data["extents"])
        self.recalculate_and_draw()


def main():
    """Defines the Streamlit user interface."""
    st.set_page_config(layout="centered", page_title="Image Measurement Tool")

    st.title("🔬 Image Measurement Tool")
    st.markdown("Click the button below to launch the desktop application for image analysis.")

    if st.button("🚀 Launch Measurement Tool"):
        # This is the key change: Always create a new QApplication instance.
        # This ensures a clean state every time the app is launched.
        app = QApplication(sys.argv)

        window = MainWindow()
        window.show()
        # This starts the PyQt event loop.
        # The Streamlit script will pause here until the window is closed.
        app.exec_()

    st.divider()

    st.markdown("#### Supported Data Types")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 📚 TIFF Stack")
        st.markdown("- Multiple `.tif` files\n- 3D analysis\n- Sagittal/Coronal views")
    with col2:
        st.markdown("##### 📷 Radiograph")
        st.markdown("- Single `.bmp`, `.png`, `.jpg`\n- 2D measurements\n- Edge detection")
    with col3:
        st.markdown("##### 🎲 STL Model")
        st.markdown("- 3D mesh files (`.stl`)\n- Voxelization\n- Advanced analysis")

    st.divider()

    st.markdown("#### ✨ Key Features")
    st.markdown("""
    - **Interactive ROIs:** Draw up to 6 resizable Regions of Interest directly on the image.
    - **Automated Measurements:** Automatically detect edges within each ROI for precise, repeatable measurements.
    - **Tilt Compensation:** Correct for sample tilt in radiographs for more accurate vertical and horizontal dimensions.
    - **Flexible Analysis:** Configure measurement pairs for height, inner diameters, and more.
    - **Export & Save:** Export the view with annotations or save ROI configurations to a JSON file for later use.
    """)


if __name__ == '__main__':
    main()