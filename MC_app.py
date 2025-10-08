import csv
import glob
import os
import sys ;

import pandas as pd
from openpyxl.styles.builtins import total

sys.setrecursionlimit(sys.getrecursionlimit() * 5)
#from tkinter import Tk, messagebox
#from tkinter.filedialog import askopenfilename, asksaveasfilename
from traceback import print_exc
from scipy.signal import savgol_filter

#import numpy as np
from PyQt5 import QtWidgets as qtw
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg


from MC import *
from MC_gui import Ui_Form


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class Mainwindow(qtw.QWidget, Ui_Form):
    def __init__(self):
        '''
        Main Window Constructor
        '''
        super().__init__()
        self.setupUi(self)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas3 = MplCanvas(self, width=5, height=4, dpi=100)
        # Create a plot window
        self.plot_widget = pg.PlotWidget()
        self.plot_derivative.addWidget(self.plot_widget)
        self.plot_RawData.addWidget(self.canvas)
        self.plot_selectedRegion.addWidget(self.canvas2)
        self.plot_RefinedCurve.addWidget(self.canvas3)
        # self.plot_RefinedCurve.addWidget(self.canvas)
        # self.plot_selectedRegion.addWidget(self.canvas)
        # self.btn_Calculate.clicked.connect(self.calcRankine)
        self.setWindowTitle('Modulus Calculator v2.4')
        self.btn_calc.clicked.connect(self.calculate)
        # self.btn_calc.clicked.connect(self.plotting)
        self.btn_import.clicked.connect(self.importfile)
        self.updatesheet_btn.clicked.connect(self.updateSheet)
        self.show()
        self.x = []
        self.cleanx = []
        self.cleany = []
        self.selected_x = []
        self.selected_y = []
        self.y = []
        self.ext = []
        self.filename = []
        self.btn_denoise.clicked.connect(self.denoise)
        self.btn_export.clicked.connect(self.export)
        self.export_stress = []  # y for ccv
        self.export_strain = []  # x for ccv
        self.strainCol = self.le_strainCol.text()
        self.stressCol = self.le_stressCol.text()
        self.stress_min = int(self.le_stressMin.text())
        self.stress_max = int(self.le_stressMax.text())
        self.btn_Fig_exp.clicked.connect(self.exportFig)
        self.update_min_max_btn.clicked.connect(self.update_range)

    # main UI code goes here

    def update_range(self):
        self.stress_min = int(self.le_stressMin.text())
        self.stress_max = int(self.le_stressMax.text())
        print(self.stress_max, self.stress_min)

    def plotting(self):
        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(self.x, self.y, 'r')
        self.canvas.axes.set_xlabel('Strain[mm/mm]')
        self.canvas.axes.set_ylabel('Stress[MPa]')
        self.canvas.draw()
        self.derivative_calc(self.x, self.y)

    def export(self):
        Tk().withdraw()
        files = [('CSV', '*.csv'), ('All Files', '*.*')]
        try:
            export_file = asksaveasfilename(filetypes=files, defaultextension=('.csv'))
            dict = {'strain': self.export_strain, 'stress': self.export_stress}
            df = pd.DataFrame(dict)
            df.to_csv(export_file)
        except:
            print('Export Failed')

    def exportFig(self):
        Tk().withdraw()
        files = [('PNG', '*.png'), ('All Files', '*.*')]
        try:
            export_file = asksaveasfilename(filetypes=files, defaultextension=('.png'))
            self.canvas3.figure.savefig(export_file)
        except:
            print('Refined Curve Failed to Save :(')

    # def plotting(self):
    #    plt.style.use('mpl15')
    #    fig, ax = plt.subplots()
    #    ax.plot(self.x, self.y, linewidth=2.0)
    #    plt.xlabel('Strain[mm/mm]')
    #    plt.ylabel('Stress[Mpa]')
    #    self.canvas.draw()
    #    self.canvas.
    #    self.canvas.show()
    '''
    def csv2xlsx(self, filename):
        for csvfile in glob.glob(os.path.join(filename, '*.csv')):
            workbook = Workbook(csvfile[:-4] + '.xlsx')
            worksheet = workbook.add_worksheet()
            with open(csvfile, 'rt', encoding='utf8') as f:
                reader = csv.reader(f)
                for r, row in enumerate(reader):
                    for c, col in enumerate(row):
                        worksheet.write(r, c, col)
            workbook.close()
    '''

    def clear_application(self):

        try:
            self.canvas2.axes.cla()
            self.canvas.axes.cla()
            self.canvas3.axes.cla()
            self.plot_widget.destroy()
            self.strainrange_le.setText('')
            self.modulus_le.setText('')
            self.r_value_le.setText('')
            self.Intercept_le.setText('')
            self.le_percentDiff.setText('')
            self.le_Ut.setText('')
            self.le_elongation.setText('')
            self.le_window_size.setText('')
            self.le_iterations.setText('')
            self.plotting()

        except:
            print('couldnt fully clean, we dont know who built the silo')
            error = 'couldnt fully clean, we dont know who built the silo'
            self.messagebox(error)
            Tk().withdraw()

    def messagebox(self, str):
        print('msgbox active')
        main = Tk()
        main.withdraw()
        main.geometry("500x400+300+300")

        main.geometry("0x0")
        messagebox.showinfo("Information", "%s" % str)

        main.mainloop()
        main.withdraw()
        main.destroy()



    def importfile(self):

        self.strainCol = self.le_strainCol.text()
        self.stressCol = self.le_stressCol.text()
        Tk().withdraw()

        try:
            request = Tk()
            request.withdraw()
            filename = askopenfilename()
            request.update()
            request.withdraw()
            # find ways to automate convert csv -> xlsx
            # read directly from csv(may require a new function)
            # break import fiel function apart from the filename, return a string(directory)
            self.filename_le.setText(str(filename))
            self.filename = filename
            print(filename)
            request.destroy()
            # csvfile=pd.read_csv('%s' % filename)
            #csvfile = pd.read_csv('%s' % str(filename))
            #print(csvfile)
            #csvfile.to_excel(filename, sheet_name=str(self.sheetname_le.text()), index=False)
        except:
            print('exception test')
            error = 'No file Selected'
            self.messagebox(error)
            Tk().withdraw()
            filename = None

        try:
            df = pd.read_excel(filename, sheet_name=str(self.sheetname_le.text()), index_col=None, header=2)
            print(df.head())  # shows preview of loaded data
            print(int(self.strainCol), int(self.stressCol))
            Tk().withdraw()
            Tk().destroy()
        except:
            print('Needs to be a .xlsx, and matching Sheet Name')
            print_exc()
            error = 'Needs to be a .xlsx, and matching Sheet Name'
            self.messagebox(error)
            Tk().withdraw()
            Tk().destroy()
            df = None

        try:
            x = list(df.iloc[:, int(self.strainCol)])  # strain column default: 2
            y = list(df.iloc[:, int(self.stressCol)])  # stress column default: 1
            print(len(x), 'data points')
            print(len(y), 'data pts')
            self.x = x
            self.y = y
            if len(self.x) > 20000:
                n_components = int(len(self.x) * 0.5)
                dfx = pd.DataFrame(self.x, columns=['Numbers'])
                dfy = pd.DataFrame(self.y, columns=['Numbers'])
                self.x = downscale_dataset(dfx, n_components)
                self.y = downscale_dataset(dfy, n_components)
                print('Data Exceeds 20k pts, downscaling by 50%: %s pts' % (len(self.x)))
            self.cleanx = x
            self.cleany = y
            self.clear_application()
            Tk().withdraw()
        except:
            print('import failed')
            error = 'import failed'
            Tk().withdraw()
            self.messagebox(error)

        if self.le_extension is None:
            pass
        else:
            try:
                ext_list = self.le_extension.text()
                self.ext = list(df.iloc[:, int(ext_list)])  # stress column default: 1
            except:
                error = 'extension error'
                Tk().withdraw()
                self.messagebox(error)
        Tk().withdraw()
        Tk().destroy()
    def derivative_calc(self, x, y):

        self.plot_widget.clear()
        x1, y1 = remove_beyond_max_y(x, y)
        x2, y2 = make_differentiable(x1, y1)

        x_smooth, y_smooth, dydx, d2ydx2, best_x, best_y, variance, total_variance = self.analyze_derivatives(x2, y2)

        print('variance:', variance)
        print('total variance: ', total_variance)
        print('variance%', variance/total_variance * 100, '%')

        self.selected_x = best_x
        self.selected_y = self.y

        self.plot_widget.plot(x_smooth, d2ydx2, pen='r', name = '2nd Derivative')
        self.plot_widget.getPlotItem().getViewBox().enableAutoRange()
        # if best_x is not None and best_y is not None:
        #     scatter = pg.ScatterPlotItem(pen='c', symbol='o')
        #     scatter.setData(best_x, best_y)
        #     self.plot_widget.plot.addItem(scatter)

        if best_x is not None and best_y is not None:
            p1 = (min(best_x), max(best_y)) #top left
            p2 = (max(best_x), max(best_y))
            p4 = (min(best_x), min(best_y))
            p3 = (max(best_x), min(best_y)) # bottom right
            x_box = (p1[0], p2[0], p3[0], p4[0], p1[0])
            y_box = (p1[1], p2[1], p3[1], p4[1], p1[1])


            self.plot_widget.plot(x_box, y_box, pen=pg.mkPen('b', width=2), symbol=None)

        '''
        # Calculate first derivative
        dy_dx = np.gradient(y, x)

        # Calculate second derivative
        d2y_dx2 = np.gradient(dy_dx, x)
        self.plot_widget.plot(x, d2y_dx2, pen='r')
        self.plot_widget.getPlotItem().getViewBox().enableAutoRange()
        '''

    def analyze_derivatives_AI(self, x, y, poly=3, min_segment_length=5, variance_percentile_threshold=20):
        """
        Computes the first and second derivatives of y with respect to x,
        and finds the region with low variance, prioritizing length.

        This function smooths the data using a Savitzky-Golay filter,
        computes the first and second derivatives, and then identifies
        the segment of the second derivative that exhibits low variance
        over the widest possible range. Segments are initially filtered
        based on a percentile threshold of their variance relative to
        all segment variances. Among the segments passing the filter,
        the one with the greatest length is selected. If there are ties
        in length, the segment with the lowest variance is chosen.

        Parameters:
            x (array-like): Input x values.
            y (array-like): Input y values.
            poly (int): Polynomial order for Savitzky-Golay filter (default: 3).
            min_segment_length (int): Minimum length of segments to consider (default: 5).
                                      Must be at least 2 to compute variance.
            variance_percentile_threshold (float): Segments with variance above this percentile
                                                   of all segment variances are filtered out.
                                                   Value should be between 0 and 100 (default: 20).

        Returns:
            tuple: (x_smooth, y_smooth, dydx, d2ydx2, best_segment_x, best_segment_d2ydx2,
                    best_segment_variance, best_segment_length, all_segment_variances)
                   Returns None for segment-related outputs if no suitable segment is found
                   or if there is insufficient data for analysis.
        """

        # Ensure numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Remove duplicate x values and corresponding y values
        unique_x, unique_indices = np.unique(x, return_index=True)
        y = y[unique_indices]
        x = unique_x

        n_points = len(x)

        # Scale window size based on data length
        window = max(5, n_points // 20)
        # Ensure window size is odd for Savitzky-Golay filter
        if window % 2 == 0:
            window += 1

        # Handle insufficient data for smoothing and derivatives
        # Need at least window points for smoothing, and poly < window
        # Need at least 2 points for np.gradient
        if n_points < max(window, poly + 2) or n_points < 2:
            print("Warning: Not enough data points for effective analysis.")
            return x, y, None, None, None, None, None, 0, None

        # Ensure window size is within bounds and odd
        if window > n_points:
            window = n_points if n_points % 2 != 0 else n_points - 1
        if window < 3 and n_points >= 3:  # Ensure minimum window if enough data for it
            window = 3
        elif window < 1:  # Should ideally not happen with initial max(5, ...)
            window = 1

        # Ensure poly order is less than window size
        if poly >= window:
            poly = window - 1
            if poly < 0: poly = 0

        y_smooth = savgol_filter(y, window, poly)

        # Compute first and second derivatives
        dydx = np.gradient(y_smooth, x)
        d2ydx2 = np.gradient(dydx, x)

        # --- Find segments and their variances ---
        segment_variances = []
        segment_info = []  # Store (variance, start_index, end_index, length)

        # Ensure minimum segment length is at least 2 for variance calculation
        min_segment_length = max(2, min_segment_length)

        # Iterate through possible segment lengths
        for current_segment_length in range(min_segment_length, len(d2ydx2) + 1):
            # Slide a window of current_segment_length across the second derivative
            for i in range(len(d2ydx2) - current_segment_length + 1):
                segment = d2ydx2[i: i + current_segment_length]
                variance = np.var(segment)
                segment_variances.append(variance)
                segment_info.append((variance, i, i + current_segment_length, current_segment_length))

        if not segment_variances:  # No segments found (e.g., data too short for min_segment_length)
            print("Warning: No segments found for analysis.")
            return x, y_smooth, dydx, d2ydx2, None, None, None, 0, None

        # --- Filter segments based on variance percentile ---
        # Ensure percentile threshold is within valid range
        variance_percentile_threshold = max(0, min(100, variance_percentile_threshold))

        # Calculate the variance threshold based on the specified percentile
        variance_threshold = np.percentile(segment_variances, variance_percentile_threshold)

        filtered_segments = [info for info in segment_info if info[0] <= variance_threshold]

        # --- Find the best segment among filtered segments ---
        # Prioritize length (descending) then variance (ascending)
        # This sorts to put the longest segments first, and for segments of the same length,
        # puts the ones with lower variance first.
        filtered_segments.sort(key=lambda item: (-item[3], item[0]))

        best_segment_x = None
        best_segment_d2ydx2 = None
        best_segment_variance = None
        best_segment_length = 0

        if filtered_segments:
            best_segment_info = filtered_segments[0]
            _, start_index, end_index, best_segment_length = best_segment_info
            best_segment_x = x[start_index: end_index]
            best_segment_d2ydx2 = d2ydx2[start_index: end_index]
            best_segment_variance = best_segment_info[0]
        else:
            print(f"Warning: No segments found with variance below the {variance_percentile_threshold}th percentile.")
            # Optionally, you could fall back to the segment with the absolute minimum variance here
            # or return None as done below. Returning None aligns with not finding a "suitable" region
            # based on the criteria.

        return (x, y_smooth, dydx, d2ydx2, best_segment_x, best_segment_d2ydx2,
                best_segment_variance, best_segment_length, np.array(segment_variances))

    def analyze_derivatives(self, x, y, poly=3):
        """
        Computes the first and second derivatives of y with respect to x,
        and finds the portion where the second derivative has the least variance,
        ignoring variances that are less than 20% of the maximum variance.

        Parameters:
            x (array-like): Input x values.
            y (array-like): Input y values.
            poly (int): Polynomial order for Savitzky-Golay filter (default: 3).

        Returns:
            tuple: (x_smooth, y_smooth, dydx, d2ydx2, best_segment_x, best_segment_y)
        """

        # DEV NOTE: needs to account for length as well, utilize the region thats most consistent for the longest


        # Ensure numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Remove duplicate x values and corresponding y values
        unique_x, unique_indices = np.unique(x, return_index=True)
        y = y[unique_indices]
        x = unique_x

        # Scale window size based on data length
        window = max(5, len(x) // 20)  # At least 5 points, 5% of data length
        if window % 2 == 0:
            window += 1  # Ensure window size is odd for Savitzky-Golay filter

        # Smooth y values using Savitzky-Golay filter
        y_smooth = savgol_filter(y, window, poly)

        # Compute first and second derivatives
        dydx = np.gradient(y_smooth, x)
        d2ydx2 = np.gradient(dydx, x)

        # Define segment size
        segment_size = max(len(x) // 10, 5)  # At least 5 points per segment
        min_var = float('inf')
        best_segment = (None, None)

        # Compute max variance for thresholding
        max_variance = np.var(d2ydx2)
        threshold = 0.2 * max_variance  # 20% of max variance

        # Slide a window across the second derivative to find lowest variance segment
        for i in range(len(d2ydx2) - segment_size + 1):
            segment = d2ydx2[i:i + segment_size]
            variance = np.var(segment)

            if variance < min_var and variance >= threshold:
                min_var = variance
                best_segment = (x[i:i + segment_size], d2ydx2[i:i + segment_size])

        return x, y_smooth, dydx, d2ydx2, best_segment[0], best_segment[1], variance, max_variance

    def updateSheet(self):
        try:
            self.strainCol = self.le_strainCol.text()
            self.stressCol = self.le_stressCol.text()
            df = pd.read_excel(str(self.filename), sheet_name=str(self.sheetname_le.text()), index_col=None, header=2)
            print(self.sheetname_le.text())
            print(df.head())  # shows preview of loaded data
            x = list(df.iloc[:, int(self.strainCol)])  # strain column default: 2
            y = list(df.iloc[:, int(self.stressCol)])  # stress column default: 1
            print(len(x), 'data points')
            print(len(y), 'data pts')
            self.x = x
            self.y = y
            self.plotting()

        except:
            print('error updating sheet')
            error = 'error updating sheet'
            self.messagebox(error)

    def modulus_incriment(self, x_limited, y_limited):

        inputs = [1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
        valid_r = []
        valid_s = []
        valid_int = []
        valid_x_list = []
        valid_y_list = []
        for input_value in inputs:
            x_list, y_list = increments(x_limited, y_limited, n=input_value)  # breaks raw data into incriments
            r, s, intercept = modulus(x_list, y_list)
            valid_r.append(r)
            valid_s.append(s)
            valid_int.append(intercept)
            valid_x_list.append(x_list)
            valid_y_list.append(y_list)

    def strain_limited_data(self, min, max, x, y):
        x_limited_idx = []
        for idx in range(0, len(x)):
            if int(min) < x[idx] < int(max):  # 10 and 25 need to be text box inputs from gui
                x_limited_idx.append(idx)
        y_limited = [x[1] for x in enumerate(y) if x[0] in x_limited_idx]
        x_limited = [x[1] for x in enumerate(x) if x[0] in x_limited_idx]
        return x_limited, y_limited

    def filter_xy(self, x_list, y_list, x_min, x_max):
        """
        Filters x values within the specified range [x_min, x_max] and updates y values accordingly.

        Parameters:
        x_list (list of float): List of x values.
        y_list (list of float): List of y values corresponding to x_list.
        x_min (float): Minimum x value to include.
        x_max (float): Maximum x value to include.

        Returns:
        tuple: Filtered lists (filtered_x, filtered_y).
        """
        filtered_x = []
        filtered_y = []

        for x, y in zip(x_list, y_list):
            if x_min <= x <= x_max:
                filtered_x.append(x)
                filtered_y.append(y)

        return filtered_x, filtered_y

    def max_R_slope(self, r, s, intercept):

        rmax = maxelements(r)
        print('rawr=', r)
        print('rmax=', rmax)
        print('s=', s)
        s_map = list(map(s.__getitem__, rmax))
        int_map = list(map(intercept.__getitem__, rmax))
        print('int_map and s_map', int_map, s_map)
        ymaxx = maxelements(self.y)
        y_end = len(self.y)
        raw_dotted_x = self.x
        del raw_dotted_x[ymaxx[0]:y_end]
        selected_s = s.count(s_map) # what is this
        selected_int = intercept.count(int_map)
        #selected_s = s_map[0]
        #selected_int = int_map[0]
        return raw_dotted_x, selected_s, selected_int, s_map, int_map

    def calculate_BU(self):

        self.update_range()
        print(self.stress_min, self.stress_max)
        x_limited, y_limited = self.stress_limited_data(self.stress_min, self.stress_max)

        # print(len(y_limited))
        # print(len(x_limited))
        # print('limted x\n', x_limited)
        # print('limited y\n', y_limited)

        # r,s, intercept, x_list, y_list = self.modulus_incriment(x_limited, y_limited)

        if self.FilamentBtn.isChecked():
            x_list, y_list = increments(x_limited, y_limited, n=1200)  # breaks raw data into incriments
            r, s, intercept = modulus(x_list, y_list, 3)
        else:
            x_list, y_list = increments(x_limited, y_limited, n=15)  # breaks raw data into incriments
            r, s, intercept = modulus(x_list, y_list, 3)


        print('modulus test; r= %s s= %s int= %s' % (r[0], s[0], intercept[0]))

        raw_dotted_x, selected_s, selected_int, s_map, int_map = self.max_R_slope(r, s, intercept)

        self.r_value_le.setText(str(round((max(r)), 5)))
        self.modulus_le.setText(str(round(s_map[0], 4))) #why does this get the slope value for the text

        s_min = (min(x_list[selected_s]))
        s_max = (max(x_list[selected_s]))
        region_x = x_list[selected_s]
        region_y = y_list[selected_s]  # plot xy on 2nd figure (possibly with regression line dotted)

        try:
            strain_range_output = "%s - %s" % (round(s_min, 4), (round(s_max, 4)))
            self.strainrange_le.setText(strain_range_output)
            self.Intercept_le.setText(str(round(max(intercept), 4)))
        except:
            print('strain not given due to s calculations')
        try:
            intercept = intercept[0]
            s = s[0]
            # linreg_dotted_y = int_map[0] + (s_map[0] * np.array(region_x)) bad code for examples
            linreg_dotted_y = intercept + (s * np.array(region_x))  # use selected one from og list
            print('raw dotted x', raw_dotted_x)
            raw_dotted_y = intercept + (s * np.array(raw_dotted_x))  # for raw data
            print(len(raw_dotted_x), len(raw_dotted_y))

            self.canvas2.axes.cla()  # Clear the canvas.
            self.canvas2.axes.plot(region_x, region_y, 'r')
            self.canvas2.axes.plot(region_x, linreg_dotted_y, 'k', linestyle='dashed')
            self.canvas.axes.plot(raw_dotted_x, raw_dotted_y, 'k',
                                  linestyle='dashed')  # find the index of self.y max, and only plot the x till ymax
            self.canvas2.axes.set_xlabel('Strain[mm/mm]')
            self.canvas2.axes.set_ylabel('Stress[MPa]')
            self.canvas2.draw()
            self.canvas.draw()  # add the dashed line
        except:
            print('linreg_dotted_y error')

    def line_similarity(self, line1_x,line1_y, line2_x, line2_y, std_dev_range):

        """
        Compare each point of two lines and calculate the percentage difference based on a range of standard deviation.

        Parameters:
            line1 (array-like): Array of (x, y) coordinates representing the first line.
            line2 (array-like): Array of (x, y) coordinates representing the second line.
            std_dev_range (int): Number of standard deviations to consider when calculating the percentage difference.

        Returns:
            (bool): True if the lines are within the specified range of standard deviation, False otherwise.
        """


        # Combine x and y coordinates into a list of (x, y) tuples for each line
        line1_list = list(zip(line1_x, line1_y))
        line2_list = list(zip(line2_x, line2_y))

        # Convert lists to NumPy arrays
        line1 = np.array(line1_list)
        line2 = np.array(line2_list)

        # Ensure that the shorter line is resized to match the length of the longer line
        max_length = max(len(line1), len(line2))
        line1_resized = np.resize(line1, (max_length, 2))
        line2_resized = np.resize(line2, (max_length, 2))


        # Calculate differences between corresponding points
        differences = np.abs(line1_resized - line2_resized)

        # Calculate mean and standard deviation for each coordinate
        mean_diff = np.mean(differences, axis=0)
        std_dev = np.std(differences, axis=0)

        # Calculate percentage difference based on the specified range of standard deviation
        max_diff = mean_diff + std_dev_range * std_dev
        min_diff = mean_diff - std_dev_range * std_dev

        # Check if all points fall within the range of standard deviation
        within_range = np.all((differences <= max_diff) & (differences >= min_diff), axis=1)

        # Calculate the percentage of points within the range
        percentage_within_range = np.sum(within_range) / len(within_range) * 100

        return percentage_within_range

    def euclidean_distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            a (tuple): Coordinates of the first point (x1, y1).
            b (tuple): Coordinates of the second point (x2, y2).

        Returns:
            (float): Euclidean distance between the two points.
        """
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def calculate(self):
        try:
            self.denoise()
        except:
            print('RIP')
        try:
            self.calc()
        except:
            print('An Error occured, change parameters')
            error = 'iAn Error occured, change parameters'
            print_exc()
            self.messagebox(error)


    def calc(self):

        self.update_range()
        #print(self.stress_min, self.stress_max)
        selected_x_start = findIndex(self.x, self.selected_x[0])
        selected_x_end = findIndex(self.x, max(self.selected_x))

        #x_limited, y_limited = self.strain_limited_data(self.selected_x[0], max(self.selected_x), self.x, self.y)
        x_limited, y_limited = self.filter_xy(self.x, self.y, self.selected_x[0], max(self.selected_x))


        #del self.selected_y[0: selected_x_start]
        #del self.selected_y[selected_x_end: len(self.selected_y)]


        #r_list, modulus_list, window_size_list = modulus_loop(x_limited, y_limited)
        #sorted_outputs = sort_and_assign_coefficients_with_values(modulus_list, window_size_list, r_list)
        #score = sorted_outputs[3]
        #optimal_window_size = sorted_outputs[1]
        optimal_window_size = int((0.30 * (len(x_limited))))
        if optimal_window_size < 2:
            optimal_window_size = 2
        r, s, intercept, start_index, end_index = modulus(x_limited, y_limited, optimal_window_size)
        start_index = selected_x_start
        end_index = selected_x_end



        print('modulus test; r= %s s= %s int= %s' % (r, s, intercept))
        #print('window size = %s #of iterations = %s' % (optimal_window_size, len(window_size_list)))

        #raw_dotted_x, selected_s, selected_int, s_map, int_map = self.max_R_slope(r, s, intercept)

        self.r_value_le.setText(str(round(r, 5)))
        self.modulus_le.setText(str(int(s))) #why does this get the slope value for the text
        #self.le_iterations.setText(str(len(window_size_list)))
        #self.le_window_size.setText(str(optimal_window_size))
        s_min = self.x[start_index]
        s_max = self.x[end_index]

        # Ultimate Strength Calculation and Display
        ultimate_str_index=(maxelements(self.y))
        ut = self.y[ultimate_str_index[0]]
        self.le_Ut.setText(str(int(ut)))

        ''' the ult str index to find the coordinating strain'''
        #yeild_strength = self.x[ultimate_str_index[0]] * 0.002 #start value
        #print('yeild_strength',yeild_strength)
        #yeild_strength_x_start_index = findIndex(self.x, yeild_strength)
        #yeild_strength_x = [x + yeild_strength for x in self.x]
        #yeild_strength_x = self.x

        #print('yeild_strength_x_start_index', yeild_strength_x_start_index)
        #del yeild_strength_x[0:yeild_strength_x_start_index]
        #print('yeild_strength_x', yeild_strength_x)
        #del yeild_strength_x[ultimate_str_index[0]: len(yeild_strength_x)]

        #yeild_strength_y = yeild_strength_x * s(s * np.array(region_x))
        #yeild_strength_y = s * np.array(yeild_strength_x)
        #yeild_strength_y = [x - yeild_strength_y[0] for x in yeild_strength_y]
        try:
            # Elongation at Break Calculation and Display
            elongation = self.ext[ultimate_str_index[0]]
            self.le_elongation.setText(str(elongation))
        except:
            print_exc()
            pass

        region_x = x_limited
        region_y = y_limited
        print('del region_x[%s:%s, %s:%s]'% (0, start_index, end_index, len(region_x)))
        #del region_x[0:start_index]
        #del region_x[end_index:len(region_x)]
        #del region_y[0:start_index]
        #del region_y[end_index:len(region_y)]

        try:
            strain_range_output = "%s - %s" % (round(s_min, 4), (round(s_max, 4)))
            self.strainrange_le.setText(strain_range_output)
            self.Intercept_le.setText(str(int(intercept)))
        except:
            print('strain/intercept not given due to s calculations')
            error = 'strain/intercept not given due to s calculations'
            self.messagebox(error)
            print_exc()
            pass

        try:
            ymaxx = maxelements(self.y)

            y_end = len(self.y)
            raw_dotted_x = self.x
            del raw_dotted_x[ymaxx[0]:y_end]
            raw_dotted_x = np.array(raw_dotted_x)
            raw_dotted_y=intercept + (s * raw_dotted_x)  # for raw data
            self.canvas.axes.plot(raw_dotted_x, raw_dotted_y, 'k',
                              linestyle='dashed')  # find the index of self.y max, and only plot the x till ymax
            self.canvas.draw()  # add the dashed line

        except:
            print_exc()
            print('2nd attempt dotted line failed')
            pass

        try:
            ymaxx = maxelements(self.y)

            y_end = len(self.y)
            raw_dotted_x = self.x
            del raw_dotted_x[ymaxx[0]:y_end]
            raw_dotted_x = np.array(raw_dotted_x)
            # linreg_dotted_y = int_map[0] + (s_map[0] * np.array(region_x)) bad code for examples
            linreg_dotted_y = intercept + (s * raw_dotted_x)  # use selected one from og list
            #print('raw dotted x', raw_dotted_x)
            raw_dotted_y = intercept + (s * raw_dotted_x)  # for raw data
            print(len(raw_dotted_x), len(raw_dotted_y))
            # %Diff Calculation
            #x_limited_diff, y_limited_diff = self.stress_limited_data(self.stress_min, self.stress_max, raw_dotted_x, raw_dotted_y)
            x_limited_diff, y_limited_diff = self.filter_xy(raw_dotted_x, raw_dotted_y, self.selected_x[0], max(self.selected_x))

            percent_diff = self.line_similarity(x_limited, y_limited, x_limited_diff, y_limited_diff, std_dev_range=1)
            self.le_percentDiff.setText(str(int(percent_diff)))
            self.canvas2.axes.cla()  # Clear the canvas.
            self.canvas2.axes.plot(region_x, region_y, 'r')
            self.canvas2.axes.plot(region_x, linreg_dotted_y, 'k', linestyle='dashed')
            self.canvas.axes.plot(raw_dotted_x, raw_dotted_y, 'k',
                                  linestyle='dashed')  # find the index of self.y max, and only plot the x till ymax
            #self.canvas.axes.plot(yeild_strength_x, yeild_strength_y, 'c', linestyle='dotted')
            self.canvas2.axes.set_xlabel('Strain[mm/mm]')
            self.canvas2.axes.set_ylabel('Stress[MPa]')
            self.canvas2.draw()
            self.canvas.draw()  # add the dashed line


        except:
            print('linreg_dotted_y error')
            error = 'linreg_dotted_y error'
            self.messagebox(error)
            print_exc()
            pass




        #Try to rework all graphing logic, its omega fucked 3/6/25





    def denoise(self):
        # incriment og x and y
        # using similar trick for limited x and y, use limited slope, and track all the blocks that are negative
        # remove blocks from both x and y
        # denoise_x, denoise_y = incriments(self.x, self.y)

        def remove_preload(stress, strain, threshold):
            """
            Eliminates prelude and failure data from a stress-strain curve based on a threshold.

           Parameters:
               stress (array-like): Array of stress data.
               strain (array-like): Array of strain data.
               threshold (float): Threshold to determine the start and end of valid data.

           Returns:
               (tuple): Tuple containing cleaned stress and strain data.
           """
            # Find the peak stress and corresponding strain
            peak_stress_index = np.argmax(stress)
            peak_stress = stress[peak_stress_index]
            peak_strain = strain[peak_stress_index]

            # Determine the threshold levels for strain
            start_threshold = peak_strain * threshold
            end_threshold = peak_strain * (1 - threshold)

            # Find the index where strain exceeds the start threshold
            start_index = np.argmax(strain >= start_threshold)

            # Find the index where strain exceeds the end threshold
            end_index = len(strain) - np.argmax(strain[::-1] >= end_threshold) - 1
            return stress[start_index:end_index], strain[start_index:end_index]



        def delete_failure_portion(stress, strain, failure_threshold):
            """
            Remove the failure portion of a stress-strain curve based on a failure threshold.

            Parameters:
                stress (array-like): Array of stress data.
                strain (array-like): Array of strain data.
                failure_threshold (float): Threshold to determine the failure portion of the curve.

            Returns:
                (tuple): Tuple containing stress and strain data with the failure portion removed.
            """
            max_stress = np.max(stress)
            failure_index = np.argmax(stress >= max_stress * failure_threshold)
            return stress[:failure_index], strain[:failure_index]

        threshold = 0.15
        stress = np.array(self.cleany)
        strain = np.array(self.cleanx)

        # preload
        cleaned_stress, cleaned_strain = remove_preload(stress, strain, threshold)

        # failure
        cleaned_stress, cleaned_strain = delete_failure_portion(cleaned_stress, cleaned_strain, failure_threshold=0.95)

        #if cleaned_stress[0] > 2:
        #    y_offset = cleaned_stress[0] * 2

        self.export_strain = cleaned_strain
        self.export_stress = cleaned_stress
        self.canvas3.axes.cla()  # Clear the canvas.
        self.canvas3.axes.plot(cleaned_strain, cleaned_stress, 'r')
        self.canvas3.axes.set_xlabel('Strain[mm/mm]')
        self.canvas3.axes.set_ylabel('Stress[MPa]')
        self.canvas3.draw()
    def old_denoise(self):
        # incriment og x and y
        # using similar trick for limited x and y, use limited slope, and track all the blocks that are negative
        # remove blocks from both x and y
        # denoise_x, denoise_y = incriments(self.x, self.y)
        y_denoise = self.y
        x_denoise = self.x
        ymax = (maxelements(self.y))
        yfinal = len(self.y)

        del y_denoise[ymax[0]:yfinal]  # deletes all points after max stress
        del x_denoise[ymax[0]:yfinal]

        # self.calculate()               # we need the slope to extrapolate later

        x_denoise_list, y_denoise_list = increments(x_denoise, y_denoise, 20)
        r, s, int_denoise = modulus(x_denoise_list, y_denoise_list)
        s_zero = []
        # print('slope list', s)
        for s_idx in range(0, 100):
            if s[s_idx] < 500:
                s_zero.append(s_idx)
        print('first slope is', s[max(s_zero)])
        print('first strain', min(x_denoise_list[max(s_zero)]))

        del x_denoise_list[0:max(s_zero)]
        del y_denoise_list[0:max(s_zero)]
        xdl = flatten(x_denoise_list)
        ydl = flatten(y_denoise_list)

        int_slope, r_int, intercept_int = linreg(xdl[0:200], ydl[0:200])
        # calc x-int
        x_int = (-1 * intercept_int) / int_slope
        print('x-int is', round(x_int, 5), ',', 0)
        # this adds (x-int, 0) into the xy list at the start
        xdl.insert(0, x_int)
        ydl.insert(0, 0)
        xdl = xdl - xdl[0]
        print('x-int is', round(x_int, 5), ',', 0)
        print('xdl\n', xdl)
        '''
        x_int = np.linspace(0, min(x_denoise_list[max(s_zero)]), 10)
        ss_int = InterpolatedUnivariateSpline(xdl[0:2000], ydl[0:2000], k=1) # probably needs a new extrap method
        y_int = ss_int(x_int)
        for i in reversed(range(len(x_int))):
            xdl.insert(0, x_int[i])

        for i in reversed(range(len(y_int))):
            ydl.insert(0, y_int[i])
        print(s_zero)
        print('xdl', xdl, '\n ydl', ydl)
        '''
        self.export_strain = xdl
        self.export_stress = ydl
        self.canvas3.axes.cla()  # Clear the canvas.
        self.canvas3.axes.plot(xdl, ydl, 'r')
        self.canvas3.axes.set_xlabel('Strain[mm/mm]')
        self.canvas3.axes.set_ylabel('Stress[MPa]')
        self.canvas3.draw()

    def test(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]
        slope, R = linreg(x, y)
        self.plotting(x, y)
        print('slope=', slope, 'R=', R)

    def test2(self):
        print('yepp2')


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = Mainwindow()
    # mw.setWindowTitle('Otto Cycle Calculator')
    sys.exit(app.exec())
