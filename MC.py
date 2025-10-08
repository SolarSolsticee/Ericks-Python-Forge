
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename

import numpy as np
import pandas as pd
#from PyQt5.QtWidgets.QWidget import window
#from jupyterlab.semver import valid_range
from scipy import stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def linreg(x,y):
    regr = stats.linregress(x,y)
    return regr[0], regr[2], regr[1] # slope, R Value, int

def incriments_splits(x,y, n):
    # n number of splits

    x_list = np.array_split(x,n)
    y_list = np.array_split(y,n)
    return x_list, y_list

def increments(x, y, n):
    # n number of elements per list
    # n=1200 #approx 0.015 strain inc
    x_list = [x[i * n:(i+1) * n] for i in range((len(x)+n-1) // n)]
    y_list = [y[i * n:(i + 1) * n] for i in range((len(y) + n - 1) // n)]
    return x_list, y_list

def flatten(l):
    return [item for sublist in l for item in sublist]


def region_with_max_length_min_variance(x, y, variance_threshold):
    """
    Function to find the region with the maximum length and minimal variance in y data.

    Parameters:
    x (list or array): x data
    y (list or array): y data
    variance_threshold (float): the maximum allowed variance for the region

    Returns:
    tuple: start and end indices of the region with the maximum length and minimal variance
    """
    max_length = 0
    min_variance = float('inf')
    best_region = None

    for region_size in range(1, len(y) + 1):
        for i in range(len(y) - region_size + 1):
            region = y[i:i + region_size]
            variation = np.var(region)

            if variation <= variance_threshold and region_size > max_length:
                max_length = region_size
                best_region = (i, i + region_size - 1)

    return best_region


def trim_first_last_10_percent(x, y):
    """
    Removes the first and last 10% of elements from both x and y lists.

    :param x: List of x values
    :param y: List of y values
    :return: Trimmed x and y lists
    """
    if not x or not y or len(x) != len(y):
        raise ValueError("Both x and y must be non-empty lists of the same length.")

    n = len(x)
    start = int(n * 0.10)  # Index to start keeping values
    end = int(n * 0.90)    # Index to stop keeping values

    return x[start:end], y[start:end]


def sort_and_assign_coefficients_with_values(modulus, window_size, r):
    # Sort the lists in descending order
    sorted_modulus = sorted(modulus, reverse=True)
    sorted_window_size = sorted(window_size, reverse=True)
    sorted_r = sorted(r, reverse=True)

    # Assign coefficients to each rank
    coefficients_modulus = [i + 1 for i in range(len(sorted_modulus))]
    coefficients_window_size = [i + 1 for i in range(len(sorted_window_size))]
    coefficients_r = [i + 1 for i in range(len(sorted_r))]

    # modify weight
    x1 = 2 # window size weight
    x2 = 0.2 # modulus weight
    x3 = 1.5 # r^2 weight
    coefficients_window_size = [x * x1 for x in coefficients_window_size]
    coefficients_modulus = [x * x2 for x in coefficients_modulus]
    coefficients_r = [x * x3 for x in coefficients_r]

    # Create a dictionary to store the original index and its coefficient
    index_coefficients = {i: 0 for i in range(len(modulus))}

    # Assign coefficients based on the sorted lists
    for i, value in enumerate(sorted_modulus):
        index = modulus.index(value)
        index_coefficients[index] += coefficients_modulus[i]

    for i, value in enumerate(sorted_window_size):
        index = window_size.index(value)
        index_coefficients[index] += coefficients_window_size[i]

    for i, value in enumerate(sorted_r):
        index = r.index(value)
        index_coefficients[index] += coefficients_r[i]

    # Generate the new list of coefficients with their associated values
    new_coefficients_with_values = [(modulus[i], window_size[i], r[i], index_coefficients[i]) for i in
                                    range(len(modulus))]

    # Find the tuple with the lowest coefficient
    min_coefficient_tuple = min(new_coefficients_with_values, key=lambda x: x[3])

    return min_coefficient_tuple


def remove_beyond_max_y(x, y):
    max_y = maxelements(y)
    del x[max_y[0]: len(y)]
    del y[max_y[0]: len(y)]

    return x, y


def make_differentiable(X, Y, method='perturb', epsilon=1e-6):
    """
    Process X and Y data to ensure differentiability by handling duplicate X values.

    Parameters:
        X (list or np.array): Input x values.
        Y (list or np.array): Corresponding y values.
        method (str): 'perturb' to slightly adjust duplicate x values,
                      'average' to average y values of duplicates.
        epsilon (float): Small value to perturb x values if method='perturb'.

    Returns:
        tuple: Processed X and Y arrays ensuring differentiability.
    """
    #X = np.array(X, dtype=float)
    #Y = np.array(Y, dtype=float)

    unique_x, counts = np.unique(X, return_counts=True)
    if np.any(counts > 1):
        if method == 'perturb':
            for i in range(len(X)):
                while np.sum(X == X[i]) > 1:
                    X[i] += np.random.uniform(-epsilon, epsilon)
        elif method == 'average':
            new_X, new_Y = [], []
            for ux in unique_x:
                indices = np.where(X == ux)[0]
                new_X.append(ux)
                new_Y.append(np.mean(Y[indices]))
            return np.array(new_X), np.array(new_Y)

    return X, Y


def delete_after_max(lst):
    if not lst:  # Check if the list is empty
        return lst

    max_value = max(lst)
    max_index = lst.index(max_value)

    # Return the list up to and including the max value
    return lst[:max_index + 1]

def window_size_calc(y_list):
    window_range = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    valid_windows = []
    valid_window_range = []
    refined_y_list = delete_after_max(y_list)
    for x in window_range:
        window_size = int((x * (len(refined_y_list))))
        if window_size >= 5:
            valid_windows.append(window_size)
            valid_window_range.append(x)
    return valid_windows[0], valid_window_range, refined_y_list

def window_size_check(initial_window_size):
    if initial_window_size < 2:
        initial_window_size = 2
    return initial_window_size

def increment_size_check(increment):
    if increment < 1:
        increment = 1
    return increment

def downscale_dataset(data, n_components):
    """
    Downscale a dataset using PCA.

    Parameters:
        data (pd.DataFrame): The input dataset.
        n_components (int): The number of principal components to keep.

    Returns:
        pd.DataFrame: The downscaled dataset.
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Create a DataFrame with the principal components
    columns = [f'PC{i+1}' for i in range(n_components)]
    downscaled_data = pd.DataFrame(data=principal_components, columns=columns)

    return downscaled_data



def modulus_loop(x_list, y_list): # increases window size and returns [modulus], [window size], [r^2] till 10% variation
    modulus_list = []
    window_size_list = []
    r_list = []
    #refined_y_list = delete_after_max(y_list)

    initial_window_size = int(len(y_list) * 0.05) # 5% of the ylist
    initial_window_size_checked = window_size_check(initial_window_size)

    increment = int(len(y_list) * 0.02) # incrimenting by 10% of the ylist
    increment = increment_size_check(increment)

    original_value = modulus(x_list, y_list, initial_window_size_checked)
    original_modulus = original_value[1]
    window_size_increment = initial_window_size_checked
    # Append initial values/results to lists
    r_list.append(original_value[0])
    modulus_list.append(original_value[1])
    window_size_list.append(initial_window_size_checked)


    while True:
        window_size_increment += increment
        new_value = modulus(x_list, y_list, window_size_increment)
        new_modulus = new_value[1]
        variation = abs(new_modulus - original_modulus) / original_modulus

        # Append initial values/results to lists
        r_list.append(new_value[0])
        modulus_list.append(new_value[1])
        window_size_list.append(window_size_increment)

        if variation > 0.2:  # Check if variation exceeds 20%
            break

        if window_size_increment > len(y_list)*0.5:
            break

    return r_list, modulus_list, window_size_list


def modulus(x_list, y_list, window_size):
    """
    Finds the straightest portion of the curve using linear regression within a sliding window.

    Parameters:
        x: array-like, x-coordinates of the data points.
        y: array-like, y-coordinates of the data points.
        window_size: int, size of the sliding window.

    Returns:
        start_index: int, index of the starting point of the straightest portion.
        end_index: int, index of the ending point of the straightest portion.
    """

    #window_size, valid_window_range, refined_y_list = window_size_calc(y_list)
    #window_size = int((0.30 * (len(x_list))))
    num_points = len(x_list)
    max_r_squared = -np.inf
    start_index, end_index = 0, 0
    modulus = None
    integer = None
    x_list = np.array(x_list)
    y_list = np.array(y_list)


    for i in range(num_points - window_size + 1):
        window_x = x_list[i:i + window_size]
        window_y = y_list[i:i + window_size]
        slope, r_value, intercept = linreg(window_x, window_y)
        r_squared = r_value ** 2

        if r_squared > max_r_squared:
            max_r_squared = r_squared
            modulus = slope
            integer = intercept
            start_index = i
            end_index = i + window_size - 1

    #print('window size input:', window_size, 'valid window range:', valid_window_range)
    #print('length of list:', len(refined_y_list))
    #print('DEBUG PERCENT:',100 * window_size/(len(refined_y_list)), '%')

    return max_r_squared, modulus, integer, start_index, end_index

def create_list(start, end, increment):
    return list(range(start, end + 1, increment))

def modulus_1_2(x_list, y_list):
    r_list = []
    slope_list =[]
    int_list = []
    for i in range(len(x_list)):
        slope, R, intercept = linreg(x_list[i], y_list[i])
        r_list.append(R)
        slope_list.append(slope)
        int_list.append(intercept)
    #print(r_list)
    #print(slope_list)
    #print('x list',x_list,'ylist', y_list)
        if slope_list is not None:
            return r_list, slope_list, int_list
    return None, None, None


def maxelements(seq):
    ''' Return list of position(s) of largest element '''
    max_indices = []
    if seq:
        max_val = seq[0]
        for i,val in ((i,val) for i,val in enumerate(seq) if val >= max_val):
            if val == max_val:
                max_indices.append(i)
            else:
                max_val = val
                max_indices = [i]
    return max_indices

def findIndex(my_list, target_value):
    my_array = np.array(my_list)
    closest_index = np.argmin(np.abs(my_array - target_value))
    print(f"Index of value closest to {target_value}: {closest_index}")
    return closest_index

def main():
    Tk().withdraw()
    filename = askopenfilename()
    print(filename)
    df=pd.read_excel(str(filename),sheet_name='1.24-4 (1)', index_col=None, header = 2)
    print(df.head()) #shows preview of loaded data
    x=list(df.iloc[:,2])
    y=list(df.iloc[:,1])
    print(len(x), 'data points')
    print(len(y), 'data pts')
    #test data
    #x = [1, 2, 3, 4, 12, 57, 93, 50]
    #y = [1, 3, 3, 4, 50, 30, 56, 95]
    x_list, y_list = increments(x, y)
    r, s, i = modulus(x_list,y_list)
    rmax=maxelements(r)
    print(r)
    print(max(r))
    print(rmax)
    print(s)
    s_map = list(map(s.__getitem__, rmax))
    selected_s = s.count(s_map)
    print('Calculated Modulus is', s_map, 'Mpa')
    print('via Linear Regression R=',max(r))
    print(selected_s)
    s_min = (min(x_list[selected_s]))
    s_max = (max(x_list[selected_s]))
    print(str(s_min),'-',str(s_max))

if __name__ == "__main__":
    main()

