#######################################
#######################################
#    MOM_Calculations.py
#       Calibrating and calculating real weights from load cell trace data
#       R.A.M and L.U.T.
#       2024-08-27 cleanup of RAM_v10
#######################################
#######################################

#######################################
#######################################
# Imports and libraries
#######################################
#######################################

import numpy as np
from scipy import stats 
from tkinter import messagebox

#######################################
#######################################
# Calibration object
#######################################
#######################################

#######
# Class Calibration
#   Stores regression coefficients to map between true (g) vs. measured (strain) values.
#   After obtaining strain values that correspond to three known-true (g) values,
#   use the regression() function to fully initialize the object.
#######
class Calibration:
    def __init__(self):
        # Default calibration values
        self.cal1_true = 15.97
        self.cal2_true = 32.59
        self.cal3_true = 50.22

        # Pre-initialized measurement values
        self.cal1_measured = -1.0
        self.cal2_measured = -1.0
        self.cal3_measured = -1.0

        self.baseline = -1.0
        self.regression_gradient = -1.0
        self.regression_intercept = -1.0
        self.regression_rsquared = -1.0

        # You can use initialized to check if the user has
        #   performed an appropriate calibration value yet
        self.initialized = False

    # Get the list of real values in grams
    def get_true(self):
        return [self.cal1_true, self.cal2_true, self.cal3_true]
    
    # Get the list of measured values in strain units
    def get_measured(self):
        return [self.cal1_measured, self.cal2_measured, self.cal3_measured]

    # Get the list of difference between measurement and baseline
    def get_difference(self):
        return [x - self.baseline for x in self.get_measured()]
    
    # Set the true values, with error catching if the GUI-entered values are invalid
    def set_true(self, cal1_true, cal2_true, cal3_true):
        try:
            self.cal1_true = float(cal1_true)
            self.cal2_true = float(cal2_true)
            self.cal3_true = float(cal3_true)
            return True
        except ValueError:
            messagebox.showerror("Error.", "User entered invalid calibration value(s).")
            return False

    # Conduct a calibration regression between stored true values and the input measurements
    def regression(self, cal_baseline, cal1_measured, cal2_measured, cal3_measured):
        # Set the measured values
        self.baseline = cal_baseline
        self.cal1_measured = cal1_measured
        self.cal2_measured = cal2_measured
        self.cal3_measured = cal3_measured

        # Get the true values
        true_values = self.get_true()

        # The regression is performed on the difference between the measured and the true
        # NOTE this needs to be consistent between calibration and bird measurements
        #      to keep the intercept meaningful
        difference_values = self.get_difference()

        # Store linear regression information across the three calibration points
        self.regression_gradient, self.regression_intercept, cal_r_value, cal_p_value, cal_std_err = stats.linregress(difference_values, true_values)
        self.regression_rsquared = cal_r_value**2

        # The regression is now complete, and this calibration object is fully initialized
        self.initialized = True

#######################################
#######################################
# Measurement functions
#######################################
#######################################

#######
# Function fit_slope
#   Fits a linear regression slope across all trace values between two points
#   NOTE these slopes are uncalibrated; 
#        need to multiply by calibration regression for slope in y-axis grams  
# Parameters:
#   dat         - full loadcell dataframe (pandas Dataframe of form Measurement | Datetime)
#   start_index - start point for trace segment (int)
#   end_index   - end point for trace segment (int)
# Returns: 
#   Uncalibrated slope points in the trace segment (float)
#######
def fit_slope(dat, start_index, end_index):
    # Isolate the trace segment
    trace_segment = dat.loc[start_index:end_index, "Measure"]
    # Get discrete range of x values across the segment
    x_values = np.arange(0, len(trace_segment))
    # Fit regression for slope
    slope, _ = np.polyfit(x_values, trace_segment, 1)
    return slope
    
#######
# Function w_mean
#   Calculates calibrated bird weight as the mean trace value between two points 
# Parameters:
#   dat         - full loadcell dataframe (pandas Dataframe of form Measurement | Datetime)
#   calibration - initialized weight calibration object (MOM_Calculations.Calibration)
#   start_index - start point for bird activity in the trace (int)
#   end_index   - end point for bird activity in the trace (int)
#   baseline    - baseline value near bird activity for calibration (float)
# Returns: 
#   Calibrated weight in grams, unrounded (float)
#   Calibrated slope associated with the trace segment (float)
#######
def w_mean(dat, calibration, start_index, end_index, baseline):
    # Get the mean measurement from the trace segment
    measure_mean = dat.loc[start_index:end_index, "Measure"].mean()
    # Calibrate raw values to grams
    calibrated_weight = (measure_mean - baseline) * calibration.regression_gradient + calibration.regression_intercept
    # Get the slope of the segment and calibrate to y-axis grams
    calibrated_slope = fit_slope(dat, start_index, end_index) * calibration.regression_gradient
    return calibrated_weight, calibrated_slope

#######
# Function w_windowed_min_slope
#   Calculates calibrated bird weight as the mean trace value
#   for the sub-window that has the minimum slope, 
#   checking all sub-windows of appropriate sizes between two points
# NOTE formerly do_slope_0
# Parameters:
#   dat               - full loadcell dataframe (pandas Dataframe of form Measurement | Datetime)
#   calibration       - initialized weight calibration object (MOM_Calculations.Calibration)
#   start_index       - start point for bird activity in the trace (int)
#   end_index         - end point for bird activity in the trace (int)
#   baseline          - baseline value near bird activity for calibration (float)
#   min_window_length - the minimum window length to check slope (int)
# Returns: 
#   Calibrated weight in grams, unrounded (float)
#   Calibrated slope of the chosen window with the minimum slope (float)
#   Window start index (int)
#   Window end index (int)
#######
def w_windowed_min_slope(dat, calibration, start_index, end_index, baseline,
                         min_window_length = 25, max_window_proportion = 0.5):
    
    # Baseline comparison values
    # Starting "best" (min) slope is the slope across the whole trace segment
    # Starting "best" window range is the full trace segment
    min_slope = fit_slope(dat, start_index, end_index)
    best_start_index = start_index
    best_end_index = end_index

    # Define the maximum window length,
    # as user-defined proportion of full trace segment size (default 0.5)
    # or minimum window length + 2, whichever longer
    max_window_length = int((end_index - start_index + 1) * max_window_proportion)    
    if max_window_length < min_window_length:
        max_window_length = min_window_length + 2 

    # Begin searching through windows, where the start index marks the LEFT edge of the window
    # For every start point
    for window_start in range(start_index, end_index - min_window_length):
        # For every allowed window size beginning at that start point
        for window_size in range(min_window_length, max_window_length):
            window_end = window_start + window_size
            
            # Fit the slope to that window of the data
            curr_slope = fit_slope(dat, window_start, window_end)

            # If the current slope is smaller in magnitude than the best slope, 
            # then this is your new preferred window
            if abs(curr_slope) <= abs(min_slope):
                min_slope = curr_slope
                best_start_index = window_start
                best_end_index = window_end

    # Calculate mean value of the preferred window (min slope)
    measure_mean = dat.loc[best_start_index:best_end_index, "Measure"].mean()

    # Calibrated raw values to grams
    calibrated_weight = (measure_mean - baseline) * calibration.regression_gradient + calibration.regression_intercept
    calibrated_slope = min_slope * calibration.regression_gradient

    # Because this function finds an optimal window, 
    # we return not only the weight calculated from that window,
    # but also the start and end indices of the window itself 
    return calibrated_weight, calibrated_slope, best_start_index, best_end_index

#######
# Function w_median
#   Calculates calibrated bird weight as the median trace value between two points 
# Parameters:
#   dat         - full loadcell dataframe (pandas Dataframe of form Measurement | Datetime)
#   calibration - initialized weight calibration object (MOM_Calculations.Calibration)
#   start_index - start point for bird activity in the trace (int)
#   end_index   - end point for bird activity in the trace (int)
#   baseline    - baseline value near bird activity for calibration (float)
# Returns: 
#   Calibrated weight in grams, unrounded (float)
#   Calibrated slope associated with the trace segment (float)
#######
def w_median(dat, calibration, start_index, end_index, baseline):
    # Get the median measurement from the trace segment
    measure_median = dat.loc[start_index:end_index, "Measure"].median()
    # Calibrate raw values to grams    
    calibrated_weight = (measure_median - baseline) * calibration.regression_gradient + calibration.regression_intercept
    # Get the slope of the segment and calibrate to y-axis grams
    calibrated_slope = fit_slope(dat, start_index, end_index) * calibration.regression_gradient
    return calibrated_weight, calibrated_slope