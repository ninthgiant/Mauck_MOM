
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backend_bases as backendgi
from setuptools import find_namespace_packages
import numpy as np
import statistics
from scipy import stats 
import os

################
# my_Do_Calibrations: to get the calibration values for the MOM on this burrow-night
#    RAM 7/25/22
#    parameters: my_dataframe -> data to work with
#    returns tuple with information about the result of the calculations
#######
def my_Do_Calibrations(my_dataframe):
        
    data = my_dataframe
    
    global calibrations

    global cal_gradient
    global cal_intercept
    global baseline_cal_mean
    global cal_r_squared

    global cal1_value
    global cal2_value
    global cal3_value

    ## changd to float, does that cure iut?
    cal1_value = float(my_entries2[0].get())
    cal2_value = float(my_entries2[1].get())
    cal3_value = float(my_entries2[2].get())

    good_to_go = True

    # Add baselines
    baseline_cal_mean, baseline_cal_markers, baseline_cal_Good, axesLimits = getTracePointPair(data, "Baseline")
    markers = baseline_cal_markers

    # Add calibrations as 3 separate pairs of points
    cal1_mean, cal1_markers, cal1_Good, axesLimits = getTracePointPair(data, "Cal1[{}]".format(cal1_value), markers, axesLimits)
    markers = pd.concat([markers, cal1_markers])

    cal2_mean, cal2_markers, cal2_Good, axesLimits = getTracePointPair(data, "Cal2[{}]".format(cal2_value), markers, axesLimits)
    markers = pd.concat([markers, cal2_markers])

    cal3_mean, cal3_markers, cal3_Good, axesLimits = getTracePointPair(data, "Cal3[{}]".format(cal3_value), markers, axesLimits)
    markers = pd.concat([markers, cal3_markers])

    # Clean up the marked calibration points data
    calibrations = pd.DataFrame({"Category":["Cal1", "Cal2", "Cal3"],
                                    "Value_True":[cal1_value, cal2_value, cal3_value],
                                    "Value_Measured":[cal1_mean, cal2_mean, cal3_mean]})
    calibrations["Value_Difference"] = abs(calibrations["Value_Measured"] - baseline_cal_mean)

    # Get the linear regression information across the three calibration points
    cal_gradient, cal_intercept, cal_r_value, cal_p_value, cal_std_err = stats.linregress(calibrations["Value_Difference"], calibrations["Value_True"])
    cal_r_squared = cal_r_value**2

    fig, ax = plt.subplots()

    ax.plot(calibrations["Value_Difference"], calibrations["Value_True"], marker="o", color="black", linestyle="None")
    ax.plot(calibrations["Value_Difference"], calibrations["Value_Difference"]*cal_gradient+cal_intercept, color="gray", linestyle="dashed")
    plt.xlabel("Measured value (strain difference from baseline)")
    plt.ylabel("True value (g)")
    plt.title("Calibration regression\n(R^2={r}, Inter={i}, Slope={s})".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5)))
    plt.show()

    ### show user in real time
    my_cal_result = "\tR^2={r}\n\tIntcpt={i}, Slope={s}".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5))
    t2.insert("1.0", my_cal_result + "\n") # add to Text widget
    t2.insert("1.0", "File: " + user_BURROW + " - Calibration regression:" + "\n") # add to Text widget
    
    # Check all the calibrations were marked successfully
    # if (not baseline_cal_Good or not cal1_Good or not cal2_Good or not cal3_Good or (cal_r_squared<0.9)):
    # NEED TO ADD: alert box saying it was a bad calibration
    if (abs(cal_r_squared) < 0.9):
        print("bad r2")
        good_to_go = False

    return good_to_go


################
#  calc_Penguin_W2: Algorithm from Afanasyev et al. 2015
#    RAM 6/25/2024 with help from ChatGPT
#       receives a dataframe with strain and unix time, the markers dataframe defining the subset of dataframe to use
#           also receives 
#       Lets user indicate baseline as normal
#           then, start and end points, as per normal
#           NEW: does the Penguin calculation between those two points
#       Penguin Calculation
#          a = Get Slope -> between the points (start and end)
#          g = 9.8
#          W1 = Mean Value between the two points - all normal to this point
#          W2 = W1 ( 1 + a/g)
#    
#    returns tuple with information about the result of the calculations
#    code below is hte complete do_Bird, but need to figure out where this goes and how it works with Do_Bird or mom_cut_Button
#######
def calc_Penguin_W2(my_data, markers, W1, SPS=60):
    # Define gravitational constant
    g = 9.81
    
    # Extract start and end indices from 'markers' dataframe
    start_index = markers.loc[0, "Index"]
    end_index = markers.loc[1, "Index"]
    
    # Extract subset of 'Measure' series from 'my_data' dataframe
    measure_series = my_data.loc[start_index:end_index, "Measure"]
    
    # Calculate original slope of the extracted 'Measure' series
    x_values = np.arange(len(measure_series))
    slope, _ = np.polyfit(x_values, measure_series, 1)
    
    # Compensate for the sampling interval
    sampling_interval = 1 / SPS
    adj_slope = slope / sampling_interval
    
    # Calculate W2 using the adjusted slope
    W2 = W1 + W1 * (1 + adj_slope / g)
    
    return W2