
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



#########################
#   do_Slope_0: A function to find the slope that is zero from all subsets of the trace, then return start and stop of that subset
#       RAM, July 11, 2024
#       ARGUMENTS: 
#       1) the array to start the process - see do_Bird to see how to get a measure_series to send to this function
#       2) starting point for the series that has been marked wihtin that series
#       3) end point to mark wihtin the series
#       4) smalles window we will consider - user input, but 20-30 points based on data in handd
#       5) How big can the window be relative to the overall size of the marked subset?
#
#       RETURNS:
#       1) series on which to do calculations based on calibrations, etc., W1 or W2
########
def do_Slope_0(measure_series, start_pt, stop_pt, min_len, max_pct = 0.5):

    print("################# DEBUG: read values from screen inside generate_final_series ##############")
    print(f"Threshold: {threshold}")


    max_len = int((stop_pt - start_pt) * max_pct)  # what is the max size of the window to calculate slope
    total_windows = max_len - min_len           # number of windows to cycle thru each time
    trace_len = stop_pt - start_pt - 1              # how many points to iterate over

    ## how big a dataframe do we need to hold trace_len (start to stop pts -1) * total_windows (how many different lengths)
    # Generate random data for each column
    wt_data = {
        'slope': np.random.uniform(-10, 10, size=(trace_len * total_windows)),
        'intercept': np.random.uniform(-10, 10, size=(trace_len * total_windows)),
        'wt_calc': np.random.uniform(0, 100, size=(trace_len * total_windows)),
        'center_pt': np.random.uniform(-50, 50, size=(trace_len * total_windows)),
        'total_len': np.random.uniform(1, 1000, size=(trace_len * total_windows))
    }

    # Create DataFrame
    df = pd.DataFrame(wt_data)

    # Reshape the DataFrame to the desired size
    df = df.values.reshape((trace_len, total_windows, len(wt_data)))

    for i in range(start_pt, stop_pt)
        for w in range( min_len, max_len ):

            # the current point is the mid-point of the vector on which to calculate slope
            win_start = i - int(w/2)
            win_end = i + int(w/2)

            # see do_Bird to see how to get a measure_series to send to this function
            # need the subset of measure_series defined by win_star and win_end
            curr_trace = measure_series                      

            ###########
            # calc stats of this vector - if the slope == 0, the the w_slopeintercept will be the mean to calculate wt
            ###
            w_slope, w_slopeintercept = np.polyfit(curr_trace, vector, 1)
            


    # Create the FWD and BkWD series (initially empty)
    FWD_series = np.zeros_like(input_array, dtype=int)
    BKWD_series = np.zeros_like(input_array, dtype=int)

    # Populate the FWD and BKWD series based on comparisons with the threshold
    for i in range(1, len(input_array) - 1):
        current_value = input_array[i]
        previous_value = input_array[i - 1]
        next_value = input_array[i + 1]

        # Check conditions using the threshold and populate the FWD series
        if abs(current_value - next_value) < threshold:
            FWD_series[i] = 1
        else:
            FWD_series[i] = 0

        # Check conditions using the threshold and populate the BKWD series
        if abs(current_value - previous_value) < threshold:
            BKWD_series[i] = 1
        else:
            BKWD_series[i] = 0

    # Create a third 'empty' array the size of the BKWD_series that sums BKWD and FWD
    sum_array = BKWD_series + FWD_series

    # sum array - a 1 value is a start or stop value; a 2 value is middle, a 0 is not a series
    # step through the array, when encounter a 1 set flg to a new series until a 2nd 1 is encountered, then flg is false

    # count array - steps through sum array and identifies start and stop points to count length of series
    # count in each direction, so need 2 arrays for the counts and one array for the sum of counts

    Count_FWD_array = np.zeros_like(input_array, dtype=int)
    Count_BKWD_array = np.zeros_like(input_array, dtype=int)
    Count_Sum_array = np.zeros_like(input_array, dtype=int)

    # Counter variable for the count array
    counter = 0
    # variable to track if in a distinct series
    inside = 0  # is 1 when we are inside a series, treat it like a boolean

    # Populate the Count_FWD_array using a counter
    for i in range(len(Count_FWD_array)):
        if sum_array[i] ==1 and inside == 0:  # we have the start with a 1 which is how it is supposed to work - ADDED THIS
            counter += 1
            Count_FWD_array[i] = counter
            inside = 1  # we are now in a series
        elif sum_array[i] == 2 and inside == 0:  # we have the start when doing a calibration or starting within a long series of consistant
            counter += 1
            Count_FWD_array[i] = counter
            inside = 1  # we are now in a series
        elif sum_array[i] == 1 and inside == 1:  # we have the end:
            counter += 1
            Count_FWD_array[i] = counter
            inside = 0
            counter = 0
        elif sum_array[i] == 2 and inside == 1:  # we have the middle:
            counter += 1
            Count_FWD_array[i] = counter
        else:  # we are not in a series
            counter = 0
            Count_FWD_array[i] = 0

    # Populate the Count_BKWD_array using a counter
    for i in range(len(Count_BKWD_array) - 1, -1, -1):
        if sum_array[i] == 1 and inside == 0:  # we have the start
            counter += 1
            Count_BKWD_array[i] = counter
            inside = 1  # we are now in a series
        elif sum_array[i] ==2 and inside == 0:  # we have the start
            counter += 1
            Count_BKWD_array[i] = counter
            inside = 1  # we are now in a series
        elif sum_array[i] == 1 and inside == 1:  # we have the end:
            counter += 1
            Count_BKWD_array[i] = counter
            inside = 0
            counter = 0
        elif sum_array[i] == 2 and inside == 1:  # we have the middle:
            counter += 1
            Count_BKWD_array[i] = counter
        else:  # we are not in a series
            counter = 0
            Count_BKWD_array[i] = 0

    # Populate the Count_Sum_array array by adding values from BkWD and FWD
    for i in range(len(Count_FWD_array)):
        Count_Sum_array[i] = Count_BKWD_array[i] + Count_FWD_array[i]

    # Create a fourth 'empty' array the size of the Count_Sum_array array
    final_array = np.zeros_like(input_array, dtype=int)

    count_criterion = myLen  # what qualifies as a series - might need to adjust +/- 1

    # Populate the fourth array based on conditions
    for i in range(len(final_array)):
        if Count_Sum_array[i] > count_criterion:
            final_array[i] = input_array[i]
        else:
            final_array[i] = 0

    # this is the resulting array from which to calculate a mean value
    return final_array


#########################
#   calc_Mean_Measure_Consec: A function to estimate the value needed to represent the mean of the passed array
#       replaces single call to statistics.mean(measures) with calc_Mean_Measure
#       Assumes have numpy and pandas loaded as np and pd
#       Calls generate_final_series to get the values from which to calculate the mean
#       Those points are used to calc a mean value for the weighed object
########
def calc_Mean_Measure_Consec(mydf, threshold = 400, myLen = 7):

  
    ## Get what is on screen - might eliminate them as arguments
    threshold = float(my_entries2_AUTO[1].get())
    myLen = float(my_entries2_AUTO[0].get())

    # Convert the Pandas Series to a NumPy array
    measures_array = mydf.values
   
    # Get a new list of values within threshold
    steady_points = generate_final_series(measures_array, threshold, myLen)

    # Create a new series composed of values in the fourth series that are > 0 - can I do this 
    filtered_final_series = steady_points[steady_points > 0]

    # Get the size of the array
    array_size = filtered_final_series.size

    ## print these for debugging purposes
    print("################# DEBUG: read values from screen ##############")
    print(f"Threshold: {threshold}")
    print(f"Length: {myLen}")
    print(f"SIZE OF ARRAY: {array_size}")

    if(array_size == 0):
        open_dialog("Error","Too few qualifying points.")
        return (0)

    # Take those points that qualify and get their mean and return it
    myMean = np.mean(filtered_final_series)

    ###### print info for debugging:
    # Count
    count = np.count_nonzero(filtered_final_series)

    # Range
    range_value = np.max(filtered_final_series) - np.min(filtered_final_series)

    # mean distance
    Diff_series = np.zeros_like(filtered_final_series, dtype=int)
    for i in range(1, len(Diff_series) -1):
        Diff_series[i]= abs(filtered_final_series[i]- filtered_final_series[i + 1])
    mean_Diff = np.mean(Diff_series)

    # Standard Deviation (STD)- may use this later for automation
    std_value = np.std(filtered_final_series)

    ## print these for debugging purposes
    print("################# DEBUG: calculation results ##############")
    print(f"Mean: {myMean}")
    print(f"Count: {count}")
    print(f"MeanDiff: {mean_Diff}")
    print(f"Range: {range_value}")
    print(f"Standard Deviation: {std_value}")
    ############ END of debugging print

    return myMean


##############
# do_Slope_0 - RAM 7/23/2024
#    Function to ID best value for conversion to bird weight
#    Parameters:
#    - measure_series: DataFrame slice containing the measure data.
#    - start_pt: Starting point (long integer). where to start and end the window counting
#    - stop_pt: Ending point (long integer).  could just be the lenght of the slice we are sending it when using real data
#    - min_len: Minimum length of windows (long integer). We have data that could determine the right number for this
#    - min_threshold: Minimum threshold for the intercept value (real number).
#    - max_pct: Max proportion of the passed series to be used to determine the max length of windows (real number, default is 0.5).
# 
#   Returns:
#    - Mean Value for calculation of mass
#########
def do_Slope_0(measure_series, start_pt, stop_pt, min_len, min_threshold, max_pct=0.5):

    do_print = True

    if isinstance(measure_series, pd.DataFrame):
        measure_series = measure_series.squeeze()

    max_len = int((stop_pt - start_pt) * max_pct)
    total_windows = max_len - min_len
    trace_len = stop_pt - start_pt - 1

    if(do_print):
        print(f"max_len: {max_len}")
        print(f"total_windows: {total_windows}")
        print(f"trace_len: {trace_len}")

    if total_windows <= 0:
        raise ValueError("total_windows must be greater than 0")

    rows = []

    for i in range(start_pt, stop_pt):
        for w in range(min_len, max_len):
            v_start = int(i - (w/2)) - 1
            v_stop = int(i + (w/2))

            if (w % 2 != 0):
                v_start = v_start - 1

            new_series = measure_series.iloc[v_start:v_stop]
            if(do_print):
                print(f"\nnew_series (i={i}, w={w}):\n{new_series}")

            if len(new_series) < 2:
                continue

            x_values = np.arange(len(new_series))
            slope, intercept = np.polyfit(x_values, new_series, 1)
            mean_val = new_series.mean()

            if(do_print):
                print(f"x_values: {x_values}")
                print(f"slope: {slope}")
                print(f"intercept: {intercept}")
                print(f"mean_val: {mean_val}")

            wt_calc = np.random.uniform(0, 100)

            row_data = {
                'slope': slope,
                'intercept': intercept,
                'wt_calc': wt_calc,
                'center_pt': i,
                'total_len': w,
                'v_start': v_start,
                'v_stop': v_stop,
                'mean_val': mean_val
            }

            rows.append(row_data)

    results_df = pd.DataFrame(rows, columns=['slope', 'intercept', 'wt_calc', 'center_pt', 'total_len', 'v_start', 'v_stop', 'mean_val'])
    results_df['slope_abs'] = abs(results_df['slope'])

    mymin = results_df['slope_abs'].min()

    if(do_print):
        print("Results DataFrame:")
        print(results_df.head())
        print(f"Min slope_abs value: {mymin}")

        # Get the row with the smallest values of 'slope_abs'
    filtered_df = results_df.nsmallest(1, 'slope_abs')
        # make sure it is above the background values
    filtered_df = filtered_df[filtered_df['mean_val'] >= min_threshold]

    if(do_print):
        print("Filtered DataFrame with 5 smallest slope_abs values:")
        print(filtered_df)
        print("Filtered DataFrame after mean_val threshold filter:")
        print(filtered_df.head())


    if not filtered_df.empty:
        min_abs_slope_mean = filtered_df['mean_val'].mean() 
        if(do_print):
            print(f"\nMean Value of the minimum absolute slope value row ({min_abs_slope_mean}):")
        return round(min_abs_slope_mean,1)
    

    else:
        print("\nFiltered DataFrame is empty.")
        return 0

