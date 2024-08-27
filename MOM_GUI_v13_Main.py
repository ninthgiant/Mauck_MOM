##############################
#
#    MOM_GUI_v13_Main.py - RAM, Aug 24, 2024
#       Builds on v_11 but which started automation process
#       -- automates Calibration
#       -- works great if have at least 2 seconds solid baseline before and next 3 are correct
#       -- does not go past calibration
#       Builds on v_12 which started process of bird weights 
#       -- works with do_Slope_0 method - adds Median and penguinW2 based on span found with do_Slope
#       -- right now, doSmoothie method doesn't work. Should use Liam's PR rule identifier for the above 3
#       Aim of v13
#       -- clean up code
#       -- format output for analysis
#       -- allow user to do all files within one folder - batch access, user doesn't open individual files
#
#       Needs
#       -- clear globals before each file opened, also, minimize use of globals
#       
#######################################

########
# import libraries, etc.
#################################
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import filedialog
from tkinter import messagebox as mb
from tkinter.simpledialog import askstring
from tkinter import *

#### imports from original Liam Taylor version
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backend_bases as backendgi
from setuptools import find_namespace_packages
import numpy as np
import statistics
from scipy import stats 
import os

########
# import MOM Methdos from MOM_Utility_v05 File - in same folder as Main file
#################################
# from MOM_Utility_v05 import Set_Globals
from MOM_Utility_v05 import open_dialog
from MOM_Utility_v05 import confirm_continue
# from MOM_Utility_v05 import get_user_input
from MOM_Utility_v05 import return_useful_name
from MOM_Utility_v05 import read_defaults_from_file # not using it yet

########
# import MOM Classes and methods from MOM_Graphing_v07 File - in same folder as Main file
#################################
from MOM_Graphing_v07 import DraggableMarker
from MOM_Graphing_v07 import AxesLimits
from MOM_Graphing_v07 import annotateCurrentMarkers

################
# Function Set_Globals to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: run at startup, but not yet doing that, using definitions above only
#           Would like it to be in MOM_Utility but doesn't work 6/6/24 so kept here
#######
def Set_Globals():
    # general info about the fle
    global user_INPATH
    global user_BURROW
    global data_DATE
    global the_Burrrow

    global data ## this  is a dataframe to be defined later
    global calibrations

    global cal_gradient
    global cal_intercept
    global baseline_cal_mean
    global cal_r_squared

    global calibrations
    global birds

    # Lists to accumulate info for different birds - Make these globals at top of app?
    global birds_datetime_starts
    global birds_datetime_ends
    global birds_data_means
    global birds_cal_means
    global birds_baseline_diff
    global birds_regression_mass
    global birds_details
    
    global myDir
    global default_window
    global my_Save_Dir
    global my_Save_Real_Dir
    global datafile_folder_path
    global datafiles

    global cal1_value  ## feature request - display after defaults set and allow user to change them in real time
    global cal2_value
    global cal3_value

    global my_SPS ## set the SPS used for time calculation - how manypoints per second based on unix time stamp diff betwen success 300 pts on 6/19/2024
    my_SPS = 60

    global my_std  # to keep track of the automation parameters
    global my_rolling_window
    global my_inclusion_threshold

    global my_Continue
    global vVersString
    global vAppName
    vVersString = " (v_09)"  ## upDATE AS NEEDED - Julhy19, 2024 - added scrollbars
    vAppName = "Mass-O-Matic Analyzer" + vVersString
   
    ### now make the arrays for the exported data
    birds_datetime_starts = []
    birds_datetime_ends = []
    birds_data_means = []
    birds_cal_means = []
    birds_baseline_diff = []
    birds_regression_mass = []
    birds_details = []

    aDefaults = []  # NOT USED will be used when we transition to non-python user default settings


#########################################################################
#   Define the functions to be used for the 3 buttons in the interface
#       June 8, 2024 - have 3 buttons
#           just explore files
#           do calculation with simple averages
#           do automated calculation
#########################################################################

###################
# mom_cut_button - bad name - used to measure weight from trace
#   argument to say what kind of averaging you will use
#   June 8, 2024 - just 2 types possible:
#                   simple mean of designated subset of a trace
#                   automated calculation of the designated subset of a trace
####
def mom_cut_button(my_Mean_Type):
    pass
    ## get a file to work with, then send it here...
    bird_fname, bird_df = mom_open_file_dialog("not") 

    if(bird_fname == None):     # catch no file chosen before it is a problem
        return None             # just shut it down

    user_BURROW = return_useful_name(bird_fname) 
    my_Continue = True

    global cal_gradient
    global cal_intercept

    if 'cal_gradient' in globals(): # do we already have this calculated from a previous bird?

        print("Cal gradent and intercept:")
        print (str(cal_gradient))
        print (str(cal_intercept))

        if(confirm_continue("Use last bird calibration?")):
            my_Continue = True
        else:
            my_Continue == my_Do_Calibrations(bird_df)

    else:
        my_Continue == my_Do_Calibrations(bird_df)

    if(my_Continue):
        Do_Multiple_Birds(bird_df, my_Mean_Type) # NEED TO ADD - alert if this isn't true
        

###################
# mom_open_file_dialog
#   user chooses a file, gets a plot of that file - stand alone or prep for calculations
#   June 8, 2024 - just 2 types possible:
#                   simple mean of designated subset of a trace
#                   automated calculation of the designated subset of a trace
####
def mom_open_file_dialog(to_show):
    global data_DATE

    my_testing = False # delete this on cleanup
    
    f_types = [('CSV files',"*.csv"), ('TXT',"*.txt") ]
    if(not(my_testing)):  # make false for testing
        f_name = filedialog.askopenfilename(initialdir = myDir,  title = "Choose MOM File", filetypes = f_types)
    else:
        f_name = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/185_One_Bird_70K_110K.TXT"

    # the_Burrrow = "999"

    # dispName = the_Burrow + ": " + f_name[(len(f_name)-60):len(f_name)]

    if(len(f_name)==0):
        open_dialog("Error","No file chosen. Try again.")
        return None, None


    global user_BURROW # get the name in a format we can use
    user_BURROW = return_useful_name (f_name) # f_name[len(myDir):len(f_name)]
    
    #### change here by RAM, 9/3/2022 to revert to old way of getting the dataframe - from lhsp_mom_viewer
    if(FALSE):
        df = pd.read_csv(f_name, header=None, skiprows=1)
    else:
        try:
            user_INPATH = f_name ## Get_File_Info()
        
            my_data = pd.read_csv(user_INPATH, header=None, names=["Measure", "Datetime"], 
                                encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", 
                                engine="python")
            my_data["Measure"] = pd.to_numeric(my_data["Measure"], errors="coerce")            

            # Convert Unix timestamp to datetime - later use this and get the following format: .strftime('%Y-%m-%d %H:%M:%S')
            my_data["Datetime"] = pd.to_datetime(my_data["Datetime"], unit='s', utc=False, errors="coerce")
            my_data["Datetime"] = my_data["Datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')

            # We've possibly forced parsing of some malformed data
            #   ("replace" utf-8 encoding errors in read_csv() 
            #     and "coerce" datetime errors in to_numeric() and to_datetime()), so 
            #   now we need to clean that up.
            # Simply drop all points from the file where Measure has been coerced to NaN
            #   and where Datetime has been coerced to NaT
            my_data = my_data[~my_data.Measure.isnull() & ~my_data.Datetime.isnull()]

        except FileNotFoundError:
            sys.exit("Input file not found. Exiting. FileNotFoundError")
        except pd.errors.EmptyDataError:
            sys.exit("Error parsing empty file. Exiting. EmptyDataError")
        except pd.errors.ParserError:
            sys.exit("Error parsing input file. Exiting. ParserError")
        except Exception as e:
            sys.exit("Error parsing input file. Exiting. {}".format(e))

        # Display input information
        data_DATE = my_data.Datetime.iloc[0]

        df = my_data 

        ## show info to user
        display_string = mom_get_file_info(df)
        display_string = return_useful_name (f_name) + " - Summary Info" + "\n" + display_string + "\n"
        t1.insert("1.0", display_string)
        #### end of showing info to user 

    if (to_show == "cut"): 
        the_start = int(my_entries[0].get())
        the_end = int(my_entries[1].get())
        df = df.iloc[the_start:the_end]
        df.plot()
        plt.show()

    if (to_show == "all"):
        fig, ax = plt.subplots()
        ax.plot(df.loc[:,"Measure"])
        ax.set_title(return_useful_name (f_name))
        plt.show()

    if(to_show == "not"):
        pass

    return f_name, df

###################
# mom_auto_button - bad name - used to measure weight from trace
#   argument to say what kind of averaging you will use
#   June 8, 2024 - just 2 types possible:
#                   simple mean of designated subset of a trace
#                   automated calculation of the designated subset of a trace
####
def mom_auto_button(to_do):
    pass
    print(to_do)

    ## get a file to work with, then send it here...
    bird_fname, bird_df = mom_open_file_dialog("not") 

    if(bird_fname == None):     # catch no file chosen before it is a problem
        return None             # just shut it down

    user_BURROW = return_useful_name(bird_fname) 
    my_Continue = True

    global cal_gradient
    global cal_intercept



    if 'cal_gradient' in globals(): # do we already have this calculated from a previous bird?

        print("Cal gradent and intercept:")
        print (str(cal_gradient))
        print (str(cal_intercept))

        if(confirm_continue("Use last bird calibration?")):
            my_Continue = True
        else:
            my_Continue, cal_gradient, cal_intercept, cal_r_squared, cal1_diff = do_Auto_Calibration(bird_df, 13, 34, 52, 1, 60 )

    else:
        print("now clear to go do AUto Calib")
        my_Continue, cal_gradient, cal_intercept, cal_r_squared, cal1_diff = do_Auto_Calibration(bird_df, 13, 34, 52, 1, 60 )

    if(my_Continue):
        pass
        # do the auto birdd finding and measuring - using the bird_df and all the returns from do_AutoCalibration
        do_Auto_Bird(bird_df, cal_gradient, cal_intercept, cal_r_squared, cal1_diff)
        # Do_Multiple_Birds(bird_df, my_Mean_Type) # NEED TO ADD - alert if this isn't true
    else:
        print("bad calibration, can't continue")   
        mb.showinfo("Cannot auto-calibrate. R2 = ", str(cal_r_squared) + ". NOT GOOD.")

        



###################
# mom_get_file_info
#   argument: dataframe builty from opening a file to analyze
#   June 8, 2024 - added the SPS to calculate duration of the dataframe
#   returns a string with all the relevant information
####
def mom_get_file_info(my_df):
    global my_SPS
    print(str(my_SPS) + "SPS" + "\n")
    #### show user info on the file chosen
    str1="\tPoints:" + str(my_df.shape[0])+ "\t\tColumns:"+str(my_df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
    str2="\tMinutes: " + str(round((my_df.shape[0])/my_SPS/60,2))+"\t"
    str3="(" + str(round((my_df.shape[0])/my_SPS/60/60,2))+" hours)\n"
    str4 = "\tMean Strain: " + str(round(my_df["Measure"].mean())) + "\n"

    return(str1 + str2 + str3 + str4)


###############################################################################################
#   Calculation methods
#       June 8, 2024
#       Includes the main workhorse GetTracePointPair
#       Includes calibaration methods
#       Includes methods needed for automation methods
#       Only one automated method - get stable points from wihtin the designated trace
############################################################################################


#########################################################
#  getTracePointPair:  A function to set a pair of draggle points on an interactive trace plot
#   Original heavy lifting from Liam Taylor - eventually move it to external file; make a module
#   RETURNS
#       mean, markers, isGood
#       mean -- mean strain measurement value between two marked points
#       markers -- dataframe of marker information, including start and end index on the trace
#       isGood -- boolean confirmation that the plot wasn't closed before both points were marked
#       newAxesLimits -- bounding box limits for the plot view that was shown right before exiting 
###################
def getTracePointPair(my_df, category, markers=None, axesLimits=None):

    data = my_df
    # Print a message on the screen for relevant step in the process
    my_Push_Enter = "Add {category} start point, then press enter.".format(category=category)

    # Turn on the Matplotlib-Pyplot viewer - interactive mode
    # Shows the trace from the globally-defined data
    plt.ion()
    fig, ax = plt.subplots()

    fig.set_size_inches((20,4))  # (default_figure_width, default_figure_height)) 
 
    ax.plot(data.loc[:,"Measure"])

    if (axesLimits is not None):
        ax.set_xlim(left=axesLimits.xstart, right=axesLimits.xend)
        ax.set_ylim(bottom=axesLimits.ystart, top=axesLimits.yend)

    if (markers is not None):
        # Add any previous markers
        annotateCurrentMarkers(markers)

    # Initialize the draggable markers
    dm = DraggableMarker(category=category, startY=min(data["Measure"]))
    ax.set_title(my_Push_Enter)
    plt.show(block=True)

    plt.ioff()

    # Gather the marked points data
    index_start = min(dm.index_start, dm.index_end)
    index_end = max(dm.index_start, dm.index_end)
    time_start = data.loc[index_start,"Datetime"]
    time_end = data.loc[index_end,"Datetime"]
    measures = data.loc[index_start:index_end,"Measure"]
   
    # Create a new variable as a pandas Series
    if category == "Bird Data Auto":   #  automation calculations - change to call to new function

        # reduce the size of array to only qualifying points - automation
        measures_series = pd.Series(measures, name='Measures Series')
        
        mean = calc_Mean_Measure_Consec(measures_series, 400,7)  # change to get screen values
    else:
        mean = statistics.mean(measures)  # just calculate on whatever is in the array 'measures'

    # Extract the axes limits for the final interactive plot view
    # in case the user wants to use those limits to restore the view on the next plot
    endView_xstart, endView_xend = ax.get_xlim()
    endView_ystart, endView_yend = ax.get_ylim()
    newAxesLimits = AxesLimits(endView_xstart, endView_xend, endView_ystart, endView_yend)

    # Confirm the plot was not exited before both points were marked
    isGood = dm.isGood

    print("""
    Measured {category} from {start} to {end}.
    Mean {category} measurement is {mean}.
    """.format(category=category, start=time_start, end=time_end, mean=round(mean,2)))

    # Create a dataframe with information about the marked points
    markers = pd.DataFrame({"Category":category,
                                "Point":["Start", "End"],
                                "Index":[index_start, index_end],
                                "Datetime":[time_start, time_end],
                                "Measure":[data.loc[index_start,"Measure"], data.loc[index_end,"Measure"]]})
    markers = markers.set_index("Index")

    return mean, markers, isGood, newAxesLimits


#########################
#   generate_final_series: A function to find consecutive values that are
#       1) with a threshold value of the previous and subsequent values
#       2) are part of a sequence of at least myLen length
#       -- Work is done on a SERIES not a dataframe, so have convert data to be received
#       TO DO:
#           1) need to catch a return array that has no values - done 2023
#           2) need a user input for threshold and length values - done 6/2/24
########
def generate_final_series(input_array, threshold, myLen):

    print("################# DEBUG: read values from screen inside generate_final_series ##############")
    print(f"Threshold: {threshold}")
    print(f"Length: {myLen}")

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

################
#  calc_W2_Penguin: Algorithm from Afanasyev et al. 2015
#    RAM 6/25/2024 with help from ChatGPT
#       receives the subset of the dataframe that was used to calculate W1 (mean values converted to grams)
#       receives W1 which was the old way of calculating weight
#       receives sampling rate - use global SPS unless reason not to
#       Penguin Calculation
#          a = Get Slope -> between the points (start and end)
#          g = 9.8
#          W1 = Mean Value between the two points - all normal to this point
#          W2 = W1 ( 1 + a/g)
#    
#    returns value for W2 increases accuracy
#    use in do_Bird - called as:  calc_W2_Penguin(measure_series, W1, cal_gradient, cal_intercept, SPS):
#######
def calc_W2_Penguin(measure_series, W1, calibration_slope, calibration_intercept, SPS=60):
    # Define gravitational constant
    g = 9.81
    
    # Apply calibration to convert raw values to grams
    measure_series_grams = calibration_slope * measure_series + calibration_intercept
    
    # Calculate original slope and intercept of the calibrated 'Measure' series
    x_values = np.arange(len(measure_series_grams))
    slope, intercept = np.polyfit(x_values, measure_series_grams, 1)

    print("Slope:    ################")
    print(round(slope, 2))

    
    # Use the slope as is, don't change for sampling interval. the paper doesn't change it, either and they sample at 200 sps
    sampling_interval = SPS
    adj_slope = slope # / sampling_interval
    
    # Calculate W2 using the adjusted slope
    W2 = W1 * (1 + adj_slope / g)
    
    return W2


###################
#   remove_outliers_and_count by ChatGPT
#    returns tuple with information about the result of the calculations
#    code below is hte complete do_Bird, but need to figure out where this goes and how it works with Do_Bird or mom_cut_Button
#######
def remove_outliers_and_count(measure_series):
    # Calculate the mean and standard deviation of the series
    mean_value = measure_series.mean()
    std_value = measure_series.std()
    
    # Identify the upper and lower bounds for outliers
    upper_bound = mean_value + 2 * std_value
    lower_bound = mean_value - 2 * std_value
    
    # Filter the series to identify outliers
    above_2std = measure_series[measure_series > upper_bound]
    below_2std = measure_series[measure_series < lower_bound]
    
    # Calculate the number of outliers
    num_above_2std = len(above_2std)
    num_below_2std = len(below_2std)
    
    # Filter the series to remove outliers
    reduced_series = measure_series[(measure_series <= upper_bound) & (measure_series >= lower_bound)]
    
    return reduced_series, num_above_2std, num_below_2std

import pandas as pd
import numpy as np

##############
# do_Slope_0 - RAM 7/23/2024
#    Function to ID best value for conversion to bird weight
#    Parameters:
#    - measure_series: DataFrame  containing the measure data. Get the the whole dataframe, so you can go oustide bounds
#       - of the window of interst of your f\jull slope measures
#    - start_pt: Starting point (long integer). where to start and end the window counting
#    - stop_pt: Ending point (long integer).  could just be the lenght of the slice we are sending it when using real data
#    - min_len: Minimum length of windows (long integer). We have data that could determine the right number for this
#    - min_threshold: Minimum threshold for the intercept value (real number).
#    - max_pct: Max proportion of the passed series to be used to determine the max length of windows (real number, default is 0.5).
# 
#   Returns:
#    - Mean Value for calculation of mass AND starting point and width of window used
#########
def do_Slope_0(measure_series, start_pt, stop_pt, min_len, min_threshold, max_pct=0.5):

    do_print = False

    #if isinstance(measure_series, pd.DataFrame):
     #   measure_series = measure_series.squeeze()

    max_len = int((stop_pt - start_pt) * max_pct)
    if(max_len <= min_len):
        max_len = min_len + 2 # (stop_pt - start_pt)
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
            slope, intercept = np.polyfit(x_values, new_series['Measure'], 1)
            mean_val = new_series['Measure'].mean()

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
        # Find the index of the row with the minimum 'slope_abs'
        min_slope_abs_index = results_df['slope_abs'].idxmin()

        # Retrieve the values of 'v_start' and 'total_len' for that row
        min_row = filtered_df.loc[min_slope_abs_index]
        v_start = min_row['v_start']
        total_len = min_row['total_len']
        min_slope = results_df['slope'].iloc[min_slope_abs_index]
        if(do_print):
            print(f"\nMean Value of the minimum absolute slope value row ({min_abs_slope_mean}):")
        return round(min_abs_slope_mean,1), min_slope, v_start, total_len
    

    else:
        print("\nFiltered DataFrame is empty.")
        return 0, 999, 0, 0

##############
# doSmoothWindowMean - LUT 7/29/2024
#    Function to calculate the mean value from a smoothed window
#       the size and position of the window is selected automatically, 
#       being the size that reduces the number of first-derivative switches in the trace
#       and the position of the window that maximizes the value 
#       (this finds the smooth "peak" at the center an elevated trace) 
#    Parameters:
#    - measure_series: DataFrame containing the measure data.
#    - start_pt: Starting point (long integer). where to start and end the window counting
#    - stop_pt: Ending point (long integer).  could just be the lenght of the slice we are sending it when using real data
#    - baseline: The baseline strain measurement value (to smooth jitters around the baseline)
# 
#   Returns:
#    - Mean strain measurement at the chosen smoothing window size (not calibrated to grams) 

def doSmoothWindowMean(measure_series, start_pt, stop_pt, baseline):
    measure = -1
    minDZeros = sys.maxsize
    window = measure_series.iloc[start_pt:(stop_pt+1)]

    print("------------ Check baseline for smoothing")
    # remove the jitters around the baseline, by setting the baseline flat
    # window[window < baseline] = baseline
    print("------------ DONE-Baseline:")
    print(baseline)
    print("start loop")
    # loop through all window sizes 
    for windowSize in range(1, window.size):
        # Smooth trace to window size
        rolled = window.rolling(windowSize, center=True).mean().fillna(baseline)
        # Get first derivative
        ds = rolled.diff()
        # Calculate sign switches in first derivative 
        signSwitches = np.sign(ds).diff().fillna(0)
        currDZeroes = np.count_nonzero(signSwitches)

        # If this window size has equal to the number of fewest changes so far,
        # take the measurement from this window size 
        if (currDZeroes <= minDZeros):
            measure = rolled.max()
            minDZeros = currDZeroes
            print("new measure")
    
    # After the loop, you will have the strain measurement from the largest window size
    # that smooths the window most effectively to a single point
    return measure


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

########
#   find_stretches_with_means
    """
    Finds four 60-line stretches within the first 2000 lines of the DataFrame with specific mean values and low variation.
    
    Parameters:
    - dataframe: pd.DataFrame with columns 'Measure' and 'Datetime'
    - window_size: Number of lines in each stretch (default is 60)
    - multipliers: List of multipliers to find the target mean values
    - std_threshold: Standard deviation threshold to consider for low variation (default is 0.01)
    - tolerance: Tolerance level to match the target mean value (default is 0.001)
    
    Returns:
    - A tuple with four mean values: baseline_cal_mean, cal1_value, cal2_value, cal3_value
    """
#####
import pandas as pd

def find_stretches_with_means(dataframe, window_size=60, multipliers=[15.97, 32.59, 50.22], std_threshold=500, tolerance=5000):
    """
    Finds four 60-line stretches within the first 2000 lines of the DataFrame with specific mean values and low variation.
    
    Parameters:
    - dataframe: pd.DataFrame with columns 'Measure' and 'Datetime'
    - window_size: Number of lines in each stretch (default is 60)
    - multipliers: List of multipliers to find the target mean values (default is [15.97, 32.59, 50.22])
    - std_threshold: Standard deviation threshold to consider for low variation (default is 500)
    - tolerance: Tolerance level to match the target mean value (default is 5000)
    
    Returns:
    - A tuple with four mean values: baseline_cal_mean, cal1_value, cal2_value, cal3_value
    """
    # Slice the first 2000 lines
    subset_data = dataframe.head(3000)
    
    # Calculate rolling mean and standard deviation
    rolling_mean = subset_data['Measure'].rolling(window=window_size).mean()
    rolling_std = subset_data['Measure'].rolling(window=window_size).std()
    
    # Drop NaN values created by the rolling window
    rolling_mean = rolling_mean.dropna()
    rolling_std = rolling_std.dropna()
    
    # Combine the results into a DataFrame for easier analysis
    results = pd.DataFrame({
        'Mean': rolling_mean,
        'Std': rolling_std
    })
    
    # Find the index of the row with the lowest mean and lowest variation
    min_mean_index = results['Mean'].idxmin()
    min_std_index = results['Std'].idxmin()
    
    # Check which index satisfies both conditions
    baseline_index = min_mean_index if min_std_index == min_mean_index else results['Std'].idxmin()
    
    # Extract the baseline window
    baseline_cal_mean = subset_data.iloc[baseline_index:baseline_index + window_size]['Measure'].mean()
    
    # Initialize the variables for higher mean values
    cal_values = [None] * len(multipliers)
    
    # Iterate over the multipliers to find the corresponding windows
    for i, multiplier in enumerate(multipliers):
        # Calculate the target mean using the new formula
        target_mean = baseline_cal_mean + ((multiplier - 0.015) / 0.00058)
        print(f"Target mean for multiplier {multiplier}: {target_mean}")
        
        # Find the indexes within the tolerance range of the target mean
        lower_bound = target_mean - tolerance
        upper_bound = target_mean + tolerance
        print(f"Searching within range: {lower_bound} to {upper_bound}")
        
        potential_indexes = results[(results['Mean'] >= lower_bound) & (results['Mean'] <= upper_bound)].index
        
        if not potential_indexes.empty:
            closest_index = potential_indexes[0]
            # Extract the window
            candidate_mean = subset_data.iloc[closest_index:closest_index + window_size]['Measure'].mean()
            candidate_std = results.loc[closest_index, 'Std']
            
            print(f"Found candidate mean: {candidate_mean} at index {closest_index}")
            print(f"Candidate Std Dev: {candidate_std}")
            
            # Ensure the variation is low
            if candidate_std < std_threshold:
                cal_values[i] = candidate_mean
            else:
                print(f"Candidate standard deviation {candidate_std} exceeds threshold {std_threshold}")
        else:
            print(f"No valid windows found for target mean {target_mean} within range {lower_bound} to {upper_bound}")
    
    # Print values before returning
    print(f"Baseline Mean Value: {baseline_cal_mean}")
    for i, cal_value in enumerate(cal_values):
        print(f"Calibration {i + 1} Value: {cal_value}")
    
    return baseline_cal_mean, *cal_values

############################
#
#   detect_sections
#       RAM, Aug 21, 2024
#
#    Detect sections in the first 5000 lines where values rise above and fall back to the baseline value.
#    Plot the data with vertical lines at start and stop points of detected sections.
#    ALSO WORKS WITH NORMAL BIRD TRACES?
#    WORKS FOR CALIBRATION AS LONG AS THERE IS AT 2 SECONDS START AT BASEILNE AND 3 WEIGHTS IN ORDER
#
#    Parameters:
#    - dataframe: pd.DataFrame with a 'Measure' column
#    - baseline_fraction: Fraction of the baseline to determine thresholds (default is 0.0002)
#    - num_sections: Number of sections to detect (default is 5)
#    - window_size: Minimum number of lines for detected section (default is 75) - default may cause some not to work
#    - std_threshold: Standard deviation threshold for baseline calculation (default is 500) - should be 150 for baseline
#    - tolerance: Tolerance for deviation from baseline mean (default is 5000)
#    - lines - how much of the file to consider. Defaults to 5000 for the header - 0 is code for entire
#    
#    Returns:
#    - A tuple with the baseline value and a DataFrame with columns 'Section', 'Start', and 'Stop' indicating the detected sections
#   
##############
def detect_sections(dataframe, baseline_fraction=0.0002, num_sections=10, window_size=75, std_threshold=500, tolerance=5000, lines = 5000):
 
    # Slice the first 5000 lines - make this a parameter - ONLY FOR Calibration does this work!
    if(lines > 0):
        subset_data = dataframe.head(5000)
    else:
        subset_data = dataframe
    
    # Calculate rolling mean and standard deviation
    rolling_mean = subset_data['Measure'].rolling(window=window_size).mean()
    rolling_std = subset_data['Measure'].rolling(window=window_size).std()
    
    # Drop NaN values created by the rolling window
    rolling_mean = rolling_mean.dropna()
    rolling_std = rolling_std.dropna()
    
    # Combine the results into a DataFrame for easier analysis
    results = pd.DataFrame({
        'Mean': rolling_mean,
        'Std': rolling_std
    })

        # Find the index of the row with the lowest mean and lowest variation
    min_mean_index_before = results['Mean'].idxmin()
    min_std_index_before = results['Std'].idxmin()
    print(f"BEFORE Filtering: mean_min index: {min_mean_index_before}, mean_std index: {min_std_index_before}")

    print(f"Length of results dataframe: {len(results)}")
    filtered_results = results[results['Std'] < std_threshold]
    if(len(filtered_results)> 0):
        # added to fix problem with 383 - gets rid of all the very low values with lots of variation
        results = results[results['Std'] < std_threshold]
        print(f"Throw out all with STD > {std_threshold}")
        print(f"Length of results now: {len(results)}: must redo indices")
        results = results.reset_index(drop=True)
        
    else:
        print("keeping messy data to find min")
    
    

    # Find the index of the row with the lowest mean and lowest variation
    min_mean_index = results['Mean'].idxmin()
    min_std_index = results['Std'].idxmin()
    print(f"AFTER Filtering: mean_min index: {min_mean_index_before}, mean_std index: {min_std_index_before}")

    
    # Check which index satisfies both conditions - tried to fix for 383 so Mean min takes precedence; didn't work
    baseline_index = min_mean_index if min_std_index == min_mean_index else results['Std'].idxmin()

    if(TRUE):  ### added to fix 383
        
        if(min_mean_index == min_std_index):
            print("Both mean and std are minima")
            baseline_index = min_std_index
        else:
            print(f"Mean of mean_min section: {min_mean_index} not same section as min_std section: {min_std_index}")
            # Get the value at index X for 'Mean' and 'Std'
            mean_value_at_minMean = results.iloc[min_mean_index]['Mean']
            std_value_at_minMean = results.iloc[min_mean_index]['Std']

            mean_value_at_minSTD = results.iloc[min_std_index]['Mean']
            std_value_at_minSTD = results.iloc[min_std_index]['Std']
            print(f"Mean of mean_min section: {min_mean_index} is: {mean_value_at_minMean}, STD: {round(std_value_at_minMean,1)}")
            print(f"Mean of mean_std section: {min_std_index} is: {mean_value_at_minSTD}, STD: {round(std_value_at_minSTD,1)}")
            print("using the Mean value. compare and use:")
            if(mean_value_at_minMean < mean_value_at_minSTD):
                print(f"using the min_mean_index {min_mean_index}")
                baseline_index = min_mean_index
            else:
                print(f"using the min_std_index {min_std_index}")
                baseline_index = min_std_index

        
        # put back in to overload all that - repeates above
    # overrid the above 
    baseline_index = min_mean_index if min_std_index == min_mean_index else results['Std'].idxmin()
    
    # Extract the baseline window
    baseline_cal_mean = subset_data.iloc[baseline_index:baseline_index + window_size]['Measure'].mean()
    
    # Define thresholds
    high_threshold = baseline_cal_mean + baseline_fraction * baseline_cal_mean
    #low_threshold = baseline_cal_mean - baseline_fraction * baseline_cal_mean  ### changed this
    low_threshold = baseline_cal_mean + (baseline_fraction*.8) * baseline_cal_mean  ### changed this
    print(f"Hi threshold: {high_threshold}")
    print(f"Lo threshold: {low_threshold}")

    # Initialize variables
    in_section = False
    start_idx = None
    detected_sections = []
    
    # Step through the data to detect rises and falls
    for i in range(len(subset_data)):
        value = subset_data['Measure'].iloc[i]
        
        if value > high_threshold and not in_section:
            # Start of a new section
            start_idx = i
            in_section = True
        elif value < low_threshold and in_section:
            # End of the current section
            stop_idx = i
            if (stop_idx - start_idx) > window_size:  # Check window size condition
                detected_sections.append((start_idx, stop_idx))
            in_section = False
            
            # Stop if we've detected the required number of sections
            if len(detected_sections) >= num_sections:
                break
    
    # Create a DataFrame to store the section start and stop indices
    sections_df = pd.DataFrame(detected_sections, columns=['Start', 'Stop'])
    sections_df.index.name = 'Section'
    sections_df.index += 1  # Start numbering from 1
    
    # print(f"Sections found: len({sections_df})")

    # Filter out sections where the difference between start and stop is less than window_size
    # sections_df = sections_df[sections_df['Stop'] - sections_df['Start'] > window_size]
    
    # Print the baseline and detected sections
    print(f"Baseline Mean Value: {baseline_cal_mean}")
    print("Detected Sections:")
    print(sections_df)
    
    if(FALSE):
        # Plot the raw data with vertical lines for start and stop points
        plt.figure(figsize=(12, 6))
        plt.plot(subset_data['Measure'], label='Raw Data')
        
        for _, row in sections_df.iterrows():
            plt.axvline(x=row['Start'], color='r', linestyle='--', label=f'Start {row.name}')
            plt.axvline(x=row['Stop'], color='g', linestyle='--', label=f'Stop {row.name}')
        
        plt.xlabel('Index')
        plt.ylabel('Measure')
        plt.title('Raw Data with Detected Sections')
        plt.legend()
        plt.show()
    
    return baseline_cal_mean, sections_df

##########################
#
#   do_Calib_Calc
#       RAM Aug 20, 2024
#       Parameters:
#           - dataframe: pd.DataFrame with a 'Measure' column
#           - sections_df: DataFrame with 'Start' and 'Stop' columns indicating detected sections within that dataframe
#
#       returns: slope, intercept, r-squared for the calibration calculation
#
##########

def do_Calib_Calc(dataframe, sections_df, mean_baseline, cal1_value, cal2_value, cal3_value):
    """
    Print the mean values of each detected section from the provided sections DataFrame.
    
    Parameters:
    - dataframe: pd.DataFrame with a 'Measure' column
    - sections_df: DataFrame with 'Start' and 'Stop' columns indicating detected sections
    
    Returns:
    - None
    """
    # Check if there are at least 3 sections
    if len(sections_df) < 3:
        print("Fewer than 3 sections detected.")
        return
    
    # Iterate through the first 3 detected sections
    for index, row in sections_df.head(3).iterrows():
        start_idx = row['Start']
        stop_idx = row['Stop']

        # first do it by removing all the wierd stuff 
        if(FALSE):
            measure_series = dataframe["Measure"][start_idx:stop_idx]
            reduced_series, n_above, n_below = remove_outliers_and_count(measure_series)
            reduced_mean = reduced_series.mean()
            reduced_diff = reduced_mean - mean_baseline
            print(f"Section {index + 1}: REDUCED, Mean Value={reduced_mean:.2f}, Mean Diff={reduced_diff:.2f}")
                                    # returns round(min_abs_slope_mean,1), mymin, v_start, total_len
        section_mean, section_slope, section_start, section_len = do_Slope_0(dataframe, start_idx, stop_idx, 60, 0, 0.7)
        section_diff = section_mean - mean_baseline
        sections_df.at[index,'sec_Mean'] = section_mean
        sections_df.at[index,'sec_Diff'] = section_diff
        # section_mean = dataframe.iloc[start_idx:stop_idx]['Measure'].mean()
        
        
        print(f"Section {index + 1}: Start={start_idx}, Stop={stop_idx}, Mean Value={section_mean:.2f}, Mean Diff={section_diff:.2f}")
    print(sections_df)

    print("Now the calculations")


    cal1_diff = sections_df['sec_Diff'].iloc[0]
    cal2_diff = sections_df['sec_Diff'].iloc[1]
    cal3_diff  = sections_df['sec_Diff'].iloc[2]

    cal1_mean = sections_df['sec_Mean'].iloc[0]
    cal2_mean = sections_df['sec_Mean'].iloc[1]
    cal3_mean = sections_df['sec_Mean'].iloc[2]

    calibrations = pd.DataFrame({"Category":["Cal1", "Cal2", "Cal3"],
                                    "Value_True":[cal1_value, cal2_value, cal3_value],
                                    "Value_Measured":[cal1_mean, cal2_mean, cal3_mean]})
    calibrations["Value_Difference"] = abs(calibrations["Value_Measured"] - mean_baseline)

    # Get the linear regression information across the three calibration points
    cal_gradient, cal_intercept, cal_r_value, cal_p_value, cal_std_err = stats.linregress(calibrations["Value_Difference"], calibrations["Value_True"])
    cal_r_squared = cal_r_value**2

    print(f"R-squared: {cal_r_squared}")
    print(f"Slope: {cal_gradient}")
    print(f"Intercept: {cal_intercept}")

            #### only show this during de-bugging
    if(FALSE):
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
        # t2.insert("1.0", "File: " + user_BURROW + " - Calibration regression:" + "\n") # add to Text widget

    ### return diff - for now, use cal2 because it is 32g., may want to use different vlaue if we change calib weights - user chooseable?
    min_diff_bird = cal2_mean - baseline_cal_mean

    return cal_gradient, cal_intercept, cal_r_squared, min_diff_bird








################
# do_Auto_Calibration: to get the calibration values for the MOM on this burrow-night automatically
#    RAM 8/18/24
#    Overview:
#       1) look at the first 1 minute (find safe starting point
#       2) assumes you have background, low, med, hi for calibrations
#       3) get a 1 second window
#       4) take means of each of those windows
#           - store values for each of those
#       5) look for 3 different levels of means - should be background, low, middle hi
#       6) apply calibration calculations to those 3 levels
#       7) return slope and intercept and FLAG for OK?
#       
#    parameters: 1) my_dataframe -> data to work with 2) 3 calibration values from user (don't use globals) 3) SPS for setting window size in seconds, usually 60
#       When calling this can get these with:   
#           cal1_value = float(my_entries2[0].get())
#           cal2_value = float(my_entries2[1].get())
#           cal3_value = float(my_entries2[2].get())
#    returns tuple with information about the result of the calculations for the calibration - cal_gradient, cal_intercept, baseline_cal_mean, cal_r_squared
#
#######
def do_Auto_Calibration(my_dataframe, cal1_value, cal2_value, cal3_value, win_size, the_SPS = 60):

    print("Entering Auto_Calibration")
    data = my_dataframe
    
    global calibrations

    # assign value to these
    global cal_gradient
    global cal_intercept
    global baseline_cal_mean
    global cal_r_squared

    print("---- starting auto calibration ----")

    if(TRUE):
        # Call the function with: dataframe, baseline_fraction=0.0002, num_sections=10, window_size=75, std_threshold=500, tolerance=5000)
        print("----- in auto calibration ----")
        baseline_cal_mean, sections_df = detect_sections(my_dataframe, 0.0002, 10, window_size=80, std_threshold=150, tolerance=5000)

        # Print the returned values
        print(f"Baseline Mean Value: {baseline_cal_mean}")
        print("Detected Sections:")
        print(sections_df)

        ## Should pass these as an argument...
        cal1_value = float(my_entries2[0].get())
        cal2_value = float(my_entries2[1].get())
        cal3_value = float(my_entries2[2].get())

        cal_gradient, cal_intercept, cal_r_squared, cal1_diff = do_Calib_Calc(my_dataframe, sections_df, baseline_cal_mean, cal1_value, cal2_value, cal3_value)

        if(FALSE):
            baseline_cal_mean, cal1_mean, cal2_mean, cal3_mean = find_stretches_with_means(my_dataframe, 50, [15.97, 32.59, 50.22], 50000, 500)

        if (abs(cal_r_squared) < 0.999):
            print("bad r2")
            my_Continue = False
        else:
            my_Continue = True

    print("Leaving Auto_Calibration")
        #include diff in cal1_mean to guage lowest possible value for a bird, return what we need to find/measure bird weights
    return my_Continue, cal_gradient, cal_intercept, cal_r_squared, cal1_diff 



###############################################################################################
#   Basic Handling methods for birds
#       Individual Bird, Multiple birds, and exporting data from measuring birds
#       June 8, 2024
############################################################################################


#############################
# Function Do_Bird_Baseline: does automatic (no user input) ID of baseline value for a bird trace
#   RAM 8/21/24
#
#   Builds a list of possible traces in the fiel up to a max value (like in autocalc, where used on first 3; use rest)
#   For all traces > 3 (calibrations), calculate value from start-end points
#
#   parameters
#       num_traces - how many traces will we evaluate?
#   returns: Mean baseline value
#######
def do_Bird_Baseline(my_bird_df, start_index, stop_index, buffer_size=800, window_size=40):
    # Define new indices for the buffer
    new_start_index = start_index - buffer_size
    new_stop_index = stop_index + buffer_size

    # Create subsets before and after the specified indices
    subset_before = my_bird_df.iloc[new_start_index:start_index]
    subset_after = my_bird_df.iloc[stop_index:new_stop_index]

    # Calculate rolling mean and standard deviation for the 'before' subset
    rolling_mean_before = subset_before['Measure'].rolling(window=window_size).mean().dropna()
    rolling_std_before = subset_before['Measure'].rolling(window=window_size).std().dropna()

    # Calculate rolling mean and standard deviation for the 'after' subset
    rolling_mean_after = subset_after['Measure'].rolling(window=window_size).mean().dropna()
    rolling_std_after = subset_after['Measure'].rolling(window=window_size).std().dropna()

    # Create DataFrames for before and after results
    results_bird_cal_before = pd.DataFrame({
        'Mean_Before': rolling_mean_before,
        'Std_Before': rolling_std_before
    })

    results_bird_cal_after = pd.DataFrame({
        'Mean_After': rolling_mean_after,
        'Std_After': rolling_std_after
    })

    # return results_bird_cal_before, results_bird_cal_after

    # Concatenate before and after values
    rolling_mean_combined = pd.concat([rolling_mean_before, rolling_mean_after], ignore_index=True)
    rolling_std_combined = pd.concat([rolling_std_before, rolling_std_after], ignore_index=True)

    # Create a DataFrame with combined rolling mean and std values
    results_bird_cal_combined = pd.DataFrame({
        'Mean': rolling_mean_combined,
        'Std': rolling_std_combined
    })

    # what are our valis
    min_mean_index = results_bird_cal_combined['Mean'].idxmin()
    min_std_index = results_bird_cal_combined['Std'].idxmin()

    mean_value_at_minSTD = results_bird_cal_combined['Mean'].iloc[min_std_index]
    std_value_at_minSTD = results_bird_cal_combined['Std'].iloc[min_std_index]

    mean_value_at_minMean = results_bird_cal_combined['Mean'].iloc[min_mean_index]
    std_value_at_minMean = results_bird_cal_combined['Std'].iloc[min_mean_index]

    if(min_mean_index != min_std_index):
        pass
        # do we want a flag to have a person do this?

    print(f"Mean at mean_min index: {min_mean_index} is: {mean_value_at_minMean}, STD: {round(std_value_at_minMean,1)}")
    print(f"Mean at mean_std index: {min_std_index} is: {mean_value_at_minSTD}, STD: {round(std_value_at_minSTD,1)}")
    print(f"------- Return value: {mean_value_at_minSTD}")

    return(mean_value_at_minSTD)


#############################
# Function Do_Auto_Bird: does automatic (no user input) calibration and bird weight data for traces in a MOM file
#   RAM 8/21/24
#
#   Builds a list of possible traces in the fiel up to a max value (like in autocalc, where used on first 3; use rest)
#   For all traces > 3 (calibrations), calculate value from start-end points
#
#   parameters
#       num_traces - how many traces will we evaluate?
#######
def do_Auto_Bird(bird_df, cal_gradient, cal_intercept, cal_r_squared, cal1_diff):
    global the_Burrow #make an argument?
    global my_SPS

    ### get list of all raised traces
    print("----- in auto bird    ----")
    baseline_cal_mean, sections_df = detect_sections(bird_df, 0.0002, 20, window_size=20, std_threshold=150000, tolerance=5000, lines = 0)

    ### get the actual bird baseline - above doesn't apply to lines overnight
    # baseline_cal_mean = do_Bird_Baseline()


    min_bird_mean = baseline_cal_mean + (cal1_diff * 0.9)  # leave room to show the first one
    
    ### get the bird baseline mean value and the bird mean value, but do it automatically
    ### for each detected section, do estimate and decide if we keep it, put into dataframe for return
    print("Here is sections_df")
    print(sections_df)

    ## now go thru each section after #3 and calculate weight

    # Columns for the results
    columns = ['Auto_Wt_1', 'Auto_Wt_2', 'Auto_Wt_3', 'Auto_Wt_4', 'Auto_Wt_5']
    for col in columns:
        sections_df[col] = None

        # Iterate through each section from #4 to the end
    for i, (start, stop) in enumerate(zip(sections_df['Start'][3:], sections_df['Stop'][3:]), start=4):
    # for i, (start, stop) in sections_df[['Start', 'Stop']].iterrows():
        results = do_Auto_Wts(bird_df, start, stop, cal1_diff, min_bird_mean) 
        for j, col in enumerate(columns):
            sections_df.at[i, col] = results[j]

    # Create new columns for the results - calc the calibrationi weights, too - change from range(4...) to range(1...)
    for i in range(1, len(sections_df) + 1):
        # Extract the 'Start' and 'Stop' values
        start = sections_df.loc[i, 'Start']
        stop = sections_df.loc[i, 'Stop']
        
    print(sections_df)

    print("remove non-bird estimates")
    # Remove rows where 'Auto_Wt_1' is less than 35
    sections_df = sections_df[sections_df['Auto_Wt_1'] >= 20]   
    print(sections_df)

    print("now add the datetime column")
    
    datetime_values = []
    
    # Iterate over each row in results DataFrame
    for index, row in sections_df.iterrows():
        start_idx = row['Start']
        
        # Check if the start_idx is within the bounds of bird_df
        if 0 <= start_idx < len(bird_df):
            # Retrieve the Datetime value at the Start index
            datetime_value = bird_df.iloc[start_idx]['Datetime']
        else:
            # If index is out of bounds, you can choose to handle it differently (e.g., set a default value or NaT)
            datetime_value = pd.NaT  # Set to Not a Time
        
        datetime_values.append(datetime_value)
    
    # Add the new column to results DataFrame
    sections_df['Datetime'] = datetime_values

    print(sections_df)

    if(FALSE):
        measure_start = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime.iloc[0]
        measure_end = bird_data_markers[bird_data_markers["Point"]=="End"].Datetime.iloc[0]

        # Calculate the baseline and regressed mass estimates for the birds
        bird_baseline_diff = abs(bird_data_mean - bird_cal_mean)
        bird_regression_mass = round(bird_baseline_diff * cal_gradient + cal_intercept, 2)

        if(TRUE):
            # Extract start and end indices from 'bird_data_markers' dataframe
            #start_index = bird_data_markers.loc[0, "Index"]
            #end_index = bird_data_markers.loc[1, "Index"]
            start_index = bird_data_markers.index[0]
            end_index = bird_data_markers.index[1]

            # Extract subset of 'Measure' series from 'my_DataFrame' dataframe
            measure_series = my_DataFrame.loc[start_index:end_index, "Measure"]
            # define W1
            W1 = bird_regression_mass
            # calc w2
            W2 = round(calc_W2_Penguin(measure_series, W1, cal_gradient, cal_intercept, my_SPS),2)

            # Calc W3
            # using do_Slope_0

            ### estimate a threshold that is at least a 25% of the other mean bird value over the background value
            est_threhold = round(((bird_data_mean - bird_cal_mean) *0.25) + bird_cal_mean,1)
        
            print(f'est threshold: {est_threhold}\tstart index: {start_index}\tend index: {end_index}')
            print(f'length of measure_series: {len(measure_series)}')

            if((end_index - start_index) < 100):
                mean_slope_0, slope_0_slope = do_Slope_0(my_DataFrame, start_index, end_index, 25, est_threhold, 0.7)
                newDiff = mean_slope_0 - bird_cal_mean
                W3 = round(newDiff * cal_gradient + cal_intercept, 2)
            else:       ### too long to go through all the windows, plus it is just fine with other methods
                W3 = W2

            ### do W4 - Mark's median
            if(True):
                Mark_median = measure_series.median()
                Mark_Diff = Mark_median - bird_cal_mean
                W4 = round(Mark_median * cal_gradient + cal_intercept, 2)

            ### do W5 - Liam's estimate
            if(True):
                liam_Mean = doSmoothWindowMean(my_DataFrame, start_index, end_index, bird_cal_mean)
                W5 = round(liam_Mean * cal_gradient + cal_intercept, 2)
            else:
                W5 = 999

            print("############## Span from new code: ")
            print(end_index - start_index)
            # print(f'W1: {W1} W2: {W2} W3: {W3} {W4}')
            print(f'Burr\tW1\tW2\tW3\tW4\tW5')
            print(f'{the_Burrow}\t{W1}\t{W2}\t{W3}\t{W4}\t{W5}')

        measure_start = bird_data_markers.index[0]
        measure_end = bird_data_markers.index[1]

        my_Points = measure_end - measure_start 
        my_Span = my_Points/my_SPS

        print("# Span from old code: ")
        print(my_Points)
        print("##################### Here is W1, then W2")
        print(W1)
        print(W2)

        if(confirm_continue("Good measurement?")):
            my_Eval = "Good"
        else:
            my_Eval = "Bad"

        data_DATE = my_DataFrame.Datetime.iloc[0]
        #data_DATE2 = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime[0]
        start_datetime = bird_data_markers.loc[bird_data_markers['Point'] == 'Start', 'Datetime'].iloc[0]


        print("###################### DEBUG bird_data_markers:")
        print(bird_data_markers)
        print("Start time: ")
        print(start_datetime)

        # Allow the user to input extra details for a "Notes" column
        bird_details = askstring('Bird', 'Enter brief details')
        if (bird_details == None): 
            bird_details = "N/A"

        # Add the info about this bird to the accumulating lists
        birds_datetime_starts.append(my_Points) # maybe replace this wih the Span in seconds
        birds_datetime_ends.append(round(my_Span,2)) # maybe replace this wih the Span in points
        birds_data_means.append(round(W2,2)) 
        birds_cal_means.append(round(bird_cal_mean,1))
        birds_baseline_diff.append(round(bird_baseline_diff,1))
        birds_regression_mass.append(bird_regression_mass)
        birds_details.append(the_Burrow + ": " + bird_details)

        # Show user info from the calculations
        t3.insert("1.0", "\tTime:       \t" + start_datetime + "\n") #3 add to Text widget
        t3.insert("1.0", "\tPoints (s): \t" + str(my_Points) + " (" + str(round(my_Span,1)) + "s)" + "\n") #2 add to Text widget
        t3.insert("1.0", "\tBird Mass:  \t" + str(bird_regression_mass) + " - " + my_Eval + "\n") #2 add to Text widget
        t3.insert("1.0", "Burr: " + the_Burrow + " File: " + user_BURROW + " - " + bird_details + "\n") # 1 add to Text widget



    # return pass
###############
#
#   do_Auto_Wts
#       start with do_Slope_0 to find the best span - calculate estimates
#           W1 = return that Mean/estimate - old W3
#           W2 = W1 adjusted for slope of it's estimate
#           W3 = Median (mark haussmann)
#           W4 = smoothie
#           W5 is if we want to make a regression
#       
########
def do_Auto_Wts(bird_df, start_index, stop_index, min_diff, cal_baseline):

    # Calc W3 using do_Slope_0 - use this to get the best start/stop within the trace for other (Median, W2, smoothie)

    ### estimate a threshold that is based on the passed value, which comes from the lowest calibration weight
    est_threhold = round(min_diff,1)

    # print(f'------length of measure_series: {len(measure_series)}')

    ## need to get a baseline around the focal point
    
    # cal_baseline = do_Bird_Baseline()
    cal_baseline = do_Bird_Baseline(bird_df, start_index, stop_index)
    bird_cal_mean = cal_baseline 

    print(f'--Baseline: {bird_cal_mean}')

    mytrunc = int((stop_index - start_index) * .1)  ## useGit as needed until we get Liam's PR locator method

    if((stop_index - start_index) < 500):
        mean_slope_0, slope_0_slope, slope_0_start, slope_0_len = do_Slope_0(bird_df, start_index, stop_index, 25, est_threhold, 0.7)
        print(f"mean_slope_0: {mean_slope_0}")
        newDiff = mean_slope_0 - bird_cal_mean
        W3 = round(newDiff * cal_gradient + cal_intercept, 2)
    else:       ### too long to go through all the windows, plus it is just fine with other methods
        W3 = 99.99  # W2  ## just get the mean value? do something other than all the figuring
        ### get the bird PR to do this...or, just go in 10% on each end
        mean_slope_0 = bird_df.loc[(start_index+mytrunc):(stop_index-mytrunc), 'Measure'].mean()
        newDiff = mean_slope_0 - bird_cal_mean
        W3 = round(newDiff * cal_gradient + cal_intercept, 2)
        slope_0_slope = 0  ### need to get slope of this, anyway

    # Calculate W2 using the adjusted slope for gravity
    # Define gravitational constant
    g = 9.81
    W2 = round(W3 * (1 + slope_0_slope / g),2)

    # median value - probably better with new_series from Liam's PR span or with trunc
    # new_series = bird_df.iloc[start_index:stop_index]
    Mark_median = bird_df.loc[(start_index+mytrunc):(stop_index-mytrunc), 'Measure'].median()
    Mark_Diff = Mark_median - bird_cal_mean
    W4 = round(Mark_Diff * cal_gradient + cal_intercept, 2)

    ### do W5 - Liam's estimate
    # liam_Mean = doSmoothWindowMean(bird_df, start_index, stop_index, bird_cal_mean) ######### not working aug 24, error line 821
    liam_Mean = 0
    W5 = round(liam_Mean * cal_gradient + cal_intercept, 2)

    W6 = 7  ## regression value of W2,3,4 or something else?

    return [W3, W2, W4, W5, W6]
    



#############################
# Function Do_Bird: get the calibration and bird weight data for a single bird in a MOM file
#    RAM 7/25/22
#    Update 1/28/2024 to use old calibration if wanted
#    parameters: NONE 
#    returns: NONE
#    RAM 6/1/24 - could change to receive a parameter "Bird Data Auto" that tells getTracePointPair it is no simple mean
#######
def Do_Bird(my_DataFrame, category):

        global the_Burrow
        global my_SPS

        bird_cal_mean, bird_cal_markers, bird_cal_good, bird_cal_axesLimits = getTracePointPair(my_DataFrame, "Calibration[Bird]")
        bird_data_mean, bird_data_markers, bird_data_good, bird_data_axesLimits = getTracePointPair(my_DataFrame, category, bird_cal_markers, bird_cal_axesLimits)

        measure_start = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime.iloc[0]
        measure_end = bird_data_markers[bird_data_markers["Point"]=="End"].Datetime.iloc[0]

        # Calculate the baseline and regressed mass estimates for the birds
        bird_baseline_diff = abs(bird_data_mean - bird_cal_mean)
        bird_regression_mass = round(bird_baseline_diff * cal_gradient + cal_intercept, 2)

        if(TRUE):
            # Extract start and end indices from 'bird_data_markers' dataframe
            #start_index = bird_data_markers.loc[0, "Index"]
            #end_index = bird_data_markers.loc[1, "Index"]
            start_index = bird_data_markers.index[0]
            end_index = bird_data_markers.index[1]

            # Extract subset of 'Measure' series from 'my_DataFrame' dataframe
            measure_series = my_DataFrame.loc[start_index:end_index, "Measure"]
            # define W1
            W1 = bird_regression_mass
            # calc w2
            W2 = round(calc_W2_Penguin(measure_series, W1, cal_gradient, cal_intercept, my_SPS),2)

            # Calc W3
            # using do_Slope_0

            ### estimate a threshold that is at least a 25% of the other mean bird value over the background value
            est_threhold = round(((bird_data_mean - bird_cal_mean) *0.25) + bird_cal_mean,1)
        
            print(f'est threshold: {est_threhold}\tstart index: {start_index}\tend index: {end_index}')
            print(f'length of measure_series: {len(measure_series)}')

            if((end_index - start_index) < 100):
                mean_slope_0, slope_0_slope = do_Slope_0(my_DataFrame, start_index, end_index, 25, est_threhold, 0.7)
                newDiff = mean_slope_0 - bird_cal_mean
                W3 = round(newDiff * cal_gradient + cal_intercept, 2)
            else:       ### too long to go through all the windows, plus it is just fine with other methods
                W3 = W2

            ### do W4 - Mark's median
            if(True):
                Mark_median = measure_series.median()
                Mark_Diff = Mark_median - bird_cal_mean
                W4 = round(Mark_median * cal_gradient + cal_intercept, 2)

            ### do W5 - Liam's estimate
            if(True):
                liam_Mean = doSmoothWindowMean(my_DataFrame, start_index, end_index, bird_cal_mean)
                W5 = round(liam_Mean * cal_gradient + cal_intercept, 2)
            else:
                W5 = 999

            print("############## Span from new code: ")
            print(end_index - start_index)
            # print(f'W1: {W1} W2: {W2} W3: {W3} {W4}')
            print(f'Burr\tW1\tW2\tW3\tW4\tW5')
            print(f'{the_Burrow}\t{W1}\t{W2}\t{W3}\t{W4}\t{W5}')

        measure_start = bird_data_markers.index[0]
        measure_end = bird_data_markers.index[1]

        my_Points = measure_end - measure_start 
        my_Span = my_Points/my_SPS

        print("# Span from old code: ")
        print(my_Points)
        print("##################### Here is W1, then W2")
        print(W1)
        print(W2)

        if(confirm_continue("Good measurement?")):
            my_Eval = "Good"
        else:
            my_Eval = "Bad"

        data_DATE = my_DataFrame.Datetime.iloc[0]
        #data_DATE2 = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime[0]
        start_datetime = bird_data_markers.loc[bird_data_markers['Point'] == 'Start', 'Datetime'].iloc[0]


        print("###################### DEBUG bird_data_markers:")
        print(bird_data_markers)
        print("Start time: ")
        print(start_datetime)

        # Allow the user to input extra details for a "Notes" column
        bird_details = askstring('Bird', 'Enter brief details')
        if (bird_details == None): 
            bird_details = "N/A"

        # Add the info about this bird to the accumulating lists
        birds_datetime_starts.append(my_Points) # maybe replace this wih the Span in seconds
        birds_datetime_ends.append(round(my_Span,2)) # maybe replace this wih the Span in points
        birds_data_means.append(round(W2,2)) 
        birds_cal_means.append(round(bird_cal_mean,1))
        birds_baseline_diff.append(round(bird_baseline_diff,1))
        birds_regression_mass.append(bird_regression_mass)
        birds_details.append(the_Burrow + ": " + bird_details)

        # Show user info from the calculations
        t3.insert("1.0", "\tTime:       \t" + start_datetime + "\n") #3 add to Text widget
        t3.insert("1.0", "\tPoints (s): \t" + str(my_Points) + " (" + str(round(my_Span,1)) + "s)" + "\n") #2 add to Text widget
        t3.insert("1.0", "\tBird Mass:  \t" + str(bird_regression_mass) + " - " + my_Eval + "\n") #2 add to Text widget
        t3.insert("1.0", "Burr: " + the_Burrow + " File: " + user_BURROW + " - " + bird_details + "\n") # 1 add to Text widget


#############################
# Function Do_Multiple_Birds: to id mltiple birds in one file
#    RAM 7/26/22
#    Parameters: dataframe
#    Returns: NONE
#
#    RAM 6/2/24
#    Add parameter for category to pass to GetPointPair
#    Allows us to designate what function to use for the calculation of mean
#    
#######
def Do_Multiple_Birds(my_DataFrame, category):
    global birds
    global the_Burrow

            # Allow the user to input the burrow this is from
    the_Burrow = askstring('Burrow', 'Enter burrow number')
    if (the_Burrow == None):  # (len(the_Burrow) == 0)|(
        the_Burrow = "N/A"
    if (len(the_Burrow) == 0):
        the_Burrow = "N/A"

    Set_Globals()  # reset the saved birds
    # assumes have lists declared as global
    # Allow the user to continue entering birds for as many times as she wants
    while (True):
        if(confirm_continue("Enter bird data?")):
            Do_Bird(my_DataFrame, category)
        else:
            break

    # Done entering bird data - Make the accumulated bird info into a clean dataframe for exporting
    birds = pd.DataFrame({"Burrow - Field": the_Burrow + " - " +user_BURROW,
                          "Date":data_DATE,
                          "Data_pts":birds_datetime_starts,
                          "Duration_secs":birds_datetime_ends,
                          "Regression_Mass_W2":birds_data_means,
                          "Mean_Calibration_Strain":birds_cal_means,
                          "Baseline_Difference":birds_baseline_diff,
                          "Regression_Mass_W1":birds_regression_mass,
                          "Details":birds_details})

    print("####### DEBUG: Bird entries complete ##########")

    if(confirm_continue("Export Results?")):
        Output_MOM_Data()

    return birds


################
# Function Output_MOM_Data to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: sends accumulated data to a csv file
#    Possible feature request: open window for user to add info to be saved with file
#######
def Output_MOM_Data():

    # Export summary info, including calibration info, to file
    data_name = user_BURROW.replace("DL", "AN")
    path_summary = "Burrow_{burrow}_{date}_SUMMARY.txt".format(burrow=user_BURROW, date=data_DATE)

    path_summary = filedialog.asksaveasfile(initialdir = my_Save_Dir, initialfile = data_name,  # changed from user_BURROW
                                    defaultextension= '.csv',
                                    filetypes=[
                                        ("Text file",".txt"),
                                        ("CSV file", ".csv"),
                                        ("All files", ".*"),
                                    ])

    # Export bird info (if any was added)
    path_bird = "Burrow_{burrow}_{date}_BIRDS.txt".format(burrow=user_BURROW, date=data_DATE)
    path_bird = path_summary

    if (len(birds_data_means) > 0):
        birds.to_csv(path_bird, index=False)
        mb.showinfo("Export Data", str(len(birds_data_means)) + " bird mass data saved")
        print("############### DEBUG: Wrote bird details to\n\t\t{bpath} ########".format(bpath=path_bird))
    else:
        mb.showinfo("Export Data", "No birds recorded.")
        print("############### DEBUG: No birds recorded ##########")


#########################################################################
#   App setup: Build the User Interface and run the app
#       June 8, 2024 - have 3 buttons
#           just explore files
#           do calculation with simple averages
#           do automated calculation
#########################################################################

##########################
#   MOM_defaults_from_file - Get user defined default values form external file
########
def MOM_defaults_from_file():
    # exec(open("MOM_GUI_set_user_values.py").read())  # feature request - make this accessible from within the app itself for changes
    #
    # SET Some user-specific data - temporary solution - don't use an external file for now so can make app
    #  This must be in same folder as the MOM_Processor_v03.py file
    #  Will eventually be replaced by a text file
    #

    # Calibration values, in grams
    global cal1_value
    global cal2_value
    global cal3_value

    ## SET THESE VALUES to match the standard weights you use in the field
    cal1_value = 15.97 	# 20 # replaced old values with weight of 3 nuts
    cal2_value = 32.59	# 40
    cal3_value = 50.22	# 60

    # Default figure view - don't change these
    global default_figure_width
    global default_figure_height

    # Default figure view - don't change these
    default_figure_width = 15
    default_figure_height = 10

    # for experimental automation process - don't change this
    global default_window

    default_window = 5  # points needed for the automated process - don't change this

    ## directories for saving, opening files
    global myDir
    global my_Save_Dir
    global my_Save_Real_Dir
    global datafile_folder_path
    global datafiles

    ### SET THESE VALUES to match your file structure
    myDir = "/Users/bobmauck/devel/LHSP_MOM_GUI/Programming/Code_python/Data_Files/Cut_With_Calib"

    default_window = 5

    my_Save_Dir = "/Users/bobmauck/devel/LHSP_MOM_GUI/Programming/output_files"  ### wehre to save the output files

    print("done setting user values")
    
    
    print("Done setting defaults: "+str(cal1_value)+", "+str(cal2_value)+", "+str(cal3_value))

###########################################
#   setup the user interface, first calling the globals
#######

### get defaults values from User defined external file "MOM_GUI_set_user_values.py" - in same folder with Main
MOM_defaults_from_file()

# declare global variables
Set_Globals()

# Define variables that deal with how big the screen should be
myWidth = 1200
myHeight = 1000

# Create the root window
root = tk.Tk()
root.geometry(f"{myWidth}x{myHeight}")
root.title(vAppName)

# Calculate the output frame width
output_width = myWidth - 2 * 10

###################
# Buttons FRAME for MOM operations
##

# Create the button frame with a 1-pixel border
buttonFrame = tk.Frame(root, width=myWidth-10, bd=1, relief=tk.SOLID)
buttonFrame.pack(pady=30)

## button ui variables
myButtonPadx = 5
myButtonPady = 5
myButtonWidth = 20
myButtonLabels = ["Browse Files", "Process Birds", "Cut Calc Mean", "Process Auto"]

# Create the buttons command=lambda num=i+1: mom_calc_button(num))
buttons = []

    #### buttons for browsing files, cutting files, or doing calculations
b1 = tk.Button(buttonFrame, text = myButtonLabels[0],command = lambda: mom_open_file_dialog("all"))
b1.pack(side=tk.LEFT, padx=myButtonPadx, pady=myButtonPady)
buttons.append(b1)

b2 = tk.Button(buttonFrame, text = myButtonLabels[1], command = lambda:mom_cut_button("Bird Data"))  
b2.pack(side=tk.LEFT, padx=myButtonPadx, pady=myButtonPady)
buttons.append(b2)

b4 = tk.Button(buttonFrame, text = myButtonLabels[3], command = lambda:mom_auto_button("Bird Data"))  
b4.pack(side=tk.LEFT, padx=myButtonPadx, pady=myButtonPady)
buttons.append(b4)


###################
# INPUT FRAME for calibration weight values
##

# important variables
myButtonPadX = 30
myButtonPadY = 5

# Create the input frame with a border
inputFrame = tk.Frame(root, width=myWidth-10, bd=1, relief=tk.SOLID)
inputFrame.pack()

# Create labels and entry widgets in a grid
my_entry_terms_02 = ["Calib. Light:", "Calib Mid:", "Calib Heavy:"]
my_entry_labels_02 = [str(cal1_value), str(cal2_value), str(cal3_value)]
my_entries2 = []
for i, label in enumerate(my_entry_labels_02):
    tk.Label(inputFrame, text=my_entry_terms_02[i]).grid(row=0, column=i, padx=myButtonPadX, pady=myButtonPadY)
    entry = tk.Entry(inputFrame)
    entry.grid(row=1, column=i, padx=5, pady=5)
    entry.insert(0, my_entry_labels_02[i])
    my_entries2.append(entry)

# Add vertical space
tk.Frame(root, height=30).pack()


###################
# INPUT FRAME for the automatic search for semi-stable data point stretches
##

# important variables
myButtonPadX = 30
myButtonPadY = 5

# Create the input frame with a border
inputFrame_AUTO = tk.Frame(root, width=myWidth-10, bd=1, relief=tk.SOLID)
inputFrame_AUTO.pack()

# Create labels and entry widgets in a grid
my_entry_terms_AUTO = ["Min Pts:", "Diff between Pts:", "STD DEV:"]
my_entry_labels_AUTO = [str(7), str(400), str(200)]   # could put these in external file like calibration wts
my_entries2_AUTO= []
for i, label in enumerate(my_entry_labels_AUTO):
    tk.Label(inputFrame_AUTO, text=my_entry_terms_AUTO[i]).grid(row=0, column=i, padx=myButtonPadX, pady=myButtonPadY)
    entry_AUTO = tk.Entry(inputFrame_AUTO)
    entry_AUTO.grid(row=1, column=i, padx=5, pady=5)
    entry_AUTO.insert(0, my_entry_labels_AUTO[i])
    my_entries2_AUTO.append(entry_AUTO)
    
# Add vertical space
tk.Frame(root, height=30).pack()

###################
# OUTPUT FRAMES to show data from files that we have opened
#   eventually we need scrollbars
####

# Variables for output frame text boxes
myOutputWidth = 50
myOutputHeight = 500
myTextPadX = 10
myTextPadY = 5

# Create the output frame with a border
outputFrame = tk.Frame(root, width=50, height=500, bd=1, relief=tk.SOLID)
outputFrame.pack(pady=5)

# Create the title label for the output frame
outputLabel = tk.Label(outputFrame, text="Summaries of MOM Traces", font=("Arial", 14))
outputLabel.pack(pady=5)

# Create a frame for each text widget
frame_t1 = tk.Frame(outputFrame)
frame_t1.pack(side=tk.LEFT, padx=myTextPadX, pady=myTextPadY, fill=tk.BOTH, expand=True)

frame_t2 = tk.Frame(outputFrame)
frame_t2.pack(side=tk.LEFT, padx=myTextPadX, pady=myTextPadY, fill=tk.BOTH, expand=True)

frame_t3 = tk.Frame(outputFrame)
frame_t3.pack(side=tk.LEFT, padx=myTextPadX, pady=myTextPadY, fill=tk.BOTH, expand=True)

# Create and configure vertical scrollbars for each frame
scrollbar_t1 = tk.Scrollbar(frame_t1, orient="vertical")
scrollbar_t1.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_t2 = tk.Scrollbar(frame_t2, orient="vertical")
scrollbar_t2.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_t3 = tk.Scrollbar(frame_t3, orient="vertical")
scrollbar_t3.pack(side=tk.RIGHT, fill=tk.Y)

# Create text widgets and place them in their respective frames
t1 = tk.Text(frame_t1, width=myOutputWidth, height=myOutputHeight, yscrollcommand=scrollbar_t1.set)
t1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_t1.config(command=t1.yview)

t2 = tk.Text(frame_t2, width=myOutputWidth, height=myOutputHeight, yscrollcommand=scrollbar_t2.set)
t2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_t2.config(command=t2.yview)

t3 = tk.Text(frame_t3, width=myOutputWidth, height=myOutputHeight, yscrollcommand=scrollbar_t3.set)
t3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_t3.config(command=t3.yview)

######################################
# All is ready, now run it
##
root.mainloop()

