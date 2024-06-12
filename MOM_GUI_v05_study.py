##############################
#
#    MOM_GUI_v05.py - RAM, June 2, 2024
#       Builds on v_04
#       Gives more than one option for how to average values in a trace
#       Changes:
#           05.01 - New file 
#           05.02 Pass parameter to mom_cut_button to designate which mehtod to use in calculating mean
#           Make third button the auto method button - this works
#           05.03 make the auto button use the values on screen as threshold and window size - works
#           05.04 on June 6, 2024 - cleaned up output data to show datapoints and time of window used
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
from tkinter import *

#### imports from LIam
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
# Function Set_Globals to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: run at startup, but not yet doing that, using definitions above only
#######
def Set_Globals():
    # general info about the fle
    global user_INPATH
    global user_BURROW
    global data_DATE

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

    global my_SPS ## set the SPS used for time calculation
    my_SPS = 80

    global my_std  # to keep track of the automation parameters
    global my_rolling_window
    global my_inclusion_threshold

    global my_Continue
    global vVersString
    global vAppName
    vVersString = " (v_05)"  ## upDATE AS NEEDED
    vAppName = "Mass-O-Matic Analyzer" + vVersString

    ### now make them
    birds_datetime_starts = []
    birds_datetime_ends = []
    birds_data_means = []
    birds_cal_means = []
    birds_baseline_diff = []
    birds_regression_mass = []
    birds_details = []


    aDefaults = []  # will be used when we transition to non-python user default settings

##########################
#
#   Get user defined default values form external file
#
########

    if(True):
        exec(open("MOM_GUI_set_user_values.py").read())  # feature request - make this accessible from within the app itself for changes
        print("Done setting defaults: "+str(cal1_value)+", "+str(cal2_value)+", "+str(cal3_value))
    else:
        read_defaults_from_file()  ## someday we will do this differently



############
# read_defaults_from_file() - assumes file Set_Defaults.txt is in same directory as this file
#######
def read_defaults_from_file():
    file_path = "Set_Defaults.txt"
    aDefaults = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                aDefaults.append(line)

    return aDefaults


##########################
#
#   Define some general utility function
#
########

############
# open_dialog
####
def open_dialog(myTitle, myInfo):
    mb.showinfo(myTitle, myInfo)




############
# confirm_continue: a utility function to get response via click
####
def confirm_continue(my_Question):
    MsgBox = mb.askquestion ('Confirm', my_Question)
    if MsgBox == 'yes':
        return True
    else:
        return False

#############
# return_useful_name: takes a path string and returns just the name of the file
####
def return_useful_name(the_path):
    where = the_path.rfind("/")
    the_name = the_path[(where + 1):(len(the_path)-4)]
    return the_name


###########################################
#   setup the user interface, first calling the globals
#######

# declare global variables
Set_Globals()

# Define variables
myWidth = 1200
myHeight = 1000

# Create the root window
root = tk.Tk()
root.geometry(f"{myWidth}x{myHeight}")
# root.title("Work with MOM datafile "+ vVersString)
root.title(vAppName)

# Calculate the output frame width
output_width = myWidth - 2 * 10


###################
# Button FRAME for calibration
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

### remove this for now because not helpful with simple operations but can add it later when we do updates
# # b3 = Button(buttonFrame, text = myButtonLabels[2], command = lambda:mom_calc_button(False))
# # b3.pack(side=tk.LEFT, padx=myButtonPadx, pady=myButtonPady)
# # buttons.append(b3) 
 
# b4 = Button(buttonFrame, text = myButtonLabels[3], command = lambda:mom_get_char_button()) 
#  change v_05.02. call to new mom_cut_button that passes a different parameter to multiple files for automatiion of mean
b4 = tk.Button(buttonFrame, text = myButtonLabels[3], command = lambda:mom_cut_button("Bird Data Auto"))  
b4.pack(side=tk.LEFT, padx=myButtonPadx, pady=myButtonPady)
buttons.append(b4)


###################
# INPUT FRAME for calibration
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
# INPUT FRAME for reading frame during the automatic search for semi-stable data point stretches
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
# Need to read these values. Not done in v04, just put them here, but don't use them in calculating restricted set
# maybe in generate_final_series, sent to do_PctChg_Bird_Calcs, 
    
# Add vertical space
tk.Frame(root, height=30).pack()

###################
# OUTPUT FRAME
#   eventually we need scrollbars
####

#### variables for output frame text boxes in fram
myOutputWidth = 50
myOutputHeight = 500
myTextPadX = 10
myTextPadY = 5

# Create the output frame with a border
outputFrame = tk.Frame(root, width=50, height=500, bd=1, relief=tk.SOLID)
outputFrame.pack(pady = 5) 

# Create the title label for the output frame
outputLabel = tk.Label(outputFrame, text="Summaries of MOM Traces", font=("Arial", 14))  # got rid of -- , "bold" -- 
outputLabel.pack(pady=5)

    # Create text columns in the output frame
t1 = tk.Text(outputFrame, width=myOutputWidth, height=myOutputHeight)
t1.pack(side=tk.LEFT, padx=myTextPadX, pady=myTextPadY)
t2 = tk.Text(outputFrame, width=myOutputWidth, height=myOutputHeight)
t2.pack(side=tk.LEFT, padx=myTextPadX, pady=myTextPadY)
t3 = tk.Text(outputFrame, width=myOutputWidth, height=myOutputHeight)
t3.pack(side=tk.LEFT, padx=myTextPadX, pady=myTextPadY)




#############################
#
#   Define the button functions to be used
#   
####


def mom_cut_button(my_Mean_Type):
    pass
    ## get a file to work with, then send it here...
    bird_fname, bird_df = mom_open_file_dialog("not") 
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
        Do_Multiple_Birds(bird_df, my_Mean_Type)
        

    

def mom_get_char_button():
    bird_fname, bird_df = mom_open_file_dialog("not") 
    user_BURROW = return_useful_name(bird_fname) 
    my_Continue = True
    my_Continue == Do_Multiple_Characteristics(bird_df)
    # print(my_Continue)
    # if(my_Continue):
    #     Do_Multiple_Characteristics(bird_df)



       
def mom_format_dataframe(mydf):
    my_cols = mydf.shape[1]
    ### if there are 3, the first one was an axis, get rid of it on not the copy, but the original (inplace = True)
    if(my_cols == 3):
        # NOTE: may need to format the Datetime column, but for now it is a string. Or test for type later 
        mydf.drop(mydf.columns[0], axis=1, inplace = True)
    ### now name the columns
    mydf.columns = ['Measure', 'Datetime']
    return mydf


def mom_open_file_dialog(to_show):
    global data_DATE
    my_testing = False # delete this on cleanup
    
    f_types = [('CSV files',"*.csv"), ('TXT',"*.txt") ]
    if(not(my_testing)):  # make false for testing
        f_name = filedialog.askopenfilename(initialdir = myDir,  title = "Choose MOM File", filetypes = f_types)
    else:
        f_name = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/185_One_Bird_70K_110K.TXT"

    dispName = f_name[(len(f_name)-60):len(f_name)]
    ## l1.config(text=dispName) # display the path ## FIX

    global user_BURROW # get the name in a format we can use
    user_BURROW = return_useful_name (f_name) # f_name[len(myDir):len(f_name)]
    
    #### change here by RAM, 9/3/2022 to revert to old way of getting the dataframe - from lhsp_mom_viewer
    if(FALSE):
        df = pd.read_csv(f_name, header=None, skiprows=1)
        df = mom_format_dataframe(df) #make sure we have 2 columns with proper names

        ## show info to user
        display_string = mom_get_file_info(df)
        # display_string = f_name[len(myDir):len(f_name)] + "\n" + display_string + "\n"  ### this doesn't work if not in MyDir!!
        display_string = return_useful_name (f_name) + "\n" + display_string + "\n"
        t1.insert("1.0", display_string)
        #### end of showing info to user 

        # set some globlas for later use
        # global user_BURROW
        
        # user_BURROW = return_useful_name (f_name) # f_name[len(myDir):len(f_name)]
        data_DATE = df.Datetime.iloc[-1] # .date()
        # data_DATE = df["Datetime"].iloc[-1].date()
    else:
        try:
            user_INPATH = f_name ## Get_File_Info()
        
            my_data = pd.read_csv(user_INPATH, header=None, names=["Measure", "Datetime"], 
                                encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", 
                                engine="python")
            my_data["Measure"] = pd.to_numeric(my_data["Measure"], errors="coerce")
            my_data["Datetime"] = pd.to_datetime(my_data["Datetime"], utc=True, errors="coerce")

            # Convert Unix timestamp to datetime
            my_data["Datetime"] = pd.to_datetime(my_data["Datetime"], unit='s', utc=True, errors="coerce")

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
                # data_DATE not being defined correctly
        data_DATE = my_data.Datetime.iloc[-1].date()
        # print("Working with data ending on: " # {date} in burrow.".format(date=data_DATE))
        # print(data_DATE)

        df = my_data # mom_format_dataframe(df) #make sure we have 2 columns with proper names

        ## show info to user
        display_string = mom_get_file_info(df)
        # display_string = f_name[len(myDir):len(f_name)] + "\n" + display_string + "\n"  ### this doesn't work if not in MyDir!!
        display_string = return_useful_name (f_name) + " - Summary Info" + "\n" + display_string + "\n"
        # t1.insert(tk.END, display_string)
        t1.insert("1.0", display_string)
        #### end of showing info to user 

    if (to_show == "cut"):
        the_start = int(my_entries[0].get())
        the_end = int(my_entries[1].get())
        df = df.iloc[the_start:the_end]

        # fig, ax = plt.subplots()
        # ax.plot(df.iloc[the_start:the_end])

        df.plot()
        plt.show()
    if (to_show == "all"):
        fig, ax = plt.subplots()
        ax.plot(df.loc[:,"Measure"])
        ## add_titlebox(ax, 'info here')
        ax.set_title(return_useful_name (f_name))
        ## df.plot() ## this was old way
        plt.show()
    if(to_show == "not"):
        pass

    return f_name, df



def mom_get_file_info(my_df):
    global my_SPS
    print(str(my_SPS) + "SPS" + "\n")
    #### show user info on the file chosen
    str1="\tPoints:" + str(my_df.shape[0])+ "\t\tColumns:"+str(my_df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
    str2="\tMinutes: " + str(round((my_df.shape[0])/my_SPS/60,2))+"\t"
    str3="(" + str(round((my_df.shape[0])/my_SPS/60/60,2))+" hours)\n"
    str4 = "\tMean Strain: " + str(round(my_df["Measure"].mean())) + "\n"

    return(str1 + str2 + str3 + str4)



def mom_do_birdplot (bird_df, bird_plot_df, my_filename, my_window, my_threshold, my_type):
    ##### need to improve - send to specific folder for these plots
    # Now make a plot of the whole thing wtih a line over the points we will use to calculate the bird
    fig, ax = plt.subplots()
    # Make mean line
    ax.hlines(y=bird_df["Measure"].mean(), xmin=bird_df.index[0], xmax=bird_df.index[-1], linewidth=2, color='r')
    # Plot the data
    # my_df["Measure"].plot(fig=fig)
    bird_plot_df["Measure"].plot(fig=fig)
    # save the plot, name assumes your original files starts wtih 3-letter burrow number
    my_label = my_entries2[0].get()
    output_filename = my_label + "_" + my_type + "_" + str(my_window) + "  win_" + str(my_threshold) + "_thr.png"
    plt.savefig(os.path.join("output_files",output_filename))
 




def mom_get_file_info(my_df):
    global my_SPS
    #### show user info on the file chosen
    str1="\tRows:" + str(my_df.shape[0])+ "\t\tColumns:"+str(my_df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
    str2="\tMinutes: " + str(round((my_df.shape[0])/my_SPS/60,2))+"\t"
    str3="(" + str(round((my_df.shape[0])/my_SPS/60/60,2))+" hours)\n"
    str4 = "\tMean Strain: " + str(round(my_df["Measure"].mean())) + "\n"

    return(str1 + str2 + str3 + str4)


###################
#  getTracPointPair:  A function to set a pair of draggle points on an interactive trace plot
# RETURNS
#       mean, markers, isGood
#       mean -- mean strain measurement value between two marked points
#       markers -- dataframe of marker information, including start and end index on the trace
#       isGood -- boolean confirmation that the plot wasn't closed before both points were marked
#       newAxesLimits -- bounding box limits for the plot view that was shown right before exiting 
def getTracePointPair(my_df, category, markers=None, axesLimits=None):

    data = my_df
    # Print a message
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
    # measures_series = pd.Series(measures, name='Measures Series')
    print("before doing means")
    
    if category == "Bird Data Auto":   #  "Bird Data":  # make this non-funtioning, should work as old

        # reduce the size of array to only qualifying points - automation - remove for now?
        measures_series = pd.Series(measures, name='Measures Series')

        mean = calc_Mean_Measure_Consec(measures_series, 400,7)  # change to get screen values
    else:
        mean = statistics.mean(measures)  # just calculate on whatever is in the array 'measures'

    print("after doing means")

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


###############################################################################################
#   Calculation methods
############################################################################################

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

    # still need to reduce the size of the array - as 4th series done in v01, but for now...

    # Display all four arrays for each line
    #for i in range(len(input_array)):
       # print(f"{input_array[i]:.2f} | {FWD_series[i]} | {BKWD_series[i]} | {sum_array[i]} | {Count_FWD_array[i]} | {Count_BKWD_array[i]} | {final_array[i]}")

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

    ## print these for debugging purposes
    print("################# read values from screen ##############")
    print(f"Threshold: {threshold}")
    print(f"Length: {myLen}")

    # Convert the Pandas Series to a NumPy array
    measures_array = mydf.values
   
    # Get a new list of values within threshold
    steady_points = generate_final_series(measures_array, threshold, myLen)

    # Create a new series composed of values in the fourth series that are > 0 - can I do this 
    filtered_final_series = steady_points[steady_points > 0]

    # Get the size of the array
    array_size = filtered_final_series.size

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

    # Standard Deviation (STD)
    std_value = np.std(filtered_final_series)

    print(f"Mean: {myMean}")
    print(f"Count: {count}")
    print(f"MeanDiff: {mean_Diff}")
    print(f"Range: {range_value}")
    print(f"Standard Deviation: {std_value}")

    ############ END of debugging print

    return myMean




#########################
#   annotateCurrentMarkers: A function to plot all markers from a markers dataframe on the current plt viewer
#   (to be used for the markers dataframe as returned by getTracePointPair)
########
def annotateCurrentMarkers(markers):
    ax = plt.gca() # this assumes a current figure object? Is this the only external assumption?

    # Plot the pairs of marker points separately, so lines aren't drawn betwen them
    for l, df in markers.groupby("Category"):
        ax.plot(df.loc[:,"Measure"], marker="o", color="black", ms=8)
        for index, row in df.iterrows():
            label = "{category} {point}".format(category=df.loc[index,"Category"], point=df.loc[index, "Point"])
            ax.annotate(label, (index, df.loc[index, "Measure"]), rotation=60)

################
# Function ƒ get the calibration for the MOM on this burrow-night
#    RAM 7/25/22
#    parameters: my_dataframe -> data to work with
#       - 
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

    print("cal1: "+str(cal1_value))
   

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

    print("Done cleanup of markers")

    # Get the linear regression information across the three calibration points
    cal_gradient, cal_intercept, cal_r_value, cal_p_value, cal_std_err = stats.linregress(calibrations["Value_Difference"], calibrations["Value_True"])
    cal_r_squared = cal_r_value**2

    print("Done getting linear regression")

    # A tiny function to confirm if we want to continue
    #   after showing the calibration plot results. Used just below.
    def continueKey(doit):
        if(doit == 'y'):
            good_to_go = True
        else:
            good_to_go = False


    # print("Showing calibration results.\nPress 'y' to proceed or 'n' to exit.")
    fig, ax = plt.subplots()
    # fig.canvas.mpl_connect('key_press_event', continueKey)

    ax.plot(calibrations["Value_Difference"], calibrations["Value_True"], marker="o", color="black", linestyle="None")
    ax.plot(calibrations["Value_Difference"], calibrations["Value_Difference"]*cal_gradient+cal_intercept, color="gray", linestyle="dashed")
    plt.xlabel("Measured value (strain difference from baseline)")
    plt.ylabel("True value (g)")
    plt.title("Calibration regression\n(R^2={r}, Inter={i}, Slope={s})".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5)))
    
    # #################
    # axes = plt.axes([0.81, 0.005, 0.1, 0.055])
    # bnext = Button(axes, 'Proceed')
    # bnext.on_clicked(continueKey("y"))

    # axes2 = plt.axes([0.1, 0.005, 0.1, 0.055])
    # bnext2 = Button(axes2, 'Stop') # , color="red")
    # bnext2.on_clicked(continueKey("n"))
    # #################
    
    print("before showing results")

    
    plt.show()
    ### show user in real time
    my_cal_result = "\tR^2={r}\n\tIntcpt={i}, Slope={s}".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5))
    t2.insert("1.0", my_cal_result + "\n") # add to Text widget
    t2.insert("1.0", "File: " + user_BURROW + " - Calibration regression:" + "\n") # add to Text widget
    

     # Check all the calibrations were marked successfully
    # if (not baseline_cal_Good or not cal1_Good or not cal2_Good or not cal3_Good or (cal_r_squared<0.9)):
    if (abs(cal_r_squared) < 0.9):
        print("bad r2")
        good_to_go = False

    return good_to_go

#############################
# Function Do_Bird: get the calibration and bird weight data for a single bird in a MOM file
#    RAM 7/25/22
#    Update 1/28/2024 to use old calibration if wanted
#    parameters: NONE 
#    returns: NONE
#    RAM 6/1/24 - could change to receive a parameter "Bird Data Auto" that tells getTracePointPair it is no simple mean
#######
def Do_Bird(my_DataFrame, category):

        # update the parameters for auto calcualtion if that is what we are using
        if category == "Bird Data Auto BAD": 
            global my_std
            global my_rolling_window
            global my_inclusion_threshold
   
            ## changd to float, does that cure iut?
            my_rolling_window = int(my_entries[3].get())
            my_inclusion_threshold = float(my_entries[2].get())


        bird_cal_mean, bird_cal_markers, bird_cal_good, bird_cal_axesLimits = getTracePointPair(my_DataFrame, "Calibration[Bird]")
       # bird_data_mean, bird_data_markers, bird_data_good, bird_data_axesLimits = getTracePointPair(my_DataFrame, "Bird Data", bird_cal_markers, bird_cal_axesLimits)
        bird_data_mean, bird_data_markers, bird_data_good, bird_data_axesLimits = getTracePointPair(my_DataFrame, category, bird_cal_markers, bird_cal_axesLimits)

        measure_start = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime.iloc[0]
        measure_end = bird_data_markers[bird_data_markers["Point"]=="End"].Datetime.iloc[0]

        # Calculate the baseline and regressed mass estimates for the birds
        bird_baseline_diff = abs(bird_data_mean - bird_cal_mean)
        bird_regression_mass = round(bird_baseline_diff * cal_gradient + cal_intercept, 2)

        # print("data_markers array:")

        print(bird_data_markers)
        print("")
        print("Columns in bird_data_markers: ", bird_data_markers.columns)
        print("")

        measure_start = bird_data_markers.index[0]
        measure_end = bird_data_markers.index[1]

        print("measure_start: " + str(measure_start))
        print("measure_end: " + str(measure_end))

        my_Points = measure_end - measure_start # bird_data_markers.Index[1] - bird_data_markers.Index[0]
        global my_SPS
        my_Span = my_Points/my_SPS

        if(confirm_continue("Good measurement?")):
            my_Eval = "Good"
        else:
            my_Eval = "Bad"

        # print(bird_regression_mass)

        # my_time = my_time + ", " + my_Eval

                # Allow the user to input extra details for a "Notes" column
        # bird_details = input("Enter any details about the bird:     ")
        bird_details = "None"

        # Add the info about this bird to the accumulating lists
        birds_datetime_starts.append(my_Points) # maybe replace this wih the Span in seconds
        birds_datetime_ends.append(my_Span) # maybe replace this wih the Span in points
        birds_data_means.append(bird_data_mean) 
        birds_cal_means.append(bird_cal_mean)
        birds_baseline_diff.append(bird_baseline_diff)
        birds_regression_mass.append(bird_regression_mass)
        birds_details.append(bird_details)

        print(bird_regression_mass)  ## reverse this
        # t3.insert("1.0", "\t" +str(bird_regression_mass) + "," + str(my_time) + "\n") #4 add to Text widget
        # t3.insert("1.0", "\tTime ON:\t" + str(my_time) + "\n") #3 add to Text widget
        # t3.insert("1.0", "\tPoints:\t" + str(myPoints) + "\n") #3 add to Text widget
        t3.insert("1.0", "\tPoints (s): \t" + str(my_Points) + " (" + str(round(my_Span,1)) + "s)" + "\n") #2 add to Text widget
        t3.insert("1.0", "\tBird Mass:  \t" + str(bird_regression_mass) + " - " + my_Eval + "\n") #2 add to Text widget
        t3.insert("1.0", "File: " + user_BURROW + " - Weight Calculation:" + "\n") # 1 add to Text widget

        t2.insert(1.0, "File: " + user_BURROW + "\n") # add to Text widget
        t2.insert(1.0, "\tBird Mass: \t" + str(bird_regression_mass) + "\n") # add to Text widget


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
    Set_Globals()  # reset the saved birds
    # assumes have lists declared as global
    # Allow the user to continue entering birds for as many times as she wants
    while (True):
        if(confirm_continue("Enter bird data?")):
            Do_Bird(my_DataFrame, category)
        else:
            break

    # Done entering bird data
    #   Make the accumulated bird info into a clean dataframe for exporting
    birds = pd.DataFrame({"Burrow":user_BURROW,
                          "Date":data_DATE,
                          "Data_pts":birds_datetime_starts,
                          "Duration_secs":birds_datetime_ends,
                          "Mean_Data_Strain":birds_data_means,
                          "Mean_Calibration_Strain":birds_cal_means,
                          "Baseline_Difference":birds_baseline_diff,
                          "Regression_Mass":birds_regression_mass,
                          "Details":birds_details})

    # # Convert the Datetime columns back to character strings for exporting
    # birds["Datetime_Measure_Start"] = birds["Datetime_Measure_Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # birds["Datetime_Measure_End"] = birds["Datetime_Measure_End"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print("Bird calculated masses: ")
    print(birds["Regression_Mass"])

    print("Bird entries complete.")

    if(confirm_continue("Export Results?")):
        Output_MOM_Data()

    return birds

#############################
# Function Do_Bird_Characteristics: get an In and Out quality for bird trip - to evaluate MOM performance
#    Choose start and stop when baseline begins/ends disturbance
#       Data to gather - start time, start point, points between markers
#           additional want: symmetrical, reliable
#    RAM 7/25/22
#    parameters: NONE 
#    returns: NONE
#######
def Do_Bird_Characteristics(my_DataFrame):

    bird_data_mean, bird_data_markers, bird_data_good, bird_data_axesLimits = getTracePointPair(my_DataFrame, "Duration")
    # measure_start = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime.iloc[0]
    # measure_end = bird_data_markers[bird_data_markers["Point"]=="End"].Datetime.iloc[0]

    measure_start = bird_data_markers.index[0]
    measure_end = bird_data_markers.index[1]

    print("measure_start: " + str(measure_start))
    print("measure_end: " + str(measure_end))

    my_Points = measure_end - measure_start # bird_data_markers.Index[1] - bird_data_markers.Index[0]
    global my_SPS
    my_Span = my_Points/my_SPS

    bird_details = "Duration"

    if(confirm_continue("Good measurement?")):
        my_Eval = "Good"
    else:
        my_Eval = "Bad"
    ### try this
    # my_time = str(measure_start.time()) + "," + str(measure_end.time())
    ### 
    # my_time = measure_start[-8:] + "," + measure_end[-8:] + "," + my_Eval

    t3.insert("1.0", "\t" + str(my_Points) + "\t" + str(my_Span) + "\n") # add to Text widget
    t3.insert("1.0", "File: " + user_BURROW + " - Duration:" + "\n") # add to Text widget
    


#############################
# Function Do_Multiple_Characteristics: to id mltiple birds in one file
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    
#######
def Do_Multiple_Characteristics(my_DataFrame):
    global birds
    Set_Globals()  # reset the saved birds
    # assumes have lists declared as global
    # Allow the user to continue entering birds for as many times as she wants
    while (True):
        if(confirm_continue("Enter bird data?")):
            Do_Bird_Characteristics(my_DataFrame)
        else:
            break


################
# Function Output_MOM_Data to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: sends accumulated data to a csv file
#######
def Output_MOM_Data():
    
    summaryDetails = "NONE" #feature request - open window for user to add info

    # Export summary info, including calibration info, to file
    path_summary = "Burrow_{burrow}_{date}_SUMMARY.txt".format(burrow=user_BURROW, date=data_DATE)

    #saveFilePath = fileDialog.asksaveasfile(mode='w', title="Save the file", defaultextension=".txt")

    path_summary = filedialog.asksaveasfile(initialdir = my_Save_Dir, initialfile = user_BURROW,
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
        mb.showinfo("Bird data saved")
        print("Wrote bird details to\n\t\t{bpath}".format(bpath=path_bird))
    else:
        mb.showinfo("No birds recorded.")
        print("No birds recorded.")




#############################
#   DraggableMarker:  A class for a set of draggable markers on a Matplotlib-plt line plot
#   Designed to record data from two separate markers, which the user confirms with an "enter key"
#   Adapted by Liam Taylor from https://stackoverflow.com/questions/43982250/draggable-markers-in-matplotlib
#############
class DraggableMarker():
    def __init__(self, category, startY, startX=0):
        self.isGood = False
        self.category = category

        self.index_start = 0
        self.index_end = 0

        self.buttonClassIndex = 0
        self.buttonClasses = ["{category} start".format(category=category), "{category} end".format(category=category)]

        self.ax = plt.gca()  # this assumes a current figure object? Is this the only external assumption?
        self.lines=self.ax.lines
        self.lines=self.lines[:]

        self.tx = [self.ax.text(0,0,"") for l in self.lines]
        self.marker = [self.ax.plot([startX],[startY], marker="o", color="red")[0]]

        self.draggable = False

        self.isZooming = False
        self.isPanning = False

        self.currX = 0
        self.currY = 0

        self.c0 = self.ax.figure.canvas.mpl_connect("key_press_event", self.key)
        self.c1 = self.ax.figure.canvas.mpl_connect("button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect("button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.drag)

    def click(self,event):
        if event.button==1 and not self.isPanning and not self.isZooming:
            #leftclick
            self.draggable=True
            self.update(event)
            [tx.set_visible(self.draggable) for tx in self.tx]
            [m.set_visible(self.draggable) for m in self.marker]
            self.ax.figure.canvas.draw_idle()        
                
    def drag(self, event):
        if self.draggable:
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def release(self,event):
        self.draggable=False
        
    def update(self, event):
        try:        
            line = self.lines[0]
            x,y = self.get_closest(line, event.xdata) 
            self.tx[0].set_position((x,y))
            self.tx[0].set_text(self.buttonClasses[self.buttonClassIndex])
            self.marker[0].set_data([x],[y])
            self.currX = x
            self.currY = y
        except TypeError:
            pass

    def get_closest(self,line, mx):
        x,y = line.get_data()
        try: 
            mini = np.argmin(np.abs(x-mx))
            return x[mini], y[mini]
        except TypeError:
            pass

    ##############
    # Not sure we even want these - would rather deal with buttons on the window
    #####
    def key(self,event):
        if (event.key == 'o'):
            self.isZooming = not self.isZooming
            self.isPanning = False
        elif(event.key == 'p'):
            self.isPanning = not self.isPanning
            self.isZooming = False
        elif(event.key == 't'):
            # A custom re-zoom, now that 'r' goes to 
            # the opening view (which might be retained from a previous view)
            line = self.lines[0]
            full_xstart = min(line.get_xdata())
            full_xend = max(line.get_xdata())
            full_ystart = min(line.get_ydata())
            full_yend = max(line.get_ydata())
            self.ax.axis(xmin=full_xstart, xmax=full_xend, ymin=full_ystart, ymax=full_yend)
        elif (event.key == 'enter'):  #### these are the event keys we need for now
            if(self.buttonClassIndex==0):
                self.ax.plot([self.currX],[self.currY], marker="o", color="yellow")
                self.buttonClassIndex=1
                self.index_start = self.currX
                plt.title("Add {category} end point, then press enter.".format(category=self.category))
            elif(self.buttonClassIndex==1):
                self.index_end = self.currX
                self.isGood = True
                plt.close()
            self.update(event)

###################
#   AxesLimits: A class defining an object that stores axes limits for
#       pyplot displays
#######
class AxesLimits():
    def __init__(self, xstart, xend, ystart, yend):
        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend




root.mainloop()

