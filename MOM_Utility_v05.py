##############################
#
#    MOM_GUI_v05_Utilities.py - RAM, June 27, 2024
#       contains utility functions for main app
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




##########################
#
#   Define some general utility functions
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



############
# get_user_input a utility function to get response via click
####
def get_user_input(my_header, my_Question):
    MsgBox = askstring(my_header, my_Question)
    return MsgBox


#############
# return_useful_name: takes a path string and returns just the name of the file
####
def return_useful_name(the_path):
    where = the_path.rfind("/")
    the_name = the_path[(where + 1):(len(the_path)-4)]
    return the_name


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
#   Convert_UNIX_Time - from ChatGPT
##########
def Convert_UNIX_Time(unix_timestamp):
    """
    Converts a Unix timestamp to a human-readable date and time using pandas.

    Parameters:
    unix_timestamp (int or float): The Unix timestamp to convert.

    Returns:
    str: The human-readable date and time as a string.
    """
    # Convert the Unix timestamp to a pandas Timestamp object
    timestamp = pd.to_datetime(unix_timestamp, unit='s')

    # Format the Timestamp object to a string
    formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_time



##########################
#   Convert_UNIX_Time_Series - from ChatGPT - not used right now - simple code in Main
##########

def Convert_UNIX_Time_Series(unix_timestamps):
    """
    Converts a series of Unix timestamps to human-readable dates and times using pandas.

    Parameters:
    unix_timestamps (list or pd.Series): A list or pandas Series of Unix timestamps to convert.

    Returns:
    pd.Series: A pandas Series of human-readable date and time strings.
    """
    # Convert the Unix timestamps to pandas Timestamps
    timestamps = pd.to_datetime(unix_timestamps, unit='s')

    # Format the Timestamps to strings
    formatted_times = timestamps.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_times

    # Example usage
    # unix_timestamps = [1651859200, 1651945600]  # Replace with your Unix timestamps
    # human_readable_times = convert_unix_series_to_timestamps(unix_timestamps)
    # print(human_readable_times)


##########################
#   Convert_UNIX_Time_Series - not used right now but may be later
##########
def mom_format_dataframe(mydf):
    my_cols = mydf.shape[1]
    ### if there are 3, the first one was an axis, get rid of it on not the copy, but the original (inplace = True)
    if(my_cols == 3):
        # NOTE: may need to format the Datetime column, but for now it is a string. Or test for type later 
        mydf.drop(mydf.columns[0], axis=1, inplace = True)
    ### now name the columns
    mydf.columns = ['Measure', 'Datetime']
    return mydf

    # A tiny function to confirm if we want to continue
    #   after showing the calibration plot results. Used just below.

##########################
#   continueKey - not used right now but may be later
##########
def continueKey(doit):
    if(doit == 'y'):
        good_to_go = True
    else:
        good_to_go = False


