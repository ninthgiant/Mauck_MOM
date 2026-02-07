#######################################
#######################################
#    MOM_Globals.py 
#       Created by Bob Mauck on 2026-02-07.
#       This file holds global variables and constants used across the Mass-O-Matic Analyzer application.
#######################################
#######################################

##############
#   ---Constants here
##############
Current_OS = "MacOS"   # Windows, MacOS, Linux - shouldn't actually need to be used if just for Windows and MacOS

############## values Now in Globals     
VERSION = "v12(2024-12-07)"
VERSION = "v13(2026-02-04)"  # updated output formats onscreen and autobatch output file, also adjusted default window size
VERSION = "v14(2026-02-06)"  # Cleand up Batch code and added a review button to review batch summaries, also added a trim button to trim the data files for better viewing and processing
VERSION = "v15(2026-02-07)"  # Cleaned output from Batch operations, more failure info, timer for elasped time to run the batch ops
VERSION = "v16(2026-02-07)"  # After Code Review, Tk thread protection, Data Race in Calibration, Files processed counter, UI conisistncy, removed dead code, did not fix baseline calib issue. will do next versoin
VERSION = "v17(2026-02-07)"  # added MOM_Globals.py to hold global variables and constants, cleaned up imports and code in other files to use this, added setup_gui() function to MOM_GUI.py to handle GUI setup and parameters, added more comments and documentation, updated version number and change log here in MOM_Globals.py

vAppName = "Mass-O-Matic Analyzer " + VERSION


############## 
#  hold globals variables here
##############

#########################
# globals used in more than one place. Could replace with calls to functions, but easier this way for now.
#########################
do_print = False  # Set to True to enable debug printing. each module responds to this

# Default monospace font for on-screen tables
screen_font = "courier"
screen_font_size = 16

#####################################
#    Global Variables used in various files
#####################################

##############
#   --- Colony level information - change as needed for different data sets --- but not yet used
######
Time_Zone_MOM = "EST" # AST is what RFID are usually on


##############
#   --- Calibration weights to be used as Default calib values (self.cal1_true...) in line 36+ of MOM_Calculations
#   --- support for multiple years. add values for more years as needed. these are used in the GUI to populate the dropdown menu for calibration year, and to set the default calibration values when a year is selected. also used in MOM_Calculations to set the default calibration values when the program starts.  
######
CALIBRATION_BY_YEAR = {
    2024: (15.97, 32.59, 50.22),
    2025: (32.59, 50.22, 100.00),
    2026: (30.00, 50.00, 70.00),
}

calib_Year = 2025. #default year

def get_calibration_year_options():
    return sorted(CALIBRATION_BY_YEAR.keys())

def get_calibration_values_for_year(year):
    selected_year = int(year)
    if selected_year not in CALIBRATION_BY_YEAR:
        selected_year = get_calibration_year_options()[0]
    return CALIBRATION_BY_YEAR[selected_year]

def set_calibration_year(year):
    global calib_Year, calibLow, calibMed, calibHi
    calib_Year = int(year)
    calibLow, calibMed, calibHi = get_calibration_values_for_year(calib_Year)
    return calibLow, calibMed, calibHi

calibLow, calibMed, calibHi = set_calibration_year(calib_Year)


##############
#   --- varialbes defined in MOM_Processing
######
max_length_secs = 20  # maximum length of time (seconds) to allow in automatic processing mode - make this a user preference later
sampling_rate = 60
max_length_auto = sampling_rate *  max_length_secs # maximum number of data points to allow in automatic processing mode - auto_one_file()
r2_threshold_auto = 0.9999  # minimum R^2 value to allow in automatic processing mode - auto_one_file() - was 0.99999, relaxed to 0.9999 to allow more files to be processed, but can adjust as needed based on results and user preference
