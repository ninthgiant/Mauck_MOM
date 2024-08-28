#######################################
#######################################
#    MOM_Processing.py 
#       Loadcell trace parsing, viewing, and processing
#       R.A.M and L.U.T.
#       2024-08-27 cleanup of RAM_v10
#######################################
#######################################

#######################################
#######################################
# Imports and libraries
#######################################
#######################################

import os
import warnings
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import MOM_Calculations

#######################################
#######################################
# I/O Parameters
#######################################
#######################################
input_directory = os.getcwd()
output_directory = os.getcwd()

PLOT_VIEWER_WIDTH = 10
PLOT_VIEWER_HEIGHT = 5

#######################################
#######################################
# File utilities
#######################################
#######################################

#######
# Function get_user_file
#   Asks user for select input measurement trace file
# Parameters: None
# Returns:
#   User-selected file path (str)
#######
def get_user_file():
    # Can update the global input_directory to call back later
    global input_directory

    # Default filetype fire
    files_types = [('TXT',"*.txt *.TXT"), 
                   ('CSV',"*.csv *.CSV")]    
    
    # Ask user to select file
    f_path = filedialog.askopenfilename(initialdir=input_directory,  title="Choose trace file", filetypes=files_types)

    # Catch no file selected
    if len(f_path) == 0:
        messagebox.showinfo("Error", "No file chosen. Try again.")
        return None, None

    # Update global input directory to call back later
    input_directory = os.path.dirname(f_path)

    return f_path

#######
# Function parse_trace_file
#   Wrangle and clean measurement strain data file 
#   Including some mangled data error catching
# Parameters:
#   path - file path to target load cell data (str)
# Returns:
#   Formatted dataframe (pandas Dataframe of form Measurement | Datetime)
#######
def parse_trace_file(path):
    # Try to parse file
    try:
        # Read original CSV file
        dat = pd.read_csv(path, header=None, names=["Measure", "Datetime"], 
                          encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", 
                          engine="python")
        
        # Convert measurement column to numeric strain values
        dat["Measure"] = pd.to_numeric(dat["Measure"], errors="coerce")            

        # Convert Unix timestamp to datetime
        dat["Datetime"] = pd.to_datetime(dat["Datetime"], unit='s', utc=False, errors="coerce")
        dat["Datetime"] = dat["Datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')

        # We've possibly forced parsing of some malformed data
        #   ("replace" utf-8 encoding errors in read_csv(), "coerce" datetime errors in to_numeric() and to_datetime(), 
        # Now we need to clean that up.
        # Simply drop all points from the file where Measure has been coerced to NaN
        #   and where Datetime has been coerced to NaT
        dat = dat[~dat.Measure.isnull() & ~dat.Datetime.isnull()]
        
        return dat
    
    # Common errors
    except FileNotFoundError:
        warnings.warn("\nInput file not found. FileNotFoundError")
    except pd.errors.EmptyDataError:
        warnings.warn("\nError parsing empty file. EmptyDataError")
    except pd.errors.ParserError:
        warnings.warn("\nError parsing input file. ParserError")
    except Exception as e:
        warnings.warn("\nError parsing input file. {}".format(e))

#######################################
#######################################
# Output formatting and writing
#######################################
#######################################

#######
# Function output_header
#   Write process function header file to GUI text frame
# Parameters:
#   dat               - full dataframe (pandas Dataframe of form Measurement | Datetime)
#   f_name            - path to measurement file (str)
#   header_label      - label to print bold, should be the processing type, e.g., "VIEW", "MANUAL PROCESSING" (str)
#   output_frame_text - tkinter output text widget frame for writing (tkinter.Text)
# Returns: None
#######
def output_header(dat, f_name, header_label, output_frame_text):
    # Configure the bold / regular fonts to stay consistent across machines
    output_frame_text.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))
    output_frame_text.tag_configure("regular", font=("TkDefaultFont", 10))
    # Set frame in writable state
    output_frame_text.configure(state="normal")
    # Write bold header label (no linebreak after)
    output_frame_text.insert("end", "\n" + header_label + " ", "bold")
    # Write formatted data/file summary
    output_string = "file {fname}\n\t{length} samples, from {start_time} to {end_time}\n".format(fname=f_name, 
                                                                                                 length=len(dat), 
                                                                                                 start_time=dat.loc[0, "Datetime"], 
                                                                                                 end_time=dat.loc[len(dat)-1, "Datetime"])
    output_frame_text.insert("end", output_string, "regular")
    # Set frame back to read-only state
    output_frame_text.configure(state="disabled")

#######
# Function output_header
#   Write an error message to GUI text frame
# Parameters:
#   e                 - error text (str)
#   output_frame_text - tkinter output text widget frame for writing (tkinter.Text)   
# Returns: None
def output_error(e, output_frame_text):
    # Configure the regular font to stay consistent across machines
    output_frame_text.tag_configure("regular", font=("TkDefaultFont", 10))
    # Set frame to writable state
    output_frame_text.configure(state="normal")
    # Write the error message (tab-indented)
    output_frame_text.insert("end", "\t" + e + "\n", "regular")
    # Set frame back to read-only state
    output_frame_text.configure(state="disabled")

#######
# Function output_calibration
#   Write initialized calibration information to GUI text frame
#   NOTE does not check if calibration is initialized, this should happen before calling
# Parameters:
#   calibration       - initialized calibration object (MOM_Calculations.Calibration)
#   output_frame_text - tkinter output text widget frame for writing (tkinter.Text)
# Returns: None
def output_calibration(calibration, output_frame_text):
    # Configure the regular font to stay consistent across machines
    output_frame_text.tag_configure("regular", font=("TkDefaultFont", 10))
    # Set frame to writable state
    output_frame_text.configure(state="normal")
    # Write formatted calibration regression information
    output_string = "\tCalibration.\tBaseline = {b}\tR^2 = {r}\tIntercept = {i}\tSlope = {s}\n".format(b=int(calibration.baseline),
                                                                                                       r=round(calibration.regression_rsquared,5),
                                                                                                       i=round(calibration.regression_intercept,5),
                                                                                                       s=round(calibration.regression_gradient,5))
    output_frame_text.insert("end", output_string, "regular")
    # Set frame back to read-only state
    output_frame_text.configure(state="disabled")

#######
# Function output_weights
#   Format CSV line for multiple weight measurement values
#   with OPTION to write to GUI
# NOTE rounding only happens here!
# Parameters:
#   f_name                 - measurement data file name (str)
#   counter                - number/id marking the relative location of the trace segment within the overall file (MOM_Calculations.Calibration)
#   datetime               - datetime of measurement (str)
#   start_index            - left index of the trace segment (int)
#   end_index              - right index of the trace segment (int)
#   window_start_index     - left index of the best (min) slope window within the trace segment, probably from MOM_Calculations.w_windowed_min_slope (int)
#   window_end_index       - right index of the best (min) slope window within the trace segment, probably from MOM_Calculations.w_windowed_min_slope (int)
#   slope                  - slope of points within the trace segment (float)
#   min_slope              - best (min) slope as found as a window within the trace segment, probably from MOM_Calculations.w_windowed_min_slope (float)
#   output_frame_text      - tkinter output text widget frame for writing (tkinter.Text)
#   include_header         - do you want a CSV header line printed before the data line? (bool, default False)
#   write_output_to_screen - do you want to write the output string to the GUI text widget? (bool, default False)
# Returns: 
#   Formatted CSV line (str)
#######
def output_weights(f_name, counter, datetime, 
                   start_index, end_index, 
                   window_start_index, window_end_index,
                   weight_mean, weight_median, weight_min_slope,
                   slope, min_slope,
                   output_frame_text, 
                   include_header=False, write_output_to_screen=True):
    output_string = ""

    # Add CSV header line before data line, if requested  
    if include_header:
        output_string = "\tFile,Trace_Segment_Num,Datetime,Samples,Sample_Min_Slope,Weight_Mean,Weight_Median,Weight_Min_Slope,Slope,Min_Slope\n"

    # Format data for CSV output
    # NOTE header line appended just before string here, if it's been added to output_string already
    output_string = output_string + "\t{fname},{counter},{dtime},{samples},{samplesMinSlope},{wMean},{wMedian},{wMinSlope},{slope},{minSlope}\n".format(fname=f_name, 
                                                                                                                                                        counter=counter,
                                                                                                                                                        dtime=datetime,
                                                                                                                                                        samples=(end_index-start_index+1),
                                                                                                                                                        samplesMinSlope=(window_end_index-window_start_index+1),
                                                                                                                                                        wMean=round(weight_mean, 2),
                                                                                                                                                        wMedian=round(weight_median,2),
                                                                                                                                                        wMinSlope=round(weight_min_slope,2),
                                                                                                                                                        slope=round(slope,6),
                                                                                                                                                        minSlope=round(min_slope,6))
    # If requested, write to GUI screen
    if write_output_to_screen:
        # Configure the regular font to stay consistent across machines    
        output_frame_text.tag_configure("regular", font=("TkDefaultFont", 10))
        # Set frame to writable state
        output_frame_text.configure(state="normal")
        # Write formatted CSV data (and/or header line, if it was added to the string)
        output_frame_text.insert("end", output_string, "regular")
        # Set frame back to read-only state
        output_frame_text.configure(state="disabled")

    # Return the formatted string, even if we did not write to GUI
    return output_string

#######################################
#######################################
# Interaction utilities 
#######################################
#######################################

#######
# Class DraggableMarkerPair
#   Draggable markers for selecting a pair of points from plots
#   User confirms selections with enter key
#   Can use is_good to confirm if both points were selected properly
# Adapted from https://stackoverflow.com/questions/43982250/draggable-markers-in-matplotlib
#######
class DraggableMarkerPair():
    def __init__(self, category, start_y, start_x=0):
        self.category = category
        self.start = 0
        self.end = 0
        self.is_good = False

        self.button_class_index = 0
        self.button_classes = ["{category} start".format(category=category), "{category} end".format(category=category)]

        self.ax = plt.gca() 
        self.lines=self.ax.lines[:]
        self.tx = [self.ax.text(0,0,"") for l in self.lines]
        self.marker = [self.ax.plot([start_x],[start_y], marker="o", color="red")[0]]

        self.draggable = False
        self.zooming = False
        self.panning = False

        self.curr_x = 0.0
        self.curr_y = 0.0

        self.c0 = self.ax.figure.canvas.mpl_connect("key_press_event", self.key)
        self.c1 = self.ax.figure.canvas.mpl_connect("button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect("button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.drag)

    def click(self, event):
        if event.button == 1 and not self.panning and not self.zooming:
            self.draggable = True
            self.update(event)
            [tx.set_visible(self.draggable) for tx in self.tx]
            [m.set_visible(self.draggable) for m in self.marker]
            self.ax.figure.canvas.draw_idle()        
                
    def drag(self, event):
        if self.draggable:
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def release(self, event):
        self.draggable = False
        
    def update(self, event):
        try:        
            line = self.lines[0]
            x, y = self.get_closest(line, event.xdata) 
            self.tx[0].set_position((x,y))
            self.tx[0].set_text(self.button_classes[self.button_class_index])
            self.marker[0].set_data([x],[y])
            self.curr_x = x
            self.curr_y = y
        except TypeError:
            pass

    def get_closest(self, line, mx):
        x, y = line.get_data()
        try: 
            mini = np.argmin(np.abs(x-mx))
            return x[mini], y[mini]
        except TypeError:
            pass

    def key(self,event):
        if event.key == 'enter':  
            if self.button_class_index == 0:
                self.ax.plot([self.curr_x],[self.curr_y], marker="o", color="yellow")
                self.button_class_index = 1
                self.start = self.curr_x
                plt.title("Add {category} end point, then press enter.".format(category=self.category))

            elif self.button_class_index == 1:
                self.end = self.curr_x
                self.is_good = True
                plt.close()

            self.update(event)

#######
# Class AxesLimits
#   Stores x- and y- axes limits for re-opening pyplot displays in the same position
#######
class AxesLimits():
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

#######
# Function annotate_current_markers
#   Adds a set of user marker points to existing (or new) axes
# Parameters:
#   markers         - a list of DraggableMarkerPair markers
# Returns: None
#######
def annote_current_markers(markers):
    # Get axes (or create new ones)
    ax = plt.gca()

    # Plot the pairs of marker points
    for _, d in markers.groupby("Category"):
        ax.plot(d.loc[:,"Measure"], marker="o", color="black", ms=8)
        for index, _ in d.iterrows():
            label = "{category} {point}".format(category=d.loc[index,"Category"], point=d.loc[index, "Point"])
            ax.annotate(label, (index, d.loc[index, "Measure"]), rotation=60)
            
#######
# Function get_trace_point_pair
#   Ask the user to select two points on an interactive pyplot window
# Parameters:
#   dat                    - full dataframe (pandas Dataframe of form Measurement | Datetime)
#   category               - a category label for the marker, probably like "Calibration" or "Bird" (str)
#   markers     (optional) - prexisting markers to add to the plot (optional, list of DraggableMarkerPairs)
#   axes_limits (optional) - prexisting axes limits to re-open plot in previous position
# Returns:
#   Confirmation if user selected two points, helps catch errors like user exiting plot window (bool)
#   Mean measurement value between the marked points (float)
#   Markers added by user (list of DraggableMarkerPair)
#   Axes limits from when the plot closed (MOM_Processing.AxesLimits) 
#######
def get_trace_point_pair(dat, category, markers=None, axes_limits=None):

    # Boot up interactive plot
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches((PLOT_VIEWER_WIDTH, PLOT_VIEWER_HEIGHT))
    ax.plot(dat.loc[:,"Measure"])

    # Restore previous axes limits, if available
    if (axes_limits is not None):
        ax.set_xlim(left=axes_limits.x_start, right=axes_limits.x_end)
        ax.set_ylim(bottom=axes_limits.y_start, top=axes_limits.y_end)

    # Add previous markers, if available
    if (markers is not None):
        annote_current_markers(markers)

    # Initialize the draggable markers
    dm = DraggableMarkerPair(category=category, start_y=min(dat["Measure"]))
    ax.set_title("Add {category} start point, then press enter.".format(category=category))
    plt.show(block=True)

    plt.ioff()

    # Catch common draggable marker errors
    # First, if the pair was not initialzied correctly (e.g., user exited window)
    # Second, if the pair marks the same index (e.g., user double clicked enter)
    if not dm.is_good or dm.start == dm.end:
        return False, None, None, None

    # Extract the selected indices
    start_index = min(dm.start, dm.end)
    end_index = max(dm.start, dm.end)

    # Extract the axes limits from interactive viewer to restore limits later
    end_view_x_start, end_view_x_end = ax.get_xlim()
    end_view_y_start, end_view_y_end = ax.get_ylim()
    new_axes_limits = AxesLimits(end_view_x_start, end_view_x_end, end_view_y_start, end_view_y_end)

    # Calculate mean measure
    measures = dat.loc[start_index:end_index, "Measure"]
    mean_measure = measures.mean()

    # Calculate datetime range
    start_time = dat.loc[start_index, "Datetime"]
    end_time = dat.loc[end_index, "Datetime"]

    # Package up marked point information
    markers = pd.DataFrame({"Category":category,
                            "Point":["Start", "End"],
                            "Index":[start_index , end_index],
                            "Datetime":[start_time, end_time],
                            "Measure":[dat.loc[start_index ,"Measure"], dat.loc[end_index,"Measure"]]})
    markers = markers.set_index("Index")

    return True, mean_measure, markers, new_axes_limits

#######################################
#######################################
# Core processing functions
#######################################
#######################################

#######
# Function run_user_calibration
#   Perform user calibration with interactive plot of load cell trace 
# NOTE this will return an UNINITIALIZED calibration object if the user input failed!
#      you have to check for this outside the function
# Parameters:
#   dat                             - full dataframe (pandas Dataframe of form Measurement | Datetime)
#   calibration                     - weight calibration object (MOM_Calculations.Calibration)
#   calibration_user_entered_values - tkinter text input frames to update initial calibration values (list of tkinter.Entry)
#   output_frame_text               - tkinter output text widget frame for writing (tkinter.Text)
# Returns:
#   calibration object, uninitialized or initialized (MOM_Calculations.Calibration) or None, if user calibration failed 
#######
def run_user_calibration(dat, calibration, calibration_user_entered_values, output_frame_text):
    # If the calibration has not been initialized OR the user declines the opportunity to use the previous one, take it
    # NOTE python short-circuits the OR operator, so if the calibration has not been initialized, we will automatically take the calibration
    if not calibration.initialized or not messagebox.askyesno("Confirmation", "Use previous calibration?"):
        # Set manual calibration values from entry frames
        # checking to see that the values are well-formed floats (not e.g., text)
        if not calibration.set_true(*[entry.get() for entry in calibration_user_entered_values]):
            output_error("ERROR invalid calibration input value", output_frame_text)
            # We return without an initialized calibration object if this failed
            return

        # Conduct calibrations
        calibration_true_values = calibration.get_true()

        # Get baseline measurement from user
        # After each measurement, we check that BOTH points are selected, and kick out of the function if not
        good, baseline_cal_mean, baseline_cal_markers, axes_limits = get_trace_point_pair(dat, "Cal[Baseline]")
        if not good:
            output_error("ERROR user marker error during calibration", output_frame_text)
            return
        markers = baseline_cal_markers
        
        # Get three calibration measurements from user
        good, cal1_mean, cal1_markers, axes_limits = get_trace_point_pair(dat, "Cal1[{}]".format(calibration_true_values[0]), markers, axes_limits)
        if not good:
            output_error("ERROR user marker error during calibration", output_frame_text)
            return
        markers = pd.concat([markers, cal1_markers])

        good, cal2_mean, cal2_markers, axes_limits = get_trace_point_pair(dat, "Cal2[{}]".format(calibration_true_values[1]), markers, axes_limits)
        if not good:
            output_error("ERROR user marker error during calibration", output_frame_text)
            return
        markers = pd.concat([markers, cal2_markers])
        
        good, cal3_mean, cal3_markers, axes_limits = get_trace_point_pair(dat, "Cal3[{}]".format(calibration_true_values[2]), markers, axes_limits)
        if not good:
            output_error("ERROR user marker error during calibration", output_frame_text)
            return
        markers = pd.concat([markers, cal3_markers])

        # Conduct calibration regression (results stored in object)
        # Now calibration.initialized is True
        calibration.regression(baseline_cal_mean, cal1_mean, cal2_mean, cal3_mean)
        
        # Show the user the regression
        # First, extracting the calibration values from the stored object
        calibration_difference_values = calibration.get_difference()
        calibration_regressed_values = [x * calibration.regression_gradient + calibration.regression_intercept for x in calibration_difference_values]

        # Second, plotting the calibrations for user confirmation 
        _, ax = plt.subplots()
        ax.plot(calibration_difference_values, calibration_true_values, marker="o", color="black", linestyle="None")
        ax.plot(calibration_difference_values, calibration_regressed_values, color="gray", linestyle="dashed")
        plt.xlabel("Measured value (strain difference from baseline)")
        plt.ylabel("True value (g)")
        plt.title("Calibration regression\n(R^2={r}, Inter={i}, Slope={s})".format(r=round(calibration.regression_rsquared,5), 
                                                                                   s=round(calibration.regression_gradient,5),
                                                                                   i=round(calibration.regression_intercept,5)))
        plt.show()

    return calibration

#######
# Function run_weights
#   Calculate multiple weights, using whatever set of functions you'd like, given a start and end index
#   Passes results to output_weights() to format return value string 
# Parameters:
#   dat                    - full dataframe (pandas Dataframe of form Measurement | Datetime)
#   calibration            - initialized weight calibration object (MOM_Calculations.Calibration)
#   start_index            - left index of the trace segment (int)
#   end_index              - right index of the trace segment (int)
#   baseline_mean          - baseline strain value near measurement (float)
#   f_name                 - measurement data file name (str)
#   counter                - number/id marking the relative location of the trace segment within the overall file (MOM_Calculations.Calibration)
#   output_frame_text      - tkinter output text widget frame for writing (tkinter.Text)
#   include_header         - do you want a CSV header line printed before the data line? (bool, default False)
#   write_output_to_screen - do you want to write the output string to the GUI text widget? (bool, default False)
# Returns: 
#   Formatted CSV line, from output_weights (str)
#######
def run_weights(dat, calibration, 
                start_index, end_index, 
                baseline_mean, 
                f_name, counter, 
                output_frame_text, 
                include_header=False, write_output_to_screen=False):
    
    # Call weight calculation functions
    weight_mean, slope = MOM_Calculations.w_mean(dat, calibration, start_index, end_index, baseline_mean)
    weight_median, _ = MOM_Calculations.w_median(dat, calibration, start_index, end_index, baseline_mean)
    weight_min_slope, min_slope, window_start_index, window_end_index = MOM_Calculations.w_windowed_min_slope(dat, calibration, start_index, end_index, baseline_mean)

    # Datetime of the center of the trace segment
    datetime = dat.loc[int((start_index+end_index)/2), "Datetime"]

    # Return the formatted output string as defined in output_weights
    formatted_output_string = output_weights(f_name=f_name,
                                             counter=counter, 
                                             datetime=datetime, 
                                             start_index=start_index, 
                                             end_index=end_index, 
                                             window_start_index=window_start_index, 
                                             window_end_index=window_end_index, 
                                             weight_mean=weight_mean, 
                                             weight_median=weight_median,
                                             weight_min_slope=weight_min_slope,
                                             slope=slope,
                                             min_slope=min_slope,
                                             output_frame_text=output_frame_text,
                                             include_header=include_header,
                                             write_output_to_screen=write_output_to_screen)
    return formatted_output_string

#######
# Function view [CORE OPERATION]
#   View a strain trace file as an interactive pyplot figure
# Parameters:
#   output_frame_text - tkinter output text widget frame for writing (tkinter.Text)
# Returns: None
#######
def view(output_frame_text):

    # User selects file
    f_path = get_user_file()

    # Parse data from the file
    dat = parse_trace_file(f_path)

    # File parse failures will return None data; exit if that occurred
    if dat is None:
        return
    
    # Extract useful file name for display / report
    f_name = os.path.basename(f_path)

    # Write the initial processing output to GUI text widget
    output_header(dat, f_name, "VIEWING", output_frame_text)

    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches((PLOT_VIEWER_WIDTH, PLOT_VIEWER_HEIGHT))
    ax.plot(dat.loc[:,"Measure"])
    ax.set_title(os.path.splitext(f_name)[0])
    plt.show()

#######
# Function process_manual [CORE OPERATION]
#   User opens a strain measurement file from a loadcell, marks calibration segments, and then measures bird segments
#   Write output to GUI text widget
# Parameters:
#   calibration                     - weight calibration object (MOM_Calculations.Calibration)
#   calibration_user_entered_values - tkinter text input frames to update initial calibration values (list of tkinter.Entry)
#   output_frame_text               - tkinter output text widget frame for writing (tkinter.Text)
# Returns: None
#######
def process_manual(calibration, calibration_user_entered_values, output_frame_text):
    
    # User selects file
    f_path = get_user_file()

    # Parse data from the file
    dat = parse_trace_file(f_path)

    # File parse failures will return None data; exit if that occurred
    if dat is None:
        return

    # Extract useful file name for display / report
    f_name = os.path.basename(f_path)

    # Write the initial processing output to GUI text widget
    output_header(dat, f_name, "MANUAL PROCESSING", output_frame_text)

    # Allow the user to calibrate
    #   (will check if calibration object is initialized and, if so, ask user if they want to retain)
    calibration = run_user_calibration(dat, calibration, calibration_user_entered_values, output_frame_text)
    if calibration is None or not calibration.initialized:
        return

    # Output calibration info to GUI
    output_calibration(calibration, output_frame_text)

    # Process multiple birds
    # Looping as long as the user confirms
    bird_counter = 1
    while (True):
        if messagebox.askyesno("Confirmation", "Enter bird data?"):
            # Process a bird

            # Get the baseline measurement from near the bird activity from the user
            # After each measurement, we check that BOTH points are selected, and allow the user to try again if not
            good, bird_baseline_mean, bird_baseline_markers, axes_limits = get_trace_point_pair(dat, "Bird[Baseline]")
            if not good:
                output_error("ERROR user marker error during bird baseline", output_frame_text)
                continue 
            markers = bird_baseline_markers

            # Measurement
            good, _, bird_value_markers, _ = get_trace_point_pair(dat, "Bird[Measure]", markers, axes_limits)
            if not good:
                output_error("ERROR user marker error during bird measurement", output_frame_text)
                continue
            markers = pd.concat([markers, bird_value_markers])

            # Get marked start and end points of bird measurements 
            bird_start_index = bird_value_markers.index[0]
            bird_end_index = bird_value_markers.index[1]

            # Calculate the weight information
            # This also write to GUI output
            # (you can store the return here if needed, which is CSV-formatted string)
            run_weights(dat=dat,
                        calibration=calibration,
                        start_index=bird_start_index,
                        end_index=bird_end_index,
                        baseline_mean=bird_baseline_mean,
                        f_name=f_name,
                        counter=bird_counter,
                        include_header=bird_counter==1,
                        output_frame_text=output_frame_text,
                        write_output_to_screen=True)
            
            # If we've successfully run the weights, we're on to the next bird
            bird_counter = bird_counter + 1
        else:
            break


#######
# ************************IN PROGRESS************************************
# Function process_manual [CORE OPERATION]
#   Currently, the function starts exactly the same as process_manual.
#   User selects a single file and conducts a manual calibration.
#   The program then automatically searches for windows >30g in the trace data.
#   Finds PR peaks (second, second-to-last)
#   Uses w_windowed_min_slope function to calculate weights between those peaks
#   Produces a plot with those weights annotated above each window
# Parameters:
#   calibration                     - weight calibration object (MOM_Calculations.Calibration)
#   calibration_user_entered_values - tkinter text input frames to update initial calibration values (list of tkinter.Entry)
#   output_frame_text               - tkinter output text widget frame for writing (tkinter.Text)
# Returns: None
#######
def process_auto(calibration, calibration_user_entered_values, output_frame_text):
    
    # ------------------
    # Manual calibration
    # ------------------

    # User selects file
    f_path = get_user_file()

    # Parse data from the file
    dat = parse_trace_file(f_path)

    # File parse failures will return None data; exit if that occurred
    if dat is None:
        return

    # Extract useful file name for display / report
    f_name = os.path.basename(f_path)

    # Write the initial processing output to GUI text widget
    output_header(dat, f_name, "AUTO PROCESSING", output_frame_text)

    # Allow the user to calibrate
    #   (will check if calibration object is initialized and, if so, ask user if they want to retain)
    calibration = run_user_calibration(dat, calibration, calibration_user_entered_values, output_frame_text)
    if calibration is None or not calibration.initialized:
        return

    # Output calibration info to GUI
    output_calibration(calibration, output_frame_text)

    # ------------------
    # Automatic measurements from single file
    # ------------------

    # Get the 30g threshold
    threshold_30 = (30 - calibration.regression_intercept) / calibration.regression_gradient + calibration.baseline

    # Threshold the data
    # Values >30g become 1
    # Values <=30g become 0 
    threshed = (dat.loc[:, "Measure"] > threshold_30).astype(int)

    # Open small blips of 0s (<10 samples long)
    # This merges windows that are close to one another
    # For example: --_-------- becomes --------------

    # A string of 0s starts when the value is 0, and the previous (shifted 1) value is 1 
    zero_starts = (threshed == 0) & (threshed.shift(1) == 1)
    # A string of 0s ends when the value is 0, and the next (shifted -1) value is 1
    zero_ends = (threshed == 0) & (threshed.shift(-1) == 1)

    # Get the integer indices associated with those starts and ends
    zero_start_indices = zero_starts[zero_starts].index
    zero_end_indices = zero_ends[zero_ends].index

    # Cut any "ends" that are actually 0s->1s at the beginning of the file
    # As a result, the starts and ends are now aligned to mark a single window of 0s
    zero_end_indices = zero_end_indices[zero_end_indices > zero_start_indices.min()]
    
    # Go through every pair of start-end values (this marks the edges of a window of 0s) 
    for start, end in zip(zero_start_indices, zero_end_indices):
        # if the size of that window of 0s is <10 samples long
        # mark those values as 1s, instead of 0s
        if end - start + 1 < 10 :
            threshed.iloc[start:(end+1)] = 1

    # Now find the windows of 1s
    # These are values registered at >30g

    # A string of 1s starts when the value is 1, and the previous (shifted 1) value is 0
    window_starts = (threshed == 1) & (threshed.shift(1) == 0)
    # A string of 1s ends when the value is 1, and the next (shifted -1) value is 0
    window_ends = (threshed == 1) & (threshed.shift(-1) == 0)

    # Extend the window by one in each direction to get the left and the rightmost peaks included in the window
    # Needed when searching for 2nd peaks for penguin rule
    window_starts_indices = window_starts[window_starts].index - 1
    window_end_indices = window_ends[window_ends].index + 1

    # Cut any "ends" that are actually 1s->0s at the beginning of the file
    # As a result, the starts and ends are now aligned to mark a single window of 1s
    window_end_indices = window_end_indices[window_end_indices > window_starts_indices.min()]

    # Close small blips of 1s (<25 samples long)
    # This erases measurement windows that are very short
    # For example: __-______ becomes _________

    # Make a new list of the open windows to retain
    # (this allows us to retain the original windows if needed, unlike when closed down the blips of 0s)
    retained_window_starts_indices = []
    retained_window_ends_indices = []

    # Go through every pair of start-end values (this marks the edges of a window of 1s) 
    for start, end in zip(window_starts_indices, window_end_indices):
        # if the size of that window of 1s is <25 samples long
        if end - start + 1 > 25:
            # mark those values as 0s, instead of 1s
            retained_window_starts_indices.append(start)
            retained_window_ends_indices.append(end)

    # Lists to keep track of selected peak locations 
    # and corresponding measurements between those peaks
    peak_markers_x = []
    peak_markers_y = []
    measure_centers_x = []
    measure_centers_y = []
    measures = []

    # For each measurement window of 1s, find the peaks and measure within those peaks
    for start, end in zip(retained_window_starts_indices, retained_window_ends_indices):
        # Get the measurement data in the window
        window = dat.loc[start:end, "Measure"]
        
        # Get the sign switches at every location in the trace
        # Start by calculating the rolling difference, like the first derivative: window.diff()
        #   series [a, b, c] becomes series[NA, b-a, c-b]
        # Simplify to the sign of the first derivative: np.sign(window.diff())
        #   increasing = 1, decreasing = -1
        # Now get the derivative of those signs: np.sign(window.diff()).diff()
        #   inc->inc =  1 -  1 =  0
        #   dec->dec = -1 - -1 =  0 
        #   inc->dec = -1 -  1 = -2
        #   dec->inc =  1 - -1 =  2
        # Fill the NA values with 0 
        # (will be the first values in the Series, because there are no previous values to take differences from with .diff)
        sign_switches = np.sign(window.diff()).diff().fillna(0)

        # The PEAKS are areas where the trace is going from increasing to decreasing
        # so find indices in the original trace where the sign_switch is -2
        # Shift one index to the left to adjust for .diff()
        peaks = sign_switches[sign_switches == -2].index - 1

        # Now we are finding the actual measurement window via penguin rule
        # By default, start with the actual starta nd end value of the window
        start_peak_index = start
        end_peak_index = end

        # If there are 3 or more peaks, take the penguin rule peaks
        if len(peaks) > 2:
            # Second peak
            start_peak_index = peaks[1]
            # Second-to-last peak
            end_peak_index = peaks[-2]

        # Find central value of the window
        measure_center = int((start_peak_index + end_peak_index) / 2)

        # Find the weight within the window
        # NOTE you could call run_weights() here, like in process_manual
        #      For doing a batch of files, you could call run_weights() and store the csv-formatted output strings
        measure, _, _, _ = MOM_Calculations.w_windowed_min_slope(dat, calibration, start_peak_index, end_peak_index, calibration.baseline)

        # This is all for plotting
        peak_markers_x.append(start_peak_index)
        peak_markers_x.append(end_peak_index)
        peak_markers_y.append(dat.loc[start_peak_index, "Measure"])
        peak_markers_y.append(dat.loc[end_peak_index, "Measure"])
        measure_centers_x.append(measure_center)
        measure_centers_y.append(dat.loc[measure_center, "Measure"])
        measures.append(round(measure, 2))
    
    # ------------------
    # This is all just plotting
    # ------------------
    peak_markers = pd.DataFrame({"x":peak_markers_x,
                                 "y":peak_markers_y})
    measure_labels = pd.DataFrame({"x":measure_centers_x,
                                   "y":measure_centers_y,
                                   "label":[str(x) for x in measures]})
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig.set_size_inches((15,4))  # (default_figure_width, default_figure_height)) 
    ax1.plot(dat.loc[:, "Measure"])
    ax1.scatter(peak_markers["x"].astype(float), peak_markers["y"].astype(float), color="red")
    for i in range(len(measure_labels)):
        ax1.text(measure_labels["x"].astype(float).iloc[i], 
                 measure_labels["y"].astype(float).iloc[i], 
                 measure_labels["label"].iloc[i], 
                 color="red", horizontalalignment="center")

    # Currently just show plot the user and exit
    ax2.plot(threshed, "r")