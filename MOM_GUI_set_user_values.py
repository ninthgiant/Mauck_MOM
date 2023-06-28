#
# SET Some user-specific data - temporary solution
#  This must be in same folder as the MOM_Processor_v03.py file
#  Will eventually be replaced by a text file
#

# Calibration values, in grams
global cal1_value
global cal2_value
global cal3_value

## SET THESE VALUES to match the standard weights you use in the field
cal1_value = 27.65
cal2_value = 50.0
cal3_value = 65.3

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
