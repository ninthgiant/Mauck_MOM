# Mauck_MOM
New repository for GUI work. Development until now done in other repositories, but I corrupted those somehow. This is a do-over.
 
As of June 28, 2023, MOM_GUI_v01.py opens a window with 3 buttons to work with MOM datafiles. The user can input the weights of the known calibration weights. Information on any opened file is presented to the user in three columns. 

In the same folder (directory) you need to have both the MOM_GUI_v01.py and the MOM_GUI_set_user_values.py files. You should change these default settings in MOM_GUI_set_user_values.py to match your local machine. However, you can also change them on-screen each time you use the app.

-- June 5, 2024
Added 3 input fields for automation of the analysis. The idea is for the user to designate the beginning and end of the points in a trace that should be considered for estimating the weight of the bird. The automation process then looks for points where the bird was not moving since these should give the most accurate reading. These points are defined as "stable" points for a minimum amount of time (i.e. # of points at 50 SPS) that are sufficiently similar to each other. These new input fields, therefore, control 1) how many consecutive points are needed to be included, 2) how different can the consecutive points and still be considered as "stable" consecutive points, and 3) STD of the consecutive points. Currently only the first two fields are used. The automation process scans the designated trace and creates a new array that only contains the sections of the trace that qualify as "stable". This will include non-contiguous "windows" that qualify as stable. The averaging then occurs only over these stable points.

The user should only choose a window for automation that excludes stable points before (i.e. baseline) and after the bird is on the platform because those will be stable and be included in the average. In general, the user will see when the bird is fully on the platform and designate only that section to be automated.

Note that the original method of averaging is still available for use using the middle button.

Todo: other methods of automated calculation, for example, methods outlined in the Penguin papers, such as multiple regression with multiple characteristics of the trace.
