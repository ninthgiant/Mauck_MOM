#######################################
#######################################
#    MOM_GUI.py
#       GUI elements and execution 
#       R.A.M and L.U.T.
#       2024-08-27 cleanup of RAM_v10
#       CHANGES: 9/1/2024
#           Add button for batch auto processing
#           Add function def: setup_gui() to properlyl handle GUI
#       CHANGES: 12/6/2024
#           Change Auto button to perform Duration calculations - change in list of button names and call to method
#######################################
#######################################

############## values to be set with preferences/globals

#######################################
#######################################
# Imports and libraries
#######################################
#######################################

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import MOM_Processing
import MOM_Calculations
import MOM_Globals

#######################################
#######################################
# GUI Parameters
#######################################
#######################################

# Program screen
# Minimum size to fall back to if the monitor is small or not yet known.
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

# Three subpanels: Inputs, Buttons, Outputs

# Input frame
INPUT_PAD_X = 30
INPUT_PAD_Y = 20

# Button frames
BUTTON_PAD_X = 5
BUTTON_PAD_Y = 5
BUTTON_WIDTH = 20

# Which interface to use
use_Duration = False

if use_Duration:
    BUTTON_LABELS = ["View", "Manual", "Duration", "Auto Batch"]
else:
    BUTTON_LABELS = ["View", "Manual", "Automatic", "Auto Batch"]

# Output frames
OUTPUT_FRAME_WIDTH = 400
OUTPUT_FRAME_HEIGHT = 300
OUTPUT_TEXT_PAD_X = 50
OUTPUT_TEXT_PAD_Y = 25

def setup_gui():
    # Initialize GUI
    # Root window
    root = tk.Tk()
    # Measure the current monitor; prefer half the screen if that is larger than the minimums.
    myScreen_Width = root.winfo_screenwidth()
    myScreen_Height = root.winfo_screenheight()
    window_width = max(SCREEN_WIDTH, myScreen_Width // 2)
    window_height = max(SCREEN_HEIGHT, myScreen_Height // 2)

    root.geometry(f"{window_width}x{window_height}")
    root.minsize(SCREEN_WIDTH, SCREEN_HEIGHT)
    root.title(MOM_Globals.vAppName)

    # Initialize calibrations
    calibration = MOM_Calculations.Calibration()

    # Input frame for user-defined calibration values 
    input_frame = tk.Frame(root, width=SCREEN_WIDTH, bd=1, padx=INPUT_PAD_X, pady=INPUT_PAD_Y, relief=tk.FLAT)
    input_frame.pack()
    initial_calibration_labels = ["Calib. mass light:", "Calib. mass med:", "Calib. mass heavy:"]
    initial_calibration_values = MOM_Globals.get_calibration_values_for_year(MOM_Globals.calib_Year)

    # Initialize the input frames, filled by default with the default-initialized true calibration values
    # NOTE these will change automatic if you change the default values in the Calibration constructor
    calibration_user_entered_values = []
    for i, label in enumerate(initial_calibration_labels):
        tk.Label(input_frame, text=initial_calibration_labels[i]).grid(row=0, column=i + 1, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
        entry = tk.Entry(input_frame)
        entry.grid(row=1, column=i + 1, padx=5, pady=5)
        entry.insert(0, initial_calibration_values[i])
        calibration_user_entered_values.append(entry)

    tk.Label(input_frame, text="Calib Year:").grid(row=0, column=0, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    year_var = tk.StringVar(value=str(MOM_Globals.calib_Year))

    def on_calibration_year_change(selected_year):
        calib_values = MOM_Globals.set_calibration_year(int(selected_year))
        for i, entry in enumerate(calibration_user_entered_values):
            entry.delete(0, tk.END)
            entry.insert(0, calib_values[i])
        calibration.set_true(*calib_values)

    year_options = [str(year) for year in MOM_Globals.get_calibration_year_options()]
    year_menu = tk.OptionMenu(input_frame, year_var, *year_options, command=on_calibration_year_change)
    year_menu.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    # Buttons
    button_frame = tk.Frame(root, width=BUTTON_WIDTH-10, bd=0, relief=tk.SOLID)
    button_frame.pack(pady=5)

    # Place buttons in frames
    # Each button is associated with a core function in MOM_Processing
    buttons = []

    def set_buttons_state(state):
        for button in buttons:
            button.config(state=state)

    # View button calls MOM_Processing.view()
    button_view = tk.Button(button_frame, text=BUTTON_LABELS[0], command=lambda: MOM_Processing.view(output_frame_text))
    button_view.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_view)

    # Manual button calls MOM_Processing.process_manual()
    button_manual = tk.Button(button_frame, text=BUTTON_LABELS[1], command=lambda: MOM_Processing.process_manual(calibration, calibration_user_entered_values, output_frame_text))
    button_manual.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_manual)

    if not use_Duration:
        # Auto button calls MOM_Processing.process_auto()
        button_auto = tk.Button(button_frame, text=BUTTON_LABELS[2], command=lambda: MOM_Processing.process_auto(calibration, calibration_user_entered_values, output_frame_text))
    else:
        # Auto button now calls MOM_Processing.view("duration")
        button_auto = tk.Button(button_frame, text=BUTTON_LABELS[2], command=lambda: MOM_Processing.view(output_frame_text, "duration"))

    button_auto.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_auto)

    # Auto Batch button calls MOM_Processing.process_auto_batch_2() 
    button_auto_batch = tk.Button(
        button_frame,
        text=BUTTON_LABELS[3],
        command=lambda: MOM_Processing.process_auto_start(
            calibration,
            calibration_user_entered_values,
            output_frame_text,
            show_graph=True,
            on_batch_start=lambda: set_buttons_state("disabled"),
            on_batch_end=lambda: set_buttons_state("normal")
        )
    )
    button_auto_batch.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_auto_batch)

    # Review Batch Summary button
    button_review = tk.Button(button_frame, text="Review Batch Summary", command=lambda: MOM_Processing.Batch_Review(output_frame_text))
    button_review.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_review)

    # Add a Trim button - calls MOM_Processing.trim()- NEW FEATURE
    # button_trim = tk.Button(button_frame, text="Trim", command=lambda: MOM_Processing.trim())
    # button_trim.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    # buttons.append(button_trim)


    # Output frame with text widget
    output_frame = tk.Frame(root, width=OUTPUT_FRAME_WIDTH, height=OUTPUT_FRAME_HEIGHT, bd=0, relief=tk.SOLID)
    output_frame.pack(padx=OUTPUT_TEXT_PAD_X, pady=OUTPUT_TEXT_PAD_Y)

    output_header_frame = tk.Frame(output_frame, bd=0, relief=tk.FLAT)
    output_header_frame.pack(fill=tk.X)

    def clear_output_with_confirm():
        if not messagebox.askyesno("Confirm Clear", "Clear all text from the output box?", default="no"):
            return
        output_frame_text.configure(state="normal")
        output_frame_text.delete("1.0", tk.END)
        output_frame_text.configure(state="disabled")

    clear_button = tk.Button(
        output_header_frame,
        text="Clear Output",
        command=clear_output_with_confirm
    )
    clear_button.pack(side=tk.RIGHT, padx=BUTTON_PAD_X, pady=(0, BUTTON_PAD_Y))

    output_body_frame = tk.Frame(output_frame, bd=0, relief=tk.FLAT)
    output_body_frame.pack(fill=tk.BOTH, expand=True)

    # Vertical scrollbar for text widget
    output_scrollbar = tk.Scrollbar(output_body_frame, orient="vertical")
    output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Place text widgets in frame
    output_frame_text = tk.Text(
        output_body_frame,
        width=OUTPUT_FRAME_WIDTH,
        height=OUTPUT_FRAME_HEIGHT,
        yscrollcommand=output_scrollbar.set,
        font=(MOM_Globals.screen_font, MOM_Globals.screen_font_size),
    )
    output_frame_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    output_scrollbar.config(command=output_frame_text.yview)

    # Output text widget cannot be configured by user
    # NOTE configure back to "normal" state before writing output from program
    #      this is usually performed within output_() functions
    output_frame_text.configure(state="disabled")

    return root


#############
# Added to let GUI present information correctly
#####
if __name__ == "__main__":
    root = setup_gui()
    def on_app_close():
        try:
            MOM_Processing.on_close()
        except Exception:
            pass
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            root.quit()
        except tk.TclError:
            pass

    root.protocol("WM_DELETE_WINDOW", on_app_close)
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
