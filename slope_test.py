import pandas as pd
import numpy as np

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

# Example usage
my_df = pd.read_csv('Slope_test.csv')
my_df.columns = ['Measure', 'UT']
measure_series = my_df['Measure']
the_answer = do_Slope_0(measure_series, 12, 80, 30, 8600000, 0.75)
print(the_answer)

