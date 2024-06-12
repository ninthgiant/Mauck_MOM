import numpy as np

# Assuming filtered_final_series is your existing series
filtered_final_series = np.array([1, 4, 7, 2, 9])  # Replace this with your actual series

# mean distance
Diff_series = np.zeros_like(filtered_final_series, dtype=int)
for i in range(1, len(Diff_series) -1):
    Diff_series[i]= abs(filtered_final_series[i]- filtered_final_series[i + 1])
    
mean_Diff = np.mean(Diff_series)
print(mean_Diff)

