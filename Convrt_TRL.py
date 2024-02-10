import datetime

# this works: replaces the Unix TimeStamp with a long date/time text stamp
# assumes: filename is in same folder as this app

# Function to convert Unix timestamp to normal timestamp format
def unix_to_normal(unix_timestamp):
    return datetime.datetime.fromtimestamp(int(unix_timestamp)).strftime('%m/%d/%Y %H:%M:%S')

# Open the file for reading
with open("DL_Expt.txt", "r") as file:
    lines = file.readlines()

# Open the same file for writing (this will overwrite the existing file)
with open("DL_Expt.txt", "w") as file:
    for line in lines:
        # Split the line into columns
        columns = line.strip().split(", ")
        
        # Convert the second column (Unix timestamp) to normal timestamp format
        columns[1] = unix_to_normal(columns[1])
        
        # Join the columns back into a line and write it to the file
        file.write(", ".join(columns) + "\n")
