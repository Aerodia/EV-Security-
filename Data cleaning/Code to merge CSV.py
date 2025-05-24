import pandas as pd
import glob

# Define the path to your CSV files
folder_path = "G:\\CICEVSE\\Network Traffic\\EVSE-B\\csv"
csv_files = glob.glob(folder_path + "\\*.csv")


# Read and combine all CSV files
df_list = [pd.read_csv(f) for f in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)  # Stacks them properly

# Save the merged file
merged_file_B = folder_path + "merged.csv"
merged_df.to_csv(merged_file_B, index=False)

print(f"Merged CSV saved as: {merged_file_B}")