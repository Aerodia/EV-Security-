import pandas as pd

# Load the dataset
file_path = '/mnt/data/EVSE-B-PowerCombined.csv'
df = pd.read_csv(file_path)

# Display the first few rows and basic information about the dataset
df.head(), df.info(), df.describe()
# Step 1: Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Step 2: Check for duplicate rows
duplicates = df.duplicated().sum()

# Step 3: Trim whitespace and standardize case in categorical columns
categorical_columns = ['State', 'Attack', 'Attack-Group', 'Label', 'interface']
df[categorical_columns] = df[categorical_columns].apply(lambda col: col.str.strip().str.lower())

# Step 4: Check for any anomalies in numerical data (e.g., outliers)
numerical_summary = df.describe()

# Return the number of duplicates and the updated numerical summary
duplicates, numerical_summary

# Remove duplicate rows
df_cleaned = df.drop_duplicates()

# Save the cleaned dataset to a new file
cleaned_file_path = '/mnt/data/EVSE-B-PowerCombined_Cleaned.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

cleaned_file_path
