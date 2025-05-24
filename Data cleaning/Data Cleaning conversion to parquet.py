import pandas as pd

# 🔹 Load the large CSV file efficiently
file_path = "G:\\CICEVSE\\Network Traffic\\EVSE-B\\csvmerged.csv"
df = pd.read_csv(file_path, low_memory=False)

# ✅ Remove duplicate rows
df.drop_duplicates(inplace=True)

# ✅ Drop columns that are mostly empty (more than 80% missing values)
df.dropna(axis=1, thresh=int(0.2 * len(df)), inplace=True)

# ✅ Fill missing values intelligently
for col in df.select_dtypes(include=['number']).columns:  
    df[col].fillna(df[col].median(), inplace=True)  # Fill missing numbers with median

for col in df.select_dtypes(include=['object']).columns:  
    df[col].fillna(df[col].mode()[0], inplace=True)  # Fill missing text with most common value

# ✅ Optimize data types for memory efficiency
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

# 🔹 Save the cleaned file
cleaned_path = file_path.replace("merged.csv", "merged_cleaned.csv")
df.to_csv(cleaned_path, index=False)

# ✅ Optional: Save as Parquet for better efficiency
try:
    df.to_parquet(cleaned_path.replace(".csv", ".parquet"), index=False)
except ImportError:
    print("Parquet libraries not installed. Skipping Parquet save.")

print(f"✅ Cleaned CSV saved at: {cleaned_path}")