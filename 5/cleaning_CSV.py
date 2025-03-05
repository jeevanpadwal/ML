import pandas as pd

# Load CSV
file_path = "D:/programming/Python/ML/5/merged_market_weather.csv"
df = pd.read_csv(file_path)

# Display missing values
print("Missing values per column:\n", df.isnull().sum())

# Replace invalid values (-999) and NaN with 0
df.replace(-999, 0, inplace=True)
df.fillna(0, inplace=True)  # Fill all missing values with 0

# Save cleaned file
cleaned_file_path = "D:/programming/Python/ML/5/cleaned_merged_market_weather1.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned CSV saved at: {cleaned_file_path}")
