import pandas as pd

# Load the dataset
file_path = "D:/programming/Python/ML/6/market_prices_history.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format (handling missing values as NaT)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Sort the dataset to ensure the latest date is at the top
df.sort_values(by=['Date'], ascending=False, inplace=True)

# Fill missing dates by backfilling
last_known_date = df['Date'].max()
date_range = pd.date_range(end=last_known_date, periods=len(df), freq='-1D')
df['Date'] = date_range[::-1]  # Reverse to maintain order

# Save the cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("Missing dates filled and saved as 'cleaned_data.csv'")
