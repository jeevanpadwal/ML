import pandas as pd
from datetime import datetime, timedelta

# Load the dataset
file_path = "your_dataset.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Function to calculate the actual date
def doy_to_date(year, doy):
    return (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")

# Add a new 'DATE' column
df["DATE"] = df.apply(lambda row: doy_to_date(row["YEAR"], row["DOY"]), axis=1)

# Save the updated dataset
df.to_csv("updated_dataset.csv", index=False)
print("Updated dataset saved with DATE column.")
