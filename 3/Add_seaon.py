import pandas as pd

# Load the dataset
file_path = "D:/programming/Python/ML/3/Pune Market Price.csv"  # Replace with your actual file path
df = pd.read_csv(file_path, parse_dates=['Arrival_Date'], dayfirst=True)

# Function to determine the season based on the month
def assign_season(month):
    if month in [10, 11, 12, 1, 2, 3]:
        return 'Rabi'      # Winter Season (Oct-Mar)
    elif month in [6, 7, 8, 9]:
        return 'Kharif'    # Monsoon Season (Jun-Sep)
    else:
        return 'Zaid'      # Summer Season (Apr-Jun)

# Ensure 'Arrival_Date' is in proper datetime format
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')

# Apply function to create a new 'Season' column
df['Season'] = df['Arrival_Date'].dt.month.apply(assign_season)

# Save the updated dataset as a new Excel file
output_file = "updated_dataset.xlsx"
df.to_excel(output_file, index=False)

print(f"Updated dataset saved as {output_file}")
