import pandas as pd

# Load the CSV
df = pd.read_csv('D:/programming/Python/ML/5/merged_market_weather.csv')

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Define numerical columns
numerical_cols = [
    'Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 
    'Modal Price (Rs./Quintal)', 'WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 
    'T2M', 'T2MDEW', 'PRECTOTCORR'
]

# Convert to numeric, coercing errors to NaN
for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Define market and weather columns
market_cols = [
    'Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 
    'Modal Price (Rs./Quintal)'
]
weather_cols = [
    'WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 'T2M', 'T2MDEW', 'PRECTOTCORR'
]

# Fill NaNs with median
for col in market_cols + weather_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Convert 'Reported Date' to datetime
if 'Reported Date' in df.columns:
    df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d-%m-%Y', errors='coerce')

# Fill YEAR and DOY from Reported Date if missing
if 'YEAR' in df.columns and 'Reported Date' in df.columns:
    df['YEAR'] = df['YEAR'].fillna(df['Reported Date'].dt.year)

if 'DOY' in df.columns and 'Reported Date' in df.columns:
    df['DOY'] = df['DOY'].fillna(df['Reported Date'].dt.dayofyear)

# Replace any remaining NaN values with 0 to prevent errors
df.fillna(0, inplace=True)

# Check data types
print(df.dtypes)

# Display the first few rows to verify
print(df.head())
