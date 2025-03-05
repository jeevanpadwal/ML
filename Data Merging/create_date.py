import pandas as pd

# Load market price data
market_file = "D:/programming/Python/ML/Data Merging/onion_pune_quanity_market_price.csv"  # Update with your file path
market_df = pd.read_csv(market_file, parse_dates=["Reported Date"], dayfirst=True)

# Load weather data
weather_file = "D:/programming/Python/ML/Data Merging/updated_dataset.csv"  # Update with your file path
weather_df = pd.read_csv(weather_file, parse_dates=["DATE"])

# Merge on matching dates
merged_df = market_df.merge(weather_df, left_on="Reported Date", right_on="DATE", how="left")

# Drop duplicate date column (optional)
merged_df.drop(columns=["DATE"], inplace=True)

# Save merged dataset
merged_file = "merged_market_weather.csv"
merged_df.to_csv(merged_file, index=False)

print(f"Merged data saved to {merged_file}")
