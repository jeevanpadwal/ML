import requests
import pandas as pd
from bs4 import BeautifulSoup

# Base URL of the website
base_url = "http://www.puneapmc.org"

# Main page URL
url = "http://www.puneapmc.org/rates.aspx"

# Headers to mimic a real browser
headers = {"User-Agent": "Mozilla/5.0"}

# Step 1: Get the main page and find all "View Rates" links
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Find all "View Rates" links
links = []
for a in soup.find_all("a", href=True):
    if "View Rates" in a.text:
        full_link = base_url + "/" + a["href"].lstrip("/")  # Ensure proper formatting
        links.append(full_link)

# Step 2: Visit each link and scrape the data
all_data = []
for link in links:
    print(f"Scraping: {link}")
    response = requests.get(link, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract date from the page
    date_heading = soup.find("h3")
    date = date_heading.text.strip() if date_heading else "Unknown Date"

    # Find the market rates table
    table = soup.find("table")
    if not table:
        print(f"Skipping {link}, no table found.")
        continue

    # Extract data from the table
    for row in table.find_all("tr"):
        cols = [col.text.strip() for col in row.find_all(["td", "th"])]
        if cols:
            cols.insert(0, date)  # Add date column
            all_data.append(cols)

# Step 3: Save the data to a CSV file
df = pd.DataFrame(all_data)
df.to_csv("market_prices_history.csv", index=False, header=False, encoding="utf-8-sig")

print("All data saved to market_prices_history.csv")
