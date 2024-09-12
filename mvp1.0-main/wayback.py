import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of materials to search for
materials = ['steel', 'lumber', 'concrete']

# List to store data
data = []

# Loop through each material
for material in materials:
    # Search a site like tradingeconomics for the material
    url = f'https://tradingeconomics.com/commodity/{material}'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Find the historical data table
    table = soup.find('table', {'class': 'historical_data_table'})
    if table:
        # Extract the dates and prices
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 2:
                date = cols[0].text.strip()
                price = cols[1].text.strip()
                data.append({'Material': material, 'Date': date, 'Price': price})
    else:
        print(f"No data found for {material}")

# Convert list of dictionaries into a dataframe
df = pd.DataFrame(data)

print(df.head())  # Display first few rows of the dataframe
