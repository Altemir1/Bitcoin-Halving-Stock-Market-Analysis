#!/usr/bin/env python
# coding: utf-8

# # Data collection

# S&P500 ETF

# In[5]:


import requests
import pandas as pd
from datetime import datetime

# Replace with your Alpha Vantage API key
API_KEY = 'Z358OL15QMSRKU2R'

def fetch_sp500_data(api_key, start_date, end_date):
    symbol = 'SPY'  # ETF tracking the S&P 500 index
    function = 'TIME_SERIES_DAILY'
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}&outputsize=full'

    response = requests.get(url)
    data = response.json()

    if 'Time Series (Daily)' in data:
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index', dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns=lambda x: x[3:])  # Remove the '1. ', '2. ', etc. prefixes
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        df.sort_index(inplace=True)
        return df
    else:
        print("Error fetching data:", data.get("Error Message", "Unknown error"))
        return None

# Define date range
start_date = datetime(2012, 11, 27)
end_date = datetime.today()

# Fetch data
sp500_data = fetch_sp500_data(API_KEY, start_date, end_date)

if sp500_data is not None:
    # Save to CSV
    sp500_data.to_csv('sp500_daily_prices.csv')
    print("Data saved to 'sp500_daily_prices.csv'")
else:
    print("Failed to retrieve data.")


# Investment Sectors Data

# In[6]:


get_ipython().system('pip install yfinance')


# In[48]:


import yfinance as yf
import pandas as pd
from datetime import datetime

# Sector Tickers
sector_tickers = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

# Define date range
start_date = "2012-11-27"
end_date = datetime.today().strftime('%Y-%m-%d')

# Fetch and save data for each sector
all_sector_data = {}

for sector, ticker in sector_tickers.items():
    print(f"Fetching data for {sector} ({ticker})...")
    try:
        # Download data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df.reset_index(inplace=True)  # Reset index to include dates as a column
        df.to_csv(f"/Users/altemir_1/Desktop/BTC-Stock-Market-Analysis/data/raw/{sector}_daily_prices.csv", index=False)  # Save to CSV
        print(f"Data for {sector} saved to '{sector}_daily_prices.csv'.")
        all_sector_data[sector] = df
    except Exception as e:
        print(f"Failed to fetch data for {sector}: {e}")


# # Data Preprocessing

# In[35]:


import pandas as pd

btc = pd.read_csv("/Users/altemir_1/Desktop/BTC-Stock-Market-Analysis/data/raw/bitcoin_historical_data.csv")
sp500 = pd.read_csv("/Users/altemir_1/Desktop/BTC-Stock-Market-Analysis/data/raw/sp500_historical_data.csv")
sectors = pd.read_csv("/Users/altemir_1/Desktop/BTC-Stock-Market-Analysis/data/raw/investment_sectors_historical_data.csv")


# In[4]:


def check_for_null_values(df):
    print("-" * 40)
    return df.isnull().sum()

data = [btc, sp500, sectors]

for df in data:
    print(check_for_null_values(df))


# ### Bitcoin and S&P 500

# Dropping and renaming columns in btc and sp500 dfs

# In[36]:


# Helper functions
def drop_unnecessary_columns(df, columns):
   return df.drop(columns=columns)

def rename_columns(df, columns):
   return df.rename(columns=columns)


# In[37]:


# Creating list and dict of columns to drop and rename
btc_columns_to_drop = ["Open", "High", "Low"]
sp500_columns_to_drop = ["open", "high", "low"]

btc_rename_columns = {"Date":"date", "Price": "price", "Vol.": "volume", "Change %": "change_rate"}
sp500_rename_columns = {"close": "price", "Unnamed: 0": "date"}

# Applying helper functions
btc = drop_unnecessary_columns(btc, btc_columns_to_drop)
btc = rename_columns(btc, btc_rename_columns)

sp500 = drop_unnecessary_columns(sp500, sp500_columns_to_drop)
sp500 = rename_columns(sp500, sp500_rename_columns)


# Converting date in both dfs to one format of date

# In[38]:


# Convert date columns in both DataFrames to datetime
btc['date'] = pd.to_datetime(btc['date'])
sp500['date'] = pd.to_datetime(sp500['date'])

# Optionally, format as string in 'YYYY-MM-DD' format
btc['date'] = btc['date'].dt.strftime('%Y-%m-%d')
sp500['date'] = sp500['date'].dt.strftime('%Y-%m-%d')

# Verify the format
print(btc['date'].head())
print(sp500['date'].head())


# Adding change_rate to sp500 df

# In[39]:


# Calculate daily percentage change for S&P 500
sp500['change_rate'] = sp500['price'].pct_change() * 100


# Converting object columns that contain numerical data to the numerical columns

# In[43]:


# Updated function to convert volume from BTC to USD
def convert_volume_to_usd(volume, price):
    """
    Convert volume from BTC to USD by handling shorthand notation
    (K for thousand, M for million, B for billion) and then multiplying
    by the Bitcoin price on the respective date.
    """
    if "K" in volume:
        volume_value = float(volume.replace("K", "")) * 1_000
    elif "M" in volume:
        volume_value = float(volume.replace("M", "")) * 1_000_000
    elif "B" in volume:
        volume_value = float(volume.replace("B", "")) * 1_000_000_000
    else:
        volume_value = float(volume)  # Handle cases without suffix

    # Convert to USD by multiplying by the price on the same date
    return volume_value * price


# Convert price to float
# Ensure the column is of string type
btc["price"] = btc["price"].astype(str)

# Replace commas and convert to float
btc["price"] = btc["price"].str.replace(",", "").astype(float)


btc["volume"] = btc["volume"].astype(str)
# Apply the updated function to the DataFrame
btc["volume"] = btc.apply(lambda row: convert_volume_to_usd(row["volume"], row["price"]), axis=1)

# Convert change_rate to float (removing '%' and dividing by 100 if needed)
btc["change_rate"] = btc["change_rate"].str.replace("%", "").astype(float)


# ### Investment Sectors

# Loading each sector into dataframe

# In[ ]:


import os
# Define the folder path where all sector CSVs are stored
folder_path = "/Users/altemir_1/Desktop/BTC-Stock-Market-Analysis/data/raw/investment_sectors/"  # Replace with the actual folder path

# Create a dictionary to store DataFrames for each sector
sector_dfs = {}

# Iterate through the folder to load all CSV files into DataFrames
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Ensure it's a CSV file
        sector_name = file_name.replace("_daily_prices.csv", "").replace("_", " ")
        file_path = os.path.join(folder_path, file_name)
        sector_dfs[sector_name] = pd.read_csv(file_path)


# Defininng functions to preprocess dataframes

# In[93]:


def drop_columns(df):
    columns = ["Adj Close", "High", "Low", "Open"]
    df.drop(columns=columns, inplace=True)

# First row contains NaN values and Tickers 
def drop_first_row(df):
    return df.iloc[1:].reset_index(drop=True)

def rename_columns(df):
    columns = {"Date": "date", "Close": "price", "Volume": "volume"}
    return df.rename(columns=columns)



# In[ ]:


for sector, df in sector_dfs.items():
    try:
        sector_dfs[sector] = drop_columns(df)
    except KeyError:
        print("Columns already deleted")
    sector_dfs[sector] = drop_first_row(df)

    sector_dfs[sector] = rename_columns(df)


    # Convert 'Date' column to datetime
    sector_dfs[sector]["date"] = pd.to_datetime(sector_dfs[sector]["date"], format="%Y-%m-%d", errors="coerce")  # Automatically detects multiple formats

    # Ensure all dates are in the same format
    sector_dfs[sector]["date"] = sector_dfs[sector]["date"].dt.strftime("%Y-%m-%d")

    # Convert object values into numerical
    sector_dfs[sector]["volume"] = sector_dfs[sector]["volume"].astype(float)
    sector_dfs[sector]["price"] = sector_dfs[sector]["price"].astype(float)

    # Adding percent of return
    sector_dfs[sector]["change_rate"] = sector_dfs[sector]['price'].pct_change() * 100
    


# Merging all sectors into one csv file

# In[98]:


def merge_sector_dfs(sector_dfs):
    # Add a new column to each DataFrame to identify its sector
    for sector, df in sector_dfs.items():
        df['sector'] = sector  # Add the sector name as a new column

    # Concatenate all DataFrames vertically
    merged_df = pd.concat(sector_dfs.values(), axis=0, ignore_index=True)

    return merged_df

all_sectors_df  = merge_sector_dfs(sector_dfs)


# In[99]:


all_sectors_df.head()


# Fill null values

# In[102]:


sp500 = sp500.fillna(0)
all_sectors_df = all_sectors_df.fillna(0)


# # Processed Data Installation

# In[103]:


import os

# Base folder path where the CSVs will be saved
base_folder_path = "/Users/altemir_1/Desktop/BTC-Stock-Market-Analysis/data/processed/"

# Ensure the folder exists
os.makedirs(base_folder_path, exist_ok=True)

# Save the BTC DataFrame
btc_csv_path = os.path.join(base_folder_path, "btc_processed.csv")
btc.to_csv(btc_csv_path, index=False)
print(f"BTC DataFrame saved to {btc_csv_path}")

# Save the SP500 DataFrame
sp500_csv_path = os.path.join(base_folder_path, "sp500_processed.csv")
sp500.to_csv(sp500_csv_path, index=False)
print(f"S&P 500 DataFrame saved to {sp500_csv_path}")

# Save the merged all_sectors_df
all_sectors_csv_path = os.path.join(base_folder_path, "all_sectors_processed.csv")
all_sectors_df.to_csv(all_sectors_csv_path, index=False)
print(f"All Sectors DataFrame saved to {all_sectors_csv_path}")

