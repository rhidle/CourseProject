import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
 
def hourly_price(df):
    df['Price'].plot()
    plt.xlabel('Hour')
    plt.ylabel('Price[$/MWh]')
    plt.title(f'Hourly Price on {df.index[0].date()}')
    plt.show()

months = ['June', 'July', 'August']

def find_NaN(df):
    print(df.isnull().sum())
for month in months:
    find_NaN(pd.read_csv(f'Raw_price_data/Texas_prices_{month}.csv', sep=','))


def get_prices(month,period):
    Prices_df = pd.read_csv(f'Raw_price_data/Texas_prices_{month}.csv', sep=',')
    Prices_df = Prices_df[Prices_df['Zone'] == 'LZ_AEN']
    Prices_df.index = pd.to_datetime(Prices_df['Date'])
    Prices_df = Prices_df[period]
    return Prices_df

all_prices_df = {}

period = slice(0, 24)
prices_df = get_prices('June', period)
all_prices_df[prices_df.index[0].date()] = prices_df[period]
print(prices_df)
# hourly_price(prices_df)

period = slice(24, 48)
for month in months:
    prices_df = get_prices(month, period)
    all_prices_df[prices_df.index[0].date()] = prices_df
    # Prices_df.to_csv(f'Prices_{month}.csv', sep=',')
    print(prices_df)
    # hourly_price(prices_df)

# Create a correlation matrix for the prices
correlation_matrix = pd.DataFrame()
for date in all_prices_df:
    correlation_matrix[date] = all_prices_df[date]['Price'].to_list()
    correlation_matrix[date].reset_index(drop=True, inplace=True)
    print(correlation_matrix[date])
print(correlation_matrix.corr())

# Look for patterns in the prices



# Resample the data to daily prices
for month in months:
    Prices_df = pd.read_csv(f'Prices_{month}.csv', sep=',')
    Prices_df.index = pd.to_datetime(Prices_df['Date'])
    Prices_df.drop(columns=['Date','Date.1','Zone'], inplace=True)
    Prices_df = Prices_df.resample('D').mean()
    print(Prices_df)
    
# Look for outliers
# for date in all_prices_df:
    # all_prices_df[date].boxplot()
    # plt.show()

# def check_outliers(temp_df):
#     # fill the appropriate z-score value to replace outlier values with nan in the next two lines   
#     temp_df['total_consumption']=temp_df['total_consumption'].mask(np.abs(stats.zscore(temp_df['total_consumption'])) > #here,np.nan)
#     temp_df['solar']=temp_df['solar'].mask(np.abs(stats.zscore(temp_df['solar'])) > #here,np.nan)
#     temp_df['total_consumption']=temp_df['total_consumption'].fillna(method='ffill')
#     temp_df['solar']=temp_df['solar'].fillna(method='ffill')    
    
#     return temp_df

# Check for autocorrelation
for month in months:
    Prices_df = pd.read_csv(f'Prices_{month}.csv', sep=',')
    data = Prices_df['Price']
    plot_acf(data, lags=24)  # Adjust lags based on your time series length and seasonality
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.title(f'Autocorrelation for the two first days of {month}')
    plt.show()
