# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 00:35:59 2022

@author: suryavp
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# load the excel file
temp_df = pd.read_csv('15minute_data_austin.csv', sep=';')
#print all column names
print(list(temp_df.columns))
# get dataid,localminute,grid and solar and solar1
df = temp_df[['dataid','localminute','grid','solar','solar2']]
print(df)
#print datatypes of all columns
print(df.dtypes)

#convert localminute datatype into datetime
df['localminute']= pd.to_datetime(df['localminute'])
print(df.dtypes) 
# print unique values in dataid
data_id = df['dataid'].unique()
print(data_id)
#count how many houses' consumption data are in dataframe using data_id.
num_houses = len(data_id)
print(num_houses)

#seaprate the data based on dataid and store it in a separate dataframe.

client_data=list()

for j in range(num_houses):
    #name=str(client_names[j])
    name=df.loc[df['dataid']==data_id[j]]
    name=name.drop(['dataid'], axis=1)    
    client_data.append(name)    


# create a function to check for NaN values in df columns: grid,solar,solar2
def check_nan(temp_df):
   #check whether there is any nan values in grid column
   check_nan = temp_df['grid'].isna().sum()
   print(f'Sum of NaN before fill in grid column: {check_nan}')
   #fill the nan values in grid column using ffill method
   temp_df['grid'] = temp_df['grid'].fillna(method='ffill')
   # check again to make sure the above method worked
   check_nan = temp_df['grid'].isna().sum()
   print(f'Sum of NaN after fill in grid column: {check_nan}')
   #check nan in solar column
   check_nan = temp_df['solar'].isna().sum()
   print(f'Sum of NaN before fill in solar column: {check_nan}')
   #replace the nan values  with zero in solar column
   temp_df['solar'] = temp_df['solar'].fillna(0)
   #check again
   check_nan = temp_df['solar'].isna().sum()
   print(f'Sum of NaN after fill in solar column: {check_nan}')
   #check nan in solar2
   check_nan = temp_df['solar2'].isna().sum()
   print(f'Sum of NaN before fill in solar 2 column: {check_nan}')
   #replace the nan values  with zero in solar2 column
   temp_df['solar2'] = temp_df['solar2'].fillna(0)
   #check again
   check_nan = temp_df['solar2'].isna().sum()
   print(f'Sum of NaN after fill in solar 2 column: {check_nan}')   
   return temp_df
# check for missing timestamp.
def check_timestamp(temp_df):
    # sort the dataframe in ascending order with column localminute using sort_values function
    temp_df = temp_df.sort_values(by='localminute') 
    temp_df=temp_df.set_index('localminute')  
    print(temp_df)
    # use reindex function to introduce missing timestamps and use ffill to fill values in those       
    temp_df = temp_df.reindex(pd.date_range(temp_df.index.min(), temp_df.index.max(), freq='15min')).ffill()
    temp_df.index.name = "localminute"    
    print(temp_df)     
    
    return temp_df
def sum_data(temp_df):
    # sum the values of grid solar and solar 2 row wise and assign it to the column total consumption.
    total = temp_df['grid']+temp_df['solar']+temp_df['solar2']
    temp_df['total_consumption'] = total    
    return temp_df

def check_negative_consumption(temp_df):
    # check if there are any negative values in total_consumption column
    negative_rows = temp_df[temp_df['total_consumption'] < 0]
    print(f'Negative rows: {negative_rows}')
    
def resample_data(temp_df):
    # resample the data to 60 minute frequency using resample function and use mean method    
    temp_df = temp_df.resample('60min').mean().round(4)
    print(temp_df)       
    return temp_df

def extract_time_features(wat):
    wat=wat.reset_index()
    print(wat)
    # use datetime index function to extract following features/values from localminute column    
    wat['day'] = wat['localminute'].dt.day
    wat['month'] = wat['localminute'].dt.month
    wat['day_of_week'] = wat['localminute'].dt.dayofweek
    wat['hour'] = wat['localminute'].dt.hour
    wat['weekend']=(wat['day_of_week'] > 4).astype(float)
    wat.set_index('localminute',inplace=True)
    print(wat)    
    return wat
def extract_cons_features(temp_df):
    # extract previous three consumption to the current timestep and assign it to c-1,c-2,c-3.
    # tip-check shift function and fill unavailable values with zero. 
    temp_df['c-1']= temp_df['total_consumption'].shift(1)
    temp_df['c-2']= temp_df['total_consumption'].shift(2)
    temp_df['c-3']= temp_df['total_consumption'].shift(3)
    temp_df=temp_df.fillna(0)
    print(temp_df)
    return temp_df
def check_outliers(temp_df):
    # fill the appropriate z-score value to replace outlier values with nan in the next two lines   
    temp_df['total_consumption']=temp_df['total_consumption'].mask(np.abs(stats.zscore(temp_df['total_consumption'])) > 3,np.nan)
    temp_df['solar']=temp_df['solar'].mask(np.abs(stats.zscore(temp_df['solar'])) > 3,np.nan)
    temp_df['total_consumption']=temp_df['total_consumption'].fillna(method='ffill')
    temp_df['solar']=temp_df['solar'].fillna(method='ffill')    
    
    return temp_df

#house number
cl_num=9 
          
client_data[cl_num]=check_nan(client_data[cl_num])
client_data[cl_num]=check_timestamp(client_data[cl_num]) 
client_data[cl_num]=sum_data(client_data[cl_num])
client_data[cl_num]=resample_data(client_data[cl_num])
client_data[cl_num]=extract_time_features(client_data[cl_num])
client_data[cl_num]=extract_cons_features(client_data[cl_num])
check_negative_consumption(client_data[cl_num])
client_data[cl_num]=check_outliers(client_data[cl_num])

# data analysis
# find total monthly consumption and plot bar graph
def monthly_consumption(temp_df):
    consumption=list()
    solar_consumption=list()
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']    
    for j in range(12): 
        #use loc function to get dataframe with only single month data.
        val = temp_df.loc[temp_df.index.month==j+1]
        # find monthly total consumption 
        t_cons = val ['total_consumption'].sum()
        # find monthly solar contribution
        sol_cons = val['solar'].sum()
        consumption.append(t_cons)
        solar_consumption.append(sol_cons)
    print(consumption)
    #plot them
    plt.bar(month,consumption)
    plt.bar(month,solar_consumption)
    plt.legend(['Total Consumption','Solar'])
    plt.xlabel('Month')
    plt.ylabel('Consumption[kWh]')
    plt.title('Monthly Consumption')
    plt.show()
    
# monthly_consumption(client_data[cl_num])

def correlation(temp_df):
    #Find the correlation between total consumption and other features using the corr() function.
    cor = temp_df.select_dtypes(include=[float, int]).corr()
    print(cor['total_consumption'])
    
correlation(client_data[cl_num])

def boxplot(temp_df):
    #box plot for total consumption and solar column values
    #create plot code here
    plt.boxplot(temp_df[['total_consumption', 'solar']], labels=['Total Consumption', 'Solar'])
    plt.ylabel('Consumption[kWh]')
    plt.title('Boxplot of Total Consumption and Solar')
    plt.show()
# boxplot(client_data[cl_num])    

def histogram(temp_df):
    # create code for total consumption histogram
    plt.hist(temp_df['total_consumption'])
    plt.xlabel('Consumption[kWh]')
    plt.title('Total Consumption Histogram')
    plt.show()
    #create code for solar histogram
    plt.hist(temp_df['solar'])
    plt.xlabel('Solar[kWh]')
    plt.title('Solar Histogram')
    plt.show()
    
# histogram(client_data[cl_num])

def min_max_avg(temp_df):
   #Find maximum, minimum, and average daily consumption    
    # temp_df=temp_df.set_index('localminute')
    temp_df=temp_df.resample('D').sum()
    print(temp_df)
    minimum_cons = temp_df['total_consumption'].min()
    print(f'minimum daily consumption: {minimum_cons}') 
    # find the date
    minimum_cons_date = temp_df['total_consumption'].idxmin()
    print(f'date of the minimum daily consumption: {minimum_cons_date}')
    maximum_cons = temp_df['total_consumption'].max()
    print(f'maximum daily consumption: {maximum_cons}')    
    maximum_cons_date = temp_df['total_consumption'].idxmax()
    print(f'date of the maximum daily consumption: {maximum_cons_date}')
    avg_cons = temp_df['total_consumption'].mean()
    print(f'Average daily consumption: {avg_cons}')    
    
def weekday_weekend(temp_df): 
    # use groupby function to group the dataframe based on day of week
    # sum the total consumption values in each group.
    dow = temp_df['total_consumption'].groupby(temp_df['day_of_week']).sum()
    print(dow)
    # this plot should give total consumption values from mon to sun
    dow.plot.bar(y='total_consumption')
    # use proper x-axis labels.
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Consumption[kWh]')
    plt.title('Total Consumption per Day of the Week')
    plt.show()

def hourly_consumption(temp_df):
    # find the hourly consumption for 1.6.2018 and 2.6.2018
    hourly = temp_df['2018-06-01':'2018-06-03']
    print(hourly)
    # Plot it
    hourly['total_consumption'].plot()
    plt.xlabel('Hour')
    plt.ylabel('Consumption[kWh]')
    plt.title('Hourly Consumption on 1.6.2018 and 2.6.2018')
    plt.show()
    

   
    
hourly_consumption(client_data[cl_num])
# min_max_avg(client_data[cl_num])
# weekday_weekend(client_data[cl_num]) 

# Save the data to a new csv file
client_data[cl_num].to_csv('house9_data.csv', sep=',')

# Prices_df = pd.read_csv('Texas_prices_July.csv', sep=',')
# Prices_df = Prices_df[Prices_df['Zone'] == 'LZ_AEN']
# Prices_df.index = pd.to_datetime(Prices_df['Date'])
# Prices_df.to_csv('Prices_July.csv', sep=',')
# print(Prices_df)

def hourly_price(df):
    df['Price'].plot()
    plt.xlabel('Hour')
    plt.ylabel('Price[$/MWh]')
    plt.title('Hourly Price on 1.6.2018 and 2.6.2018')
    plt.show()

hourly_price(Prices_df)