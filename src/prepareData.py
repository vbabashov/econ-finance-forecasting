import numpy as np
import pandas as pd
#%matplotlib qt


def splitRegionData(filepath, filename):
    df1 = pd.read_excel(filepath+'\\'+filename)
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1.set_index(['Date', 'Region'], inplace=True)
    df1.reset_index()

    # totalDemand = df1.groupby(['Region','Date']).count()
    totalDemand = df1.groupby(['Region'])
    # totalDemand.groups.keys()
    for index, group in totalDemand:
        group.to_csv(filepath+'\\'+index+".csv")



# FilePaths
filepath = "B:\\projects\\econ-finance-forecasting\\data"
filename = "originalData.xlsx"

# # Split Data into regions *** Need to do it once ***
# # splitRegionData(filepath, filename)


df = pd.read_csv(filepath+'\\'+'R1.csv')
df.drop(['Region', 'Site'], inplace=True, axis=1)
df.set_index(['Date'], inplace=True)
df['total'] = df.sum(axis=1)
df['total'].plot(kind='line')
