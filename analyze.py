import pandas as pd
import numpy as np
indicator = pd.read_csv("FRED1960-2024.csv")

long_term_avg = indicator["Indicator"].mean()


specific_yrs = ["2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

filter = []
for yr in specific_yrs:
    for month in months:

        check = indicator["DATE"].str.startswith(f'{yr}-{month}')
        specific = indicator[check]
        filter.append(specific)



filter_indicator = pd.concat(filter, ignore_index = True)
filter_indicator = filter_indicator.copy()
filter_indicator["DATE"] = pd.to_datetime(filter_indicator["DATE"], format = '%Y-%m-%d')
filter_indicator.loc[:, "Qrt"] = filter_indicator["DATE"].dt.to_period("Q")

qrt_avg = filter_indicator.groupby("Qrt")["Indicator"].mean().reset_index()

sub = long_term_avg
qrt_avg = qrt_avg.copy()
qrt_avg["+/-"] = qrt_avg["Indicator"] - sub

def label(val):
    if val >=1:
        return "Expansion"
    elif val <=-1:
        return "Recession"
    elif val <1 and val >=0:
        return "Little Growth"
    else:
        return "Little Decline" 
        

qrt_avg["Condition"] = qrt_avg["+/-"].apply(label)
qrt_avg["Qrt. change"] = qrt_avg["+/-"].diff()
final = qrt_avg.drop([0,1], axis=0).reset_index()



