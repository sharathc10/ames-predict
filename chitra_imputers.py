import pandas as pd
import math

def find_neighbours(value, df, colname):
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind] 
        
def impute_with_closest_year(df,missingcol):
    for idx in df[pd.isnull(df[missingcol])].index:
        year=df.loc[idx,'YearBuilt']
        neighbor=find_neighbours(year,df,'YearBuilt')
        df.loc[idx,missingcol]=df.loc[neighbor[0],missingcol]
    return df

def impute_with_mode(df,missingcol,groupingcol):
      for idx in df[pd.isnull(df[missingcol])].index:
            modevalue=df.groupby(groupingcol)[missingcol].agg(pd.Series.mode)
            df.loc[idx,missingcol]=modevalue[df.loc[idx,groupingcol]]

def impute_subset_with_mode(df,subset,missingcol,groupingcol):
    neighborMode=df.groupby(groupingcol)[missingcol].apply(lambda x: x.mode())
    for idx in subset.index:
        try:
            modevalue=neighborMode[df.loc[idx,groupingcol],0]
        except:
            modevalue="NA"
        df.loc[idx,missingcol]=modevalue

def impute_with_neighbor_mean(df,missingcol):
    neighborMeans=df.groupby("Neighborhood")[missingcol].agg('mean')
    missingcolIndex=df[pd.isnull(df[missingcol])].index
    for idx in missingcolIndex:
        value=neighborMeans[df.loc[idx,'Neighborhood']]
        df.loc[idx,missingcol]=value

def impute_subset_with_grouping_mean(df,subset,missingcol,groupingcol):
    """
    For a given dataframe and a subset of rows in that dataframe, impute all
    missing values in the subset within the missingcol with the aggregate
    mean by grouping on the groupingcol
    """
    neighborMeans=df.groupby(groupingcol)[missingcol].agg('mean')
    for idx in subset.index:
        value=neighborMeans[df.loc[idx,groupingcol]]
        df.loc[idx,missingcol]=math.ceil(value)