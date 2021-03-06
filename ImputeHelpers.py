import pandas as pd
import numpy as np
import math

def one_hot_encode(df, column, possible_values):
    shape = (len(df), len(possible_values))
    zeros = np.zeros(shape)
    columns = [column + '_' + val for val in possible_values]
    encoded_df = pd.DataFrame(data = zeros, columns = columns)
    
    s = df[column]
    for i in range(len(s)):
        encoded_df.iloc[i, possible_values.index(str(s[i]))] = 1
        
    df.drop(column, axis = 1, inplace = True)
    
    df = df.join(encoded_df, how='inner')
    
    return df



def mode_impute_by_neighborhood(df, col):
    """Function to impute missing values with the mode of the column by neighborhood
    
    params:
        df <pandas.dataframe>: a pandas dataframe where the imputation will take place
        col <str>: the column where the missing values are
        
        return: the new dataframe
    """
    
    # create a series with unique neighborhoods as index and the mode of that neighborhood as value
    frequent = df[[col, 'Neighborhood']].groupby(['Neighborhood'])[col].agg(pd.Series.mode)
    
    # check if there are many values tie for the mode. Choose only 1 at random
    for index in frequent.index:
        el = frequent[index]
        if type(el) == np.ndarray:
            new_el = np.random.choice(el)
            frequent[index] = new_el
            
    
    # Pass through the dataframe and replace the missing values with the mode
    for n in df['Neighborhood'].unique():
        s = df[col][df['Neighborhood'] == n].fillna(value=frequent.loc[n])
        for index in s.index:
            df.loc[df.index[index], col] = s[index]
    
    return df


def mean_impute_by_neighborhood(df, col):
    """Function to impute missing values with the mode of the column by neighborhood
    
    params:
        df <pandas.dataframe>: a pandas dataframe where the imputation will take place
        col <str>: the column where the missing values are
        
        return: the new dataframe
    """
    
    # create a series with unique neighborhoods as index and the mode of that neighborhood as value
    frequent = df[[col, 'Neighborhood']].groupby(['Neighborhood'])[col].agg('mean')
    
    # Pass through the dataframe and replace the missing values with the mode
    for n in df['Neighborhood'].unique():
        s = df[col][df['Neighborhood'] == n].fillna(value=frequent.loc[n])
        for index in s.index:
            df.loc[df.index[index], col] = s[index]
    
    return df




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
        
        