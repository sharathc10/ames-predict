import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
import ImputeHelpers as helpers
from chitra_imputers import impute_with_closest_year, impute_with_mode, impute_subset_with_mode, impute_with_neighbor_mean,impute_subset_with_grouping_mean

def impute_Neighborhood(df):
    
    missing_count = sum(df['Neighborhood'].isna())
    
    if missing_count > 0:
        unique_n = df['Neighborhood'].unique()
        s = pd.Series(np.random.choice(unique_n, size=len(df.index)))
        df['Neighborhoods'].fillna(s, inplace=True)
    
    return df

def encode_Neighborhood(df):
    
        df = helpers.one_hot_encode(df, 'Neighborhood', ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'])
        
        return df
    
    
def clean_GrLivArea(df):
    """Function to clean GrLivArea
    
    Parameters: dfframe
    returns: dfframe
    """
    col = 'GrLivArea'
    
    #Check for missing values
    missing_count = sum(df['GrLivArea'].isna())
    
    #If no missing values, just return the df
    if missing_count > 0:
        df = helpers.mean_impute_by_neighborhood(df)
    
    
    return df

def clean_MSSubClass(df):
    
    #Check for missing values
    missing_count = sum(df['MSSubClass'].isna())
    
    #If no missing values, just return the df
    if missing_count > 0:
        df = mode_impute_by_neighborhood(df, 'MSSubClass')
    
    #Encode
    df = helpers.one_hot_encode(df, 'MSSubClass', ['20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90', '120', '150', '160', '180', '190'])
    
    return df

def clean_MSZoning(df):
    df['MSZoning'] = df['MSZoning'].map(lambda x: x.rstrip(' (all)'))
    df['MSZoning'] = df['MSZoning'].map(lambda x: x.rstrip(' (agr'))
    
    missing_count = sum(df['MSZoning'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'MSZoning')
    
    #encode
    df = helpers.one_hot_encode(df, 'MSZoning', ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'])
    
    return df

def clean_LotFrontage(df):
    #TODO: For some reason it is skipping 3 rows, PID: 916253320, 916252170, 907230240
    
    #Check for missing values
    missing_count = sum(df['LotFrontage'].isna())
    
    #If no missing values, just return the df
    if missing_count > 0:
        df = helpers.mean_impute_by_neighborhood(df, 'LotFrontage')
    
    return df

def clean_LotArea(df):
    #Check for missing values
    missing_count = sum(df['LotArea'].isna())
    
    #If no missing values, just return the df
    if missing_count > 0:
        df = helpers.mean_impute_by_neighborhood(df, 'LotArea')
    
    return df

def clean_Street(df):
    #Check for missing values
    missing_count = sum(df['Street'].isna())
    
    #If no missing values, just return the df
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'Street')
        
    #Encode
    df = helpers.one_hot_encode(df, 'Street', ['Grvl', 'Pave'])
    
    return df

def clean_Alley(df):
    df['Alley'] = df['Alley'].fillna('DNE')
        
    #Encode
    df.Alley.replace({'DNE':0, 'Grvl':1, 'Pave':2}, inplace=True)
    
    return df

def clean_LotShape(df):
    #Check for missing values
    missing_count = sum(df['LotShape'].isna())
    
    #If no missing values, just return the df
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'LotShape')
        
    
    df.LotShape.replace({'IR3':1, 'IR2':2, 'IR1':3, 'Reg':4}, inplace = True)
    
    return df

def clean_LandContour(df):
    
    missing_count = sum(df['LandContour'].isna())
    if missing_count >0:
        df = helpers.mode_impute_by_neighborhood(df, 'LandContour')
    
    #encode
    df = helpers.one_hot_encode(df, 'LandContour', ['Low', 'HLS', 'Bnk', 'Lvl'])
    
    return df

def clean_Utilities(df):
    missing_count = sum(df['Utilities'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'Utilities')
        
    df.Utilities.replace({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}, inplace = True)
    
    return df

def clean_LotConfig(df):
    missing_count = sum(df['LotConfig'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'LotConfig')
        
    df = helpers.one_hot_encode(df, 'LotConfig', ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])
    
    return df

def clean_LandSlope(df):
    missing_count = sum(df['LandSlope'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'LandSlope')
        
    df.LandSlope.replace({'Sev':1, 'Mod':2, 'Gtl':3}, inplace = True)
    
    return df

def clean_Condition1(df):
    missing_count = sum(df['Condition1'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'Condition1')
        
    df = helpers.one_hot_encode(df, 'Condition1', ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
    
    return df

def clean_Condition2(df):
    missing_count = sum(df['Condition2'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'Condition2')
        
    df = helpers.one_hot_encode(df, 'Condition2', ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
    
    return df

def clean_BldgType(df):
    df['BldgType'].replace('2fmCon', '2FmCon', inplace=True)
    
    missing_count = sum(df['BldgType'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'BldgType')
        
    df = helpers.one_hot_encode(df, 'BldgType', ['TwnhsI', 'TwnhsE', 'Twnhs', 'Duplex', '2FmCon', '1Fam'])
    
    return df

def clean_HouseStyle(df):
    
    missing_count = sum(df['HouseStyle'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'HouseStyle')
        
    df.HouseStyle = df.HouseStyle.replace({'1.5Unf': 1, '1.5Fin':2, 'SFoyer':3, '2.5Unf': 4, '1Story':5, 'SLvl': 6, '2Story': 7, '2.5Fin':8})
    
    return df

def clean_OverallQual(df):
    
    missing_count = sum(df['OverallQual'].isna())
    if missing_count > 0:
        df = helpers.mean_impute_by_neighborhood(df, 'OverallQual')
    
    return df

def clean_OverallCond(df):
    
    missing_count = sum(df['OverallCond'].isna())
    if missing_count > 0:
        df = helpers.mean_impute_by_neighborhood(df, 'OverallCond')
    
    return df

def clean_YearRemodAdd(df):
    missing_count = sum(df['YearRemodAdd'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'YearRemodAdd')
    
    return df

def clean_RoofStyle(df):
    missing_count = sum(df['RoofStyle'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'RoofStyle')
        
    df.RoofStyle = df.RoofStyle.replace({'Gambrel': 1, 'Gable':2, 'Mansard':3, 'Hip':4, 'Flat': 5, 'Shed': 6})
    
    return df

def clean_RoofMatl(df):
    missing_count = sum(df['RoofMatl'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'RoofMatl')
        
    df.RoofMatl = df.RoofMatl.replace({'Roll': 1, 'CompShg':2, 'Tar&Grv':3, 'Metal':4, 'WdShake': 5, 'Membran': 6, 'WdShngl':7})
    
    return df

def clean_Exterior1st(df):
    missing_count = sum(df['Exterior1st'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'Exterior1st')
        
    df.Exterior1st = df.Exterior1st.replace({'AsphShn': 1, 'CBlock':2, 'AsbShng':3, 'WdShing':4, 'Stucco': 5, 'MetalSd': 6, 'Wd Sdng':7, 'HdBoard': 8, 'Plywood': 9, 'BrkComm': 10, 'BrkFace': 11, 'VinylSd': 12, 'CemntBd': 13, 'PreCast': 14, 'ImStucc': 15})
    
    return df

def clean_Exterior2nd(df):
    missing_count = sum(df['Exterior2nd'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'Exterior2nd')
        
    df.Exterior2nd = df.Exterior2nd.replace({'Stone': 1, 'CBlock':2, 'AsbShng':3, 'AsphShn':4, 'Stucco': 5, 'MetalSd': 6, 'Wd Shng':7, 'Wd Sdng':8, 'Brk Cmn': 9, 'HdBoard': 10, 'Plywood': 11, 'BrkFace': 12, 'ImStucc': 13, 'VinylSd': 14, 'CmentBd': 15, 'PreCast': 16})
    
    return df

def clean_MasVnrType(df):
    missing_count = sum(df['MasVnrType'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'MasVnrType')
    
    df.MasVnrType = df.MasVnrType.replace({'None': 1, 'BrkCmn':2, 'BrkFace':3, 'Stone':4})
    
    return df

def clean_MasVnrArea(df):
    missing_count = sum(df['MasVnrArea'].isna())
    if missing_count > 0:
        df = helpers.mean_impute_by_neighborhood(df, 'MasVnrArea')
    
    return df

def clean_ExterQual(df):
    missing_count = sum(df['ExterQual'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'ExterQual')
    
    df.ExterQual = df.ExterQual.replace({'Po':1, 'Fa':2, 'TA': 3, 'Gd':4, 'Ex':5})
    
    return df

def clean_ExterCond(df):
    missing_count = sum(df['ExterCond'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'ExterCond')
    
    df.ExterCond = df.ExterCond.replace({'Po':1, 'Fa':2, 'TA': 3, 'Gd':4, 'Ex':5})
    
    return df

def clean_Foundation(df):
    missing_count = sum(df['Foundation'].isna())
    if missing_count > 0:
        df = helpers.mode_impute_by_neighborhood(df, 'Foundation')
    
    df.Foundation = df.Foundation.replace({'PConc':1, 'CBlock':2, 'BrkTil': 3, 'Stone':4, 'Slab':5, 'Wood':6})
    
    return df

def clean_BsmtQual(df):
    df.BsmtQual.fillna('DNE', inplace=True)
    df.BsmtQual = df.BsmtQual.replace({'DNE':0, 'Po':1, 'Fa':2, 'TA': 3, 'Gd':4, 'Ex':5})
    
    return df

def clean_BsmtCond(df):
    df.BsmtCond.fillna('DNE', inplace=True)
    df.BsmtCond = df.BsmtCond.replace({'DNE':0, 'Po':1, 'Fa':2, 'TA': 3, 'Gd':4, 'Ex':5})
    
    return df

def clean_BsmtExposure(df):
    df.BsmtExposure.fillna('DNE', inplace=True)
    df.BsmtExposure = df.BsmtExposure.replace({'DNE':0, 'No':1, 'Mn':2, 'Av': 3, 'Gd':4})    
    
    return df

def clean_BsmtFinType1(df):
    df.BsmtFinType1.fillna('DNE', inplace=True)
    df.BsmtFinType1 = df.BsmtFinType1.replace({'DNE':0, 'Unf':1, 'LwQ': 2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})

    return df

def clean_BsmtFinSF1(df):
    
    missing_count = sum(df['Neighborhood'].isna())
    
    if missing_count > 0:
        df = mean_impute_by_neighborhood(df, 'BsmtFinSF1')
 
    return df

def clean_BsmtFinType2(df):
    df.BsmtFinType2.fillna('DNE', inplace=True)
    df.BsmtFinType2 = df.BsmtFinType2.replace({'DNE':0, 'Unf':1, 'LwQ': 2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})

    return df

def clean_BsmtFinSF2(df):
    
    missing_count = sum(df['Neighborhood'].isna())
    
    if missing_count > 0:
        df = mean_impute_by_neighborhood(df, 'BsmtFinSF2')
 
    return df

def clean_BsmtUnfSF(df):
    
    missing_count = sum(df['Neighborhood'].isna())
    
    if missing_count > 0:
        df = mean_impute_by_neighborhood(df, 'BsmtUnfSF')
 
    return df

def clean_TotalBsmtSF(df):
    
    missing_count = sum(df['Neighborhood'].isna())
    
    if missing_count > 0:
        df = mean_impute_by_neighborhood(df, 'TotalBsmtSF')
 
    return df

def cleanHeating(df):
    if pd.isnull(df['Heating']).sum()>0:
        impute_with_mode(df,'Heating','Neighborhood')
    df['GasHeating']=df.Heating.map(lambda t:1 if t in ['GasA','GasW'] else 0)
    df.drop(['Heating'],axis=1, inplace=True)
    return df

def cleanHeatingQC(df):
    if pd.isnull(df['HeatingQC']).sum()>0:
        impute_with_mode(df,'HeatingQC','Neighborhood')
    df['GoodHeating']=df.HeatingQC.map(lambda t:1 if t in ['Excellent','Good',] else 0)
    df.drop(['HeatingQC'],axis=1, inplace=True)
    return df

def cleanCentralAir(df):
    if pd.isnull(df['CentralAir']).sum()>0:
        impute_with_mode(df,'CentralAir','Neighborhood')
    df['CentralAir']=df.CentralAir.map(lambda t:1 if t=="Yes" else 0)
    return df 

def cleanElectrical(df):
    if pd.isnull(df['Electrical']).sum()>0:
        impute_with_closest_year(df,'Electrical')
    df['ModernElectrical']=df.Electrical.map(lambda t:1 if t=='SBrkr' else 0)
    df.drop(['Electrical'],axis=1, inplace=True)
    return df

def clean1stFlrSF(df):
    if pd.isnull(df['1stFlrSF']).sum()>0:
        impute_with_neighbor_mean(df,'1stFlrSF')
    df.loc[(pd.isnull(df['1stFlrSF'])), '1stFlrSF']=0
    #check GrLivArea?
    return df

def clean2ndFlrSF(df):
    if pd.isnull(df['2ndFlrSF']).sum()>0:
        impute_with_neighbor_mean(df,'2ndFlrSF')
    df.loc[(pd.isnull(df['2ndFlrSF'])), '2ndFlrSF']=0
    #check GrLivArea?
    return df

def cleanLowQualFinSF(df):
    if pd.isnull(df['LowQualFinSF']).sum()>0:
        impute_with_neighbor_mean(df,'LowQualFinSF')
    df.loc[(pd.isnull(df['LowQualFinSF'])), 'LowQualFinSF']=0
     #check GrLivArea?
    return df

def cleanBsmtFullBath(df):
    if pd.isnull(df['BsmtFullBath']).sum()>0:
        bsmtfullbath=df[pd.isnull(df['BsmtFullBath']) & (df['TotalBsmtSF']>0)]
        impute_subset_with_grouping_mean(df,bsmtfullbath,'BsmtFullBath','Neighborhood')
        df.loc[(pd.isnull(df['BsmtFullBath'])), 'BsmtFullBath']=0
    return df

def cleanBsmtHalfBath(df):
    if pd.isnull(df['BsmtHalfBath']).sum()>0:
        bsmthalfbath=df[pd.isnull(df['BsmtHalfBath']) & (df['TotalBsmtSF']>0)]
        impute_subset_with_grouping_mean(df,bsmthalfbath,'BsmtHalfBath','Neighborhood')
        df.loc[(pd.isnull(df['BsmtHalfBath'])), 'BsmtHalfBath']=0
    return df

def cleanFullBath(df):
    if pd.isnull(df['FullBath']).sum()>0:
        fullbath=df[pd.isnull(df['FullBath']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,fullbath,'FullBath','Neighborhood')
        df.loc[(pd.isnull(df['FullBath'])), 'FullBath']=0
    return df

def cleanHalfBath(df):
    if pd.isnull(df['HalfBath']).sum()>0:
        halfbath=df[pd.isnull(df['HalfBath']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,halfbath,'HalfBath','Neighborhood')
    df.loc[(pd.isnull(df['HalfBath'])), 'HalfBath']=0
    return df

def cleanBedroomAbvGr(df):
    if pd.isnull(df['BedroomAbvGr']).sum()>0:
        bedroomabvgr=df[pd.isnull(df['BedroomAbvGr']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,bedroomabvgr,'BedroomAbvGr','Neighborhood')
    df.loc[(pd.isnull(df['BedroomAbvGr'])), 'BedroomAbvGr']=0
    return df

def cleanKitchenAbvGr(df):
    if pd.isnull(df['KitchenAbvGr']).sum()>0:
        kitchenabvgr=df[pd.isnull(df['KitchenAbvGr']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,kitchenabvgr,'KitchenAbvGr','Neighborhood')
    df.loc[(pd.isnull(df['KitchenAbvGr'])), 'KitchenAbvGr']=0
    return df

def cleanKitchenQual(df):
    if pd.isnull(df['KitchenQual']).sum()>0:
        kitchenqual=df[pd.isnull(df['KitchenQual']) & (df['KitchenAbvGr']>0)]
        impute_subset_with_mode(df,kitchenqual,'KitchenQual','Neighborhood')
    df['GoodKitchen']=df.KitchenQual.map(lambda t:1 if t in ['Excellent','Good',] else 0)
    df.drop(['KitchenQual'],axis=1, inplace=True)
    return df

def cleanTotRmsAbvGrd(df):
    if pd.isnull(df['TotRmsAbvGrd']).sum()>0:
        totrmsabvgrd=df[pd.isnull(df['TotRmsAbvGrd']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,totrmsabvgrd,'TotRmsAbvGrd','Neighborhood')
    df.loc[(pd.isnull(df['TotRmsAbvGrd'])), 'TotRmsAbvGrd']=0
    return df

def cleanFunctional(df):
    df.loc[(pd.isnull(df['Functional'])), 'Functional']="Typ"
    df['Functional']=df.Functional.map(lambda t:1 if t=="Typ" else 0)
    return df

def cleanFireplaces(df):
    if pd.isnull(df['Fireplaces']).sum()>0:
        fireplaces=df[pd.isnull(df['Fireplaces']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,fireplaces,'Fireplaces','Neighborhood')
    df.loc[(pd.isnull(df['Fireplaces'])), 'Fireplaces']=0
    return df

def cleanFireplaceQu(df):
    if pd.isnull(df['FireplaceQu']).sum()>0:
        fireplacequal=df[pd.isnull(df['FireplaceQu']) & (df['Fireplaces']>0)]
        impute_subset_with_mode(df,fireplacequal,'FireplaceQu','Neighborhood')
    df.loc[(pd.isnull(df['FireplaceQu'])), 'FireplaceQu']="DNE"
    df.FireplaceQu = df.FireplaceQu.replace({'DNE':0, 'Po':1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5})
    return df

def cleanGarageType(df):
    if pd.isnull(df['GarageType']).sum()>0:
        garagetype=df[pd.isnull(df['GarageType']) & (df['GarageArea']>0)]
        impute_subset_with_mode(df,garagetype,'GarageType','Neighborhood')
    df.loc[(pd.isnull(df['GarageType'])), 'GarageType']="DNE"
    df.GarageType = df.GarageType.replace({'DNE':0, 'CarPort':1, 'Detchd': 2, '2Types':3, 'Basment':4, 'Attchd':5, 'BuiltIn':6})    
    return df

def cleanGarageYrBlt(df):
    if pd.isnull(df['GarageYrBlt']).sum()>0:
        garageyear=df[pd.isnull(df['GarageYrBlt']) & (df['GarageArea']>0)]
        impute_subset_with_mode(df,garageyear,'GarageYrBlt','Neighborhood')
    df.loc[(pd.isnull(df['GarageYrBlt'])), 'GarageYrBlt']="DNE"
  
    return df

def cleanGarageFinish(df):
    if pd.isnull(df['GarageFinish']).sum()>0:
        garagefinish=df[pd.isnull(df['GarageFinish']) & (df['GarageArea']>0)]
        impute_subset_with_mode(df,garagefinish,'GarageFinish','Neighborhood')
    df.loc[(pd.isnull(df['GarageFinish'])), 'GarageFinish']="DNE"
    df.GarageFinish = df.GarageFinish.replace({'DNE':0, 'Unf':1, 'RFn': 2, 'Fin':3})
    return df

def cleanGarageCars(df):
    if pd.isnull(df['GarageCars']).sum()>0:
        garagecars=df[pd.isnull(df['GarageCars']) & (df['GarageArea']>0)]
        impute_subset_with_grouping_mean(df,garagecars,'GarageCars','Neighborhood')
    df.loc[(pd.isnull(df['GarageCars'])), 'GarageCars']=0
    return df

def cleanGarageArea(df):
    if pd.isnull(df['GarageArea']).sum()>0:
        df.loc[(pd.isnull(df['GarageArea'])), 'GarageArea']=0
    return df

def cleanGarageQual(df):
    if pd.isnull(df['GarageQual']).sum()>0:
        garagequal=df[pd.isnull(df['GarageQual']) & (df['GarageArea']>0)]
        impute_subset_with_mode(df,garagequal,'GarageQual','Neighborhood')
    df.loc[(pd.isnull(df['GarageQual'])), 'GarageQual']="DNE"
    df.GarageQual = df.GarageQual.replace({'DNE':0, 'Po':1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5})
    
    return df

def cleanGarageCond(df):
    if pd.isnull(df['GarageCond']).sum()>0:
        garagecond=df[pd.isnull(df['GarageCond']) & (df['GarageArea']>0)]
        impute_subset_with_mode(df,garagecond,'GarageCond','Neighborhood')
    df.loc[(pd.isnull(df['GarageCond'])), 'GarageCond']="DNE"
    df.GarageCond = df.GarageCond.replace({'DNE':0, 'Po':1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5})
    return df

def cleanPavedDrive(df):
    if pd.isnull(df['PavedDrive']).sum()>0:
        paveddrive=df[pd.isnull(df['PavedDrive'])]
        impute_subset_with_mode(df,paveddrive,'PavedDrive','Neighborhood')
    df['PavedDrive']=df.PavedDrive.map(lambda t:1 if t == "Y" else 0)

    return df

def cleanWoodDeckSF(df):
    if pd.isnull(df['WoodDeckSF']).sum()>0:
        wooddeck=df[pd.isnull(df['WoodDeckSF']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,wooddeck,'WoodDeckSF','Neighborhood')
    df.loc[(pd.isnull(df['WoodDeckSF'])), 'WoodDeckSF']=0
    return df

def cleanOpenPorchSF(df):
    if pd.isnull(df['OpenPorchSF']).sum()>0:
        porch=df[pd.isnull(df['OpenPorchSF']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,porch,'OpenPorchSF','Neighborhood')
    df.loc[(pd.isnull(df['OpenPorchSF'])), 'OpenPorchSF']=0
    return df

def cleanEnclosedPorch(df):
    if pd.isnull(df['EnclosedPorch']).sum()>0:
        enclosedporch=df[pd.isnull(df['EnclosedPorch']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,enclosedporch,'EnclosedPorch','Neighborhood')
    df.loc[(pd.isnull(df['EnclosedPorch'])), 'EnclosedPorch']=0
    return df

def clean3SsnPorch(df):
    if pd.isnull(df['3SsnPorch']).sum()>0:
        porch3ssn=df[pd.isnull(df['3SsnPorch']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,porch3ssn,'3SsnPorch','Neighborhood')
    df.loc[(pd.isnull(df['3SsnPorch'])), '3SsnPorch']=0
    return df

def cleanScreenPorch(df):
    if pd.isnull(df['ScreenPorch']).sum()>0:
        screenporch=df[pd.isnull(df['ScreenPorch']) & (df['GrLivArea']>0)]
        impute_subset_with_grouping_mean(df,screenporch,'ScreenPorch','Neighborhood')
    df.loc[(pd.isnull(df['ScreenPorch'])), 'ScreenPorch']=0
    return df

def cleanPoolArea(df):
    if pd.isnull(df['PoolArea']).sum()>0:
        df.loc[(pd.isnull(df['PoolArea'])), 'PoolArea']=0
    return df

def cleanPoolQC(df):
    if pd.isnull(df['PoolQC']).sum()>0:
        poolqc=df[pd.isnull(df['PoolQC']) & (df['PoolArea']>0)]
        impute_subset_with_mode(df,poolqc,'PoolQC','Neighborhood')
    df.loc[(pd.isnull(df['PoolQC'])), 'PoolQC']="DNE"
    df.PoolQC = df.PoolQC.replace({'DNE':0, 'Po':1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5})
    return df

def cleanFence(df):
    if pd.isnull(df['Fence']).sum()>0:
        fence=df[pd.isnull(df['Fence'])]
        impute_subset_with_mode(df,fence,'Fence','Neighborhood')
    df.loc[(pd.isnull(df['Fence'])), 'Fence']="DNE"
    df.Fence = df.Fence.replace({'DNE':0, 'MnWw':1, 'GdWo': 2, 'MnPrv':3, 'GdPrv':4})
    return df

def cleanMiscFeature(df):
    if pd.isnull(df['MiscFeature']).sum()>0:
        df['MiscFeature']=df.MiscFeature.map(lambda t:1 if t in ["Elev","Gar2","Oth","Shed","TenC"] else 0)
    return df

def cleanMiscVal(df):
    #revisit this one
    df.loc[(pd.isnull(df['MiscVal'])), 'MiscVal']=0

    return df

def cleanMoSold(df):
    if pd.isnull(df['MoSold']).sum()>0:
        mosold=df[pd.isnull(df['MoSold'])]
        impute_subset_with_mode(df,mosold,'MoSold','Neighborhood')
    return df

def cleanYrSold(df):
    if pd.isnull(df['YrSold']).sum()>0:
        yrsold=df[pd.isnull(df['YrSold'])]
        impute_subset_with_mode(df,yrsold,'YrSold','Neighborhood')
    return df

def cleanSaleType(df):
    if pd.isnull(df['SaleType']).sum()>0:
        saletype=df[pd.isnull(df['SaleType'])]
        impute_subset_with_mode(df,saletype,'SaleType','Neighborhood')
    df['NewHome']=df.SaleType.map(lambda t:1 if t=="New" else 0)
    df.drop(['SaleType'],axis=1, inplace=True)

    return df

def cleanSaleCondition(df):
    if pd.isnull(df['SaleCondition']).sum()>0:
        salecondition=df[pd.isnull(df['SaleCondition'])]
        impute_subset_with_mode(df,salecondition,'SaleCondition','Neighborhood')
    df['NormalSale']=df.SaleCondition.map(lambda t:1 if t=="Normal Sale" else 0)
    df.drop(['SaleCondition'],axis=1, inplace=True)
    return df