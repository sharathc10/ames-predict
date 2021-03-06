import pandas as pd
import numpy as np
import CleaningFunctions as cf

def clean(path):
    df = pd.read_csv(path)
    df.drop('Unnamed: 0', axis= 1, inplace=True)

    df = df[df['Neighborhood'] != 'Landmrk']
    df = df[df['Neighborhood'] != 'GrnHill']
    df = df[df['Neighborhood'] != 'Greens']
    df.drop_duplicates(subset='PID', keep='first', inplace=True)

    df.reset_index(drop=True, inplace=True)
    
    df = cf.impute_Neighborhood(df)
    df = cf.clean_GrLivArea(df)
    df = cf.clean_MSSubClass(df)
    df = cf.clean_MSZoning(df)
    df = cf.clean_LotFrontage(df)
    df = cf.clean_LotArea(df)
    df = cf.clean_Street(df)
    df = cf.clean_Alley(df)
    df = cf.clean_LotShape(df)
    df = cf.clean_LandContour(df)
    df = cf.clean_Utilities(df)
    df = cf.clean_LotConfig(df)
    df = cf.clean_LandSlope(df)
    df = cf.clean_Condition1(df)
    df = cf.clean_Condition2(df)
    df = cf.clean_BldgType(df)
    df = cf.clean_HouseStyle(df)
    df = cf.clean_OverallQual(df)
    df = cf.clean_OverallCond(df)
    df = cf.clean_YearRemodAdd(df)
    df = cf.clean_RoofStyle(df)
    df = cf.clean_RoofMatl(df)
    df = cf.clean_Exterior1st(df)
    df = cf.clean_Exterior2nd(df)
    df = cf.clean_MasVnrType(df)
    df = cf.clean_MasVnrArea(df)
    df = cf.clean_ExterQual(df)
    df = cf.clean_ExterCond(df)
    df = cf.clean_Foundation(df)
    df = cf.clean_BsmtQual(df)
    df = cf.clean_BsmtCond(df)
    df = cf.clean_BsmtExposure(df)
    df = cf.clean_BsmtFinType1(df)
    df = cf.clean_BsmtFinSF1(df)
    df = cf.clean_BsmtFinType2(df)
    df = cf.clean_BsmtFinSF2(df)
    df = cf.clean_BsmtUnfSF(df)
    df = cf.clean_TotalBsmtSF(df)
    df = cf.cleanHeating(df)
    df = cf.cleanHeatingQC(df)
    df = cf.cleanCentralAir(df)
    df = cf.cleanElectrical(df)
    df = cf.clean1stFlrSF(df)
    df = cf.clean2ndFlrSF(df)
    df = cf.cleanLowQualFinSF(df)
    df = cf.cleanBsmtFullBath(df)
    df = cf.cleanBsmtHalfBath(df)
    df = cf.cleanFullBath(df)
    df = cf.cleanHalfBath(df)
    df = cf.cleanBedroomAbvGr(df)
    df = cf.cleanKitchenAbvGr(df)
    df = cf.cleanKitchenQual(df)
    df = cf.cleanTotRmsAbvGrd(df)
    df = cf.cleanFunctional(df)
    df = cf.cleanFireplaces(df)
    df = cf.cleanFireplaceQu(df)
    df = cf.cleanGarageType(df)
    df = cf.cleanGarageYrBlt(df)
    df = cf.cleanGarageFinish(df)
    df = cf.cleanGarageCars(df)
    df = cf.cleanGarageArea(df)
    df = cf.cleanGarageQual(df)
    df = cf.cleanGarageCond(df)
    df = cf.cleanPavedDrive(df)
    df = cf.cleanWoodDeckSF(df)
    df = cf.cleanOpenPorchSF(df)
    df = cf.cleanEnclosedPorch(df)
    df = cf.clean3SsnPorch(df)
    df = cf.cleanScreenPorch(df)
    df = cf.cleanPoolArea(df)
    df = cf.cleanPoolQC(df)
    df = cf.cleanFence(df)
    df = cf.cleanMiscFeature(df)
    df = cf.cleanMiscVal(df)
    df = cf.cleanMoSold(df)
    df = cf.cleanYrSold(df)
    df = cf.cleanSaleType(df)
    df = cf.cleanSaleCondition(df)
    df = cf.encode_Neighborhood(df)
    
    
    return df