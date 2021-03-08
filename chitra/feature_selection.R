library(dplyr)
library(factoextra)

ames_house_price_train=read.csv("./data/imputedTrainData.csv",na.strings=c("DNE"),check.names = FALSE)

#linear regression with all lasso variables
#VIF analysis
x_lasso_variables = ames_house_price_train %>%  dplyr::select(
  'GrLivArea', 'OverallQual', 'YearBuilt', 'OverallCond', 'BsmtFinSF1',
  'TotalBsmtSF', 'Neighborhood_Crawfor', 'GarageCars',
  'Neighborhood_Somerst', 'Neighborhood_NridgHt', '1stFlrSF', 'LotArea',
  'Neighborhood_StoneBr', 'BsmtExposure', 'YearRemodAdd', 'ExterQual',
  'Functional', 'Exterior1st', 'FireplaceQu', 'ScreenPorch', 
  'Fireplaces',
  'GarageCond', 'EnclosedPorch', 'Neighborhood_NoRidge', 'PavedDrive',
  'GarageArea', 'MSSubClass_70', 'MSSubClass_50', 'BsmtFullBath',
  'Condition1_Norm', 'MSSubClass_20', 'TotRmsAbvGrd', 'BsmtQual',
  'BsmtFinType1', 'BldgType_1Fam', 'Neighborhood_ClearCr',
  'Neighborhood_BrkSide', 'HalfBath', 'FullBath', 'BsmtFinSF2', 'NewHome',
  'LandContour_HLS', '3SsnPorch', 'Condition1_PosN', 'MSSubClass_75',
  'WoodDeckSF', 'Neighborhood_Timber', 'GarageFinish', 'Condition2_PosA',
  'PoolQC', 'MSZoning_RL', 'Neighborhood_IDOTRR', 'RoofMatl',
  'MSSubClass_85', 'Condition1_PosA', 'RoofStyle', 'Neighborhood_Blueste',
  'Utilities', 'Alley', 'GasHeating', 'Street_Grvl', 'LotConfig_CulDSac',
  'GarageQual', 'Neighborhood_Sawyer', 'Condition2_Artery', 'MasVnrArea',
  'ModernElectrical', 'Neighborhood_BrDale', 'LotFrontage', 'ExterCond',
  'Neighborhood_CollgCr', 'GarageType'
)
model = lm(log(ames_house_price_train$SalePrice) ~ ., data=x_lasso_variables)
summary(model)
plot(model)
vif(model)
vif_pre_values=vif(model)
vif_pre_df = data.frame(
  name=names(vif_pre_values),
  value=vif_pre_values
)
ggplot(vif_pre_df, aes(x=name, y=value) ) +
  geom_bar(stat='identity', fill="blue") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) +
  xlab("Feature names") +
  ylab("VIF values") 

#backward step wise regression
#did not improve VIF
model.empty = lm(log(ames_house_price_train$SalePrice)  ~ 1, data = x_lasso_variables)
scope = list(lower = formula(model.empty), upper = formula(model))
backwardAIC = step(model, scope, direction = "backward", k = 2)
summary(backwardAIC)
vif(backwardAIC)
backwardBIC = step(model, scope, direction = "backward", k = log(50))
vif(backwardAIC)

#PCA 
res.pca <- prcomp(x_lasso_variables, scale = TRUE)
fviz_eig(res.pca)
fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

#Post VIF
x_variables = ames_house_price_train %>%  dplyr::select(
  'GrLivArea', 'OverallQual', 'YearBuilt', 
  'Neighborhood_Crawfor', 
  'Neighborhood_Somerst', 'Neighborhood_NridgHt', 'LotArea',
  'Neighborhood_StoneBr', 'BsmtExposure', 'YearRemodAdd', 'ExterQual',
  'Functional', 'Exterior1st', 'FireplaceQu', 'ScreenPorch', 
  'Fireplaces',
  'GarageCond', 'EnclosedPorch', 'PavedDrive',
  'GarageArea', 'BsmtFullBath',
  'Condition1_Norm', 'MSSubClass_20', 'TotRmsAbvGrd', 'BsmtQual',
  'BsmtFinType1', 'BldgType_1Fam', 'Neighborhood_ClearCr',
  'Neighborhood_BrkSide', 'HalfBath', 'BsmtFinSF2', 'NewHome',
  'LandContour_HLS', '3SsnPorch', 'Condition1_PosN', 
  'WoodDeckSF', 'Neighborhood_Timber', 'GarageFinish', 
  'PoolQC', 'MSZoning_RL', 'Neighborhood_IDOTRR', 'RoofMatl',
  'RoofStyle', 
  'Alley', 'GasHeating',  'LotConfig_CulDSac',
  'Neighborhood_Sawyer', 'MasVnrArea',
  'ModernElectrical',  'LotFrontage', 'ExterCond',
  'Neighborhood_CollgCr', 'GarageType'
)
model = lm(log(ames_house_price_train$SalePrice) ~ ., data=x_variables)
summary(model)
plot(model)
vif(model)
vif_values=vif(model)
vif_df = data.frame(
  name=names(vif_values),
  value=vif_pre_values
)
ggplot(vif_df, aes(x=name, y=value) ) +
  geom_bar(stat='identity', fill="blue") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) +
  xlab("Feature names") +
  ylab("VIF values") 


