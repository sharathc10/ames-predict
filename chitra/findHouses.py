import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import CustomPipeline as cp
from geopy.distance import geodesic 
imputedHousingData=pd.read_csv("./data/predictedSalePrice.csv")
mappingData = pd.read_csv('./location_mapping/full_latlon.csv')

def convert_latitude(coordinate):
    return float(str(coordinate).replace("(","").replace(")","").split(",")[0])

def convert_longitude(coordinate):
    if (len(str(coordinate).replace("(","").replace(")","").split(",")))==2:
        return float(str(coordinate).replace("(","").replace(")","").split(",")[1])
    else:
        return 0

def get_distance(pointa,pointb):
    return geodesic(pointa,pointb).miles

def find_homes(df,distance):
    print(df[((df['distUni']<=distance) & 
     ((df['BedroomAbvGr']/df['FullBath']<=2) | 
      (((df['BsmtFinType1']>=5) | 
        (df['BsmtFinType2']>=5) & 
        (df["BsmtFullBath"]>=1)))) &
    (df['OverallCond'] <= 3) & (df['OverallQual']<=3))])
    
imputedHousingData_map = pd.merge(imputedHousingData, mappingData, on='PID')
imputedHousingData_map["latitude"]=imputedHousingData_map['coord'].apply(convert_latitude)
imputedHousingData_map["longitude"]=imputedHousingData_map['coord'].apply(convert_longitude)
ten_percent = imputedHousingData_map.shape[0]*0.1
cheapest_homes=imputedHousingData_map.sort_values("SalePrice").head(round(ten_percent))
uni_coord=pd.DataFrame({'name':['Iowa State University'],'latitude':[42.023949], 'longitude':[-93.647595]})
university_coordinates = (uni_coord.loc[0,'latitude'],uni_coord.loc[0,'longitude'])
cheapest_homes.loc[:,'distUni']=cheapest_homes.apply(lambda x: get_distance((x.latitude, x.longitude),university_coordinates), axis=1)
distance_values=[0.5,1,1.25,1.5]
for val in distance_values:
    if val==1:
        print(f'Homes within {val} mile:\n')
    else:
        print(f'Homes within {val} miles:\n')
    find_homes(cheapest_homes,val)
    print("\n")