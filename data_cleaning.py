# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:54:39 2019

@author: Vincenzo

It's a python script to clean data and add feature engineering without copy paste it each time I changed something 
for test, train data.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def clean_fe(df, drop_outliners = False):
    df_dropped = df.drop(['PassengerId','Cabin','Ticket'], axis = 1)
    df_dropped['Fare']=df_dropped['Fare'].fillna(df_dropped['Fare'].median())
    df_dropped['Embarked']=df_dropped['Embarked'].fillna('S')
    
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df_dropped["Name"]]
    df_dropped["Title"] = pd.Series(dataset_title)
    df_dropped["Title"] = df_dropped["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_dropped["Title"] = df_dropped["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    df_dropped["Title"] = df_dropped["Title"].astype(int)

    df_dropped['Age']=np.where((df_dropped['Title']=='Master') & (np.isnan(df_dropped['Age'])), 12, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Miss') & (np.isnan(df_dropped['Age'])), 14, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Mr') & (np.isnan(df_dropped['Age'])), 35, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Mrs') & (np.isnan(df_dropped['Age'])), 24, df_dropped['Age'])
    df_dropped['Age']=df_dropped['Age'].fillna(df_dropped['Age'].median())
    
    df_dropped['AgeStatus'] = np.nan
    df_dropped['AgeStatus'] = np.where((df_dropped['Age']<=12),0,df_dropped['AgeStatus'])
    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>12) & (df_dropped['Age']<=18),1,df_dropped['AgeStatus'] ) 
    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>18) & (df_dropped['Age']<=24),2,df_dropped['AgeStatus'] )
    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>24) & (df_dropped['Age']<=55),3,df_dropped['AgeStatus'] ) 
    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>55) & (df_dropped['Age']<=70),4,df_dropped['AgeStatus'] ) 
    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>70),5,df_dropped['AgeStatus'] ) 


    df_dropped = df_dropped.drop(['Name','Age'], axis = 1)
    df_drop_nonan = df_dropped.dropna()
    
    df_drop_nonan.loc[:,'IsAlone'] = np.where(df_drop_nonan.loc[:,'Parch'] + df_drop_nonan.loc[:,'SibSp'] == 0,1,0)

    
    df_drop_nonan = df_drop_nonan.replace('male',1)
    df_drop_nonan = df_drop_nonan.replace('female',0)
    
    df_drop_nonan['Embarked'] = df_drop_nonan['Embarked'].replace('S',0)
    df_drop_nonan['Embarked'] = df_drop_nonan['Embarked'].replace('C',1)
    df_drop_nonan['Embarked'] = df_drop_nonan['Embarked'].replace('Q',2)
    
    from scipy import stats
    if drop_outliners == True:
        df_drop_nonan = df_drop_nonan[(np.abs(stats.zscore(df_drop_nonan)) < 3).all(axis=1)]
    
    df_drop_nonan.reset_index(inplace = True)
     
    scaler = StandardScaler()
    columns_to_scale = ['Fare']

    temp_df = df_drop_nonan[columns_to_scale]

    temp_df = pd.DataFrame(scaler.fit_transform(temp_df))
    temp_df.columns = columns_to_scale
    
    df_drop_nonan_sc = df_drop_nonan.drop(columns_to_scale, axis = 1)
    df_drop_nonan_sc = df_drop_nonan_sc.join(temp_df)
    
    return df_drop_nonan_sc
