# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:54:39 2019

@author: Vincenzo

It's a python script to clean data and add feature engineering without copy paste it each time I changed something 
for test, train data.
"""
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd


def clean_fe(df, drop_outliners = False):
    df_dropped = df.drop(['PassengerId','Ticket'], axis = 1)
    df_dropped['Embarked']=df_dropped['Embarked'].fillna('S')
    
    dataset_title = [i.split(',')[1].split('.')[0].strip() for i in df_dropped['Name']]

    df_dropped['Title'] = pd.Series(dataset_title)
    to_fix = {'Ms':'Miss','Mme':'Madame', 'Mlle':'Mademoiselle', 'Mrs':'Missus',
              'Mr':'Mister'}
    df_dropped['Title'] = df_dropped['Title'].replace(to_fix)
    df_dropped['Title'] = df_dropped['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major',
              'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    df_dropped['Cabin'] = df_dropped['Cabin'].fillna('U')
    df_dropped['Cabin'] = [i.split(' ')[0] for i in df_dropped['Cabin']]
    df_dropped['Cabin'] = df_dropped['Cabin'].str.replace('\d', '')
    
    df_dropped['CabinA'] = np.where((df_dropped['Cabin']=='A'), 1,0)
    df_dropped['CabinB'] = np.where((df_dropped['Cabin']=='B'), 1,0)
    df_dropped['CabinC'] = np.where((df_dropped['Cabin']=='C'), 1,0)
    df_dropped['CabinD'] = np.where((df_dropped['Cabin']=='D'), 1,0)
    df_dropped['CabinE'] = np.where((df_dropped['Cabin']=='E'), 1,0)
    df_dropped['CabinF'] = np.where((df_dropped['Cabin']=='F'), 1,0)
    df_dropped['CabinG'] = np.where((df_dropped['Cabin']=='G'), 1,0)
    df_dropped['CabinU'] = np.where((df_dropped['Cabin']=='U'), 1,0)
    
    df_dropped = df_dropped.drop(columns = 'Cabin')
    
    
    mean_age_master = df_dropped['Age'][df_dropped['Title']=='Master'].median()
    mean_age_mister = df_dropped['Age'][df_dropped['Title']=='Mister'].median()
    mean_age_miss = df_dropped['Age'][df_dropped['Title']=='Miss'].median()
    mean_age_madame = df_dropped['Age'][df_dropped['Title']=='Madame'].median()
    mean_age_missus = df_dropped['Age'][df_dropped['Title']=='Missus'].median()
    mean_age_madmo = df_dropped['Age'][df_dropped['Title']=='Mademoiselle'].median()


    df_dropped['Age']=np.where((df_dropped['Title']=='Master') & (np.isnan(df_dropped['Age'])), mean_age_master, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Mister') & (np.isnan(df_dropped['Age'])), mean_age_mister, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Miss') & (np.isnan(df_dropped['Age'])), mean_age_miss, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Missus') & (np.isnan(df_dropped['Age'])), mean_age_missus, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Mademoiselle') & (np.isnan(df_dropped['Age'])), mean_age_madmo, df_dropped['Age'])
    df_dropped['Age']=np.where((df_dropped['Title']=='Madame') & (np.isnan(df_dropped['Age'])), mean_age_madame, df_dropped['Age'])
    df_dropped['Age']=df_dropped['Age'].fillna(df_dropped['Age'].median())
#    
#    
#    df_dropped['AgeStatus'] = np.nan
#    df_dropped['AgeStatus'] = np.where((df_dropped['Age']<=8),0,df_dropped['AgeStatus'])
#    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>8) & (df_dropped['Age']<=16),1,df_dropped['AgeStatus'] ) 
#    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>16) & (df_dropped['Age']<=24),2,df_dropped['AgeStatus'] )
#    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>24) & (df_dropped['Age']<=38),3,df_dropped['AgeStatus'] ) 
#    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>38) & (df_dropped['Age']<=55),4,df_dropped['AgeStatus'] ) 
#    df_dropped['AgeStatus'] = np.where((df_dropped['Age']>55),5,df_dropped['AgeStatus'] ) 
#    
#    df_dropped['AgeStatus1'] = np.where(df_dropped['AgeStatus'] == 0, 1, 0)
#    df_dropped['AgeStatus2'] = np.where(df_dropped['AgeStatus'] == 1, 1, 0)
#    df_dropped['AgeStatus3'] = np.where(df_dropped['AgeStatus'] == 2, 1, 0)
#    df_dropped['AgeStatus4'] = np.where(df_dropped['AgeStatus'] == 3, 1, 0)
#    df_dropped['AgeStatus5'] = np.where(df_dropped['AgeStatus'] == 4, 1, 0)
#    
#    df_dropped = df_dropped.drop(columns = ['Age', 'AgeStatus'])
    

    df_dropped['Title_mister'] = np.where(df_dropped['Title'] =='Mister', 1, 0)
    df_dropped['Title_miss'] = np.where(df_dropped['Title'] =='Miss', 1, 0)
    df_dropped['Title_madame'] = np.where(df_dropped['Title'] =='Madame', 1, 0)
    df_dropped['Title_missus'] = np.where(df_dropped['Title'] =='Missus', 1, 0)
    df_dropped['Title_master'] = np.where(df_dropped['Title'] =='Master', 1, 0)
    df_dropped['Title_rare'] = np.where(df_dropped['Title'] =='Rare', 1, 0)
    df_dropped['Title_madmo'] = np.where(df_dropped['Title'] =='Mademoiselle', 1, 0)
    
    df_dropped = df_dropped.drop(columns = 'Title')

    df_dropped.loc[:,'IsAlone'] = np.where(df_dropped.loc[:,'Parch'] + df_dropped.loc[:,'SibSp'] == 0,1,0)
    df_dropped.loc[:,'FamiliySize'] = df_dropped.loc[:,'Parch'] + df_dropped.loc[:,'SibSp'] + 1

    df_dropped = df_dropped.drop(['Name', 'Parch','SibSp'], axis = 1)
    
    
    label = LabelEncoder()
    df_dropped['AgeBin'] = pd.qcut(df_dropped['Age'], 7)
    df_dropped['AgeBin_Code'] = label.fit_transform(df_dropped['AgeBin'])
    df_dropped.drop(['AgeBin','Age'], 1, inplace=True)
    
    df_dropped['Fare']=df_dropped['Fare'].fillna(df_dropped['Fare'].median())
    df_dropped['FareBin'] = pd.qcut(df_dropped['Fare'], 5)
    df_dropped['FareBin_Code'] = label.fit_transform(df_dropped['FareBin'])
    df_dropped.drop(['Fare','FareBin'], 1, inplace=True)


    df_dropped['isFemale'] = np.where(df_dropped['Sex'] =='female', 1, 0)
    df_dropped['isMale'] = np.where(df_dropped['Sex'] =='male', 1, 0)
    
    df_dropped = df_dropped.drop(columns = 'Sex')
    
    
    df_dropped['Embarked_Q'] = np.where(df_dropped['Embarked'] =='Q', 1, 0)
    df_dropped['Embarked_S'] = np.where(df_dropped['Embarked'] =='S', 1, 0)
    df_dropped['Embarked_C'] = np.where(df_dropped['Embarked'] =='C', 1, 0)
    
    df_dropped = df_dropped.drop(columns = 'Embarked')
    
    
    df_dropped['Pclass1'] = np.where(df_dropped['Pclass'] == 1, 1, 0)
    df_dropped['Pclass2'] = np.where(df_dropped['Pclass'] == 2, 1, 0)
    df_dropped['Pclass3'] = np.where(df_dropped['Pclass'] == 3, 1, 0)
    
    df_dropped = df_dropped.drop(columns = 'Pclass')


    
    return df_dropped
