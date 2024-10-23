# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:13:24 2024

@author: ksn71
"""

import pandas as pd

#csv
df=pd.read_csv("auto-mpg.csv", names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

#Excel
df_excel=pd.read_excel("auto-mpg.xlsx",engine='openpyxl',header=None,sheet_name=None)
#df_excel=pd.read_excel("auto-mpg.xlsx",engine='openpyxl',header=None,sheet_name=['Auto_Mpg','Feature_Info])

#Dictionary
df_excel['Auto_Mpg']
df_excel['Feature_Info']


Auto_Mpg=df_excel['Auto_Mpg']
Auto_Mpg.columns

columns_name=df_excel['Feature_Info']

Auto_Mpg.columns=columns_name[0]


Auto_Mpg.isna().sum()
Auto_Mpg.isna().all()
Auto_Mpg.isna().any()

Auto_Mpg.isna().all(axis=1)
Auto_Mpg.isna().any(axis=1)
Auto_Mpg.isna().sum(axis=1)


df_drop=Auto_Mpg.dropna()

df_drop.reset_index(drop=True,inplace=True)










