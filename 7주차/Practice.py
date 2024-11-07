# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Practice
# =============================================================================


import pandas as pd

# CSV
autoMPG_df_Ex = pd.read_csv('auto-mpg.csv', names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Excel
autoMPG_df = pd.read_excel('auto-mpg.xlsx', engine='openpyxl', sheet_name=None, header=None)
# autoMPG_df = pd.read_excel('auto-mpg.xlsx', engine='openpyxl', sheet_name=['Feature_Info','Auto_Mpg'], header=None)

# Dictionary
autoMPG_df['Auto_Mpg']
autoMPG_df['Feature_Info']

# Save Auto_Mpg DataFrame separately
autoMPG_df_Final = autoMPG_df['Auto_Mpg']

# column name
autoMPG_df_Final.columns
autoMPG_df_Final.columns = autoMPG_df['Feature_Info'][0] 
# type(autoMPG_df['Feature_Info'][0])
# rename()

# Check missing data
# For column 
autoMPG_df_Final.isna()
autoMPG_df_Final.isna().sum()
autoMPG_df_Final.isna().all() 
autoMPG_df_Final.isna().any() 

# For row
autoMPG_df_Final.isna().sum(axis=1)
autoMPG_df_Final.isna().all(axis=1) 
autoMPG_df_Final.isna().any(axis=1) 

# dropna
autoMPG_df_Final_drop=autoMPG_df_Final.dropna() 
autoMPG_df_Final.dropna(inplace=True) 

# Reset index number
autoMPG_df_Final.reset_index(drop=True, inplace=True)

# Basic investigation
autoMPG_df_Final.columns
autoMPG_df_Final.index
autoMPG_df_Final.values

autoMPG_df_Final.head()
autoMPG_df_Final.tail()
autoMPG_df_Final.shape

autoMPG_df_Final.info()
autoMPG_des = autoMPG_df_Final.describe()

# Indexing & slicing
# Handle missing data
# Handle outliers

# Standardization
from sklearn.preprocessing import StandardScaler

autoMPG_df_temp = autoMPG_df_Final.drop(columns=['model_year', 'origin', 'car_name'])

scaler = StandardScaler()

scaler.fit(autoMPG_df_temp)
scaler.mean_
scaler.var_

autoMPG_df_temp_scaled = scaler.transform(autoMPG_df_temp)

autoMPG_df_temp_scaled_df = pd.DataFrame(autoMPG_df_temp_scaled, columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']) 
autoMPG_df_temp_scaled_df.mean()
autoMPG_df_temp_scaled_df.std()

autoMPG_df_concat = pd.concat([autoMPG_df_temp_scaled_df, autoMPG_df_Final[['model_year', 'origin', 'car_name']]], axis=1)

# Normalization
from sklearn.preprocessing import MinMaxScaler

scaler_n = MinMaxScaler()
autoMPG_df_temp_scaled_n = scaler_n.fit_transform(autoMPG_df_temp)
autoMPG_df_temp_scaled_n_df = pd.DataFrame(autoMPG_df_temp_scaled_n, columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']) 
autoMPG_df_temp_scaled_n_df.min()
autoMPG_df_temp_scaled_n_df.max()

autoMPG_df_n_concat = pd.concat([autoMPG_df_temp_scaled_n_df, autoMPG_df_Final[['model_year', 'origin', 'car_name']]], axis=1)

# Export excel
with pd.ExcelWriter("auto-mpg.xlsx", mode="a", engine="openpyxl") as writer:
    autoMPG_des.to_excel(writer, sheet_name="Statistics")
    autoMPG_df_concat.to_excel(writer, sheet_name="Standardization", index=False)
    autoMPG_df_n_concat.to_excel(writer, sheet_name="Normalization", index=False)
