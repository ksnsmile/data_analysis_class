# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Pre-processing
# =============================================================================


## 1. Load dataset ##

import numpy as np
import pandas as pd

pd.read_csv('BreastCancer.csv')

bc=pd.read_csv('BreastCancer.csv')

# Basic information
bc.columns
bc.index
bc.values


## 2. Handle missing data

# 2.1 Check missing data

# True = NA = NaN
# bc.isna() = bc.isnull()
bc.isna() 

# True(1) - missing data, False(0) - normal data
bc.isna().sum()

# True(1) - normal data, False(0) - missing data
bc.notna()

# True - variable that all data is NaN
bc.isna().all() 

# True - variable with at least one NaN
# Column : Bare.nuclei  
bc.isna().any()

# DataFrame Axis = 0 & Axis = 1 (see Material)

# True - row(index) where all data is NaN
bc.isna().all(axis=1) 

# True - row(index) with at least one NaN
bc.isna().any(axis=1)

# Check the location of missing data
bc[bc.isna().any(axis=1)] 

# Error
# bc[bc.isna().any()]

# Column names with missing data  
bc.columns[bc.isna().any()]

# 2.2 Remove missing data

# dropna()
# bc.dropna(inplace=True) => bc=bc.dropna()
bc_na_drop=bc.dropna() 


# Remove missing data for certain column/variable
# bc.dropna(subset=['column name'])

# 2.3 Handle missing data (fillna())

# =============================================================================
# How to handle missing data :  
# 1) Single value [0]
# 2) Different values per column
# 3) Different average values per column 
# 4) Reflecting the characteristics of each group
# 5) Applying machine learning algorithms 
# =============================================================================

# 1) Fill NaN with single value [0]

bc_fillna0=bc.fillna(0)

# Column names with missing data 
bc.columns[bc.isna().any()].values

# Check the location of missing data in the bc DataFrame
bc[bc['Bare.nuclei'].isna()] 
bc[bc['Bare.nuclei'].isna()]['Bare.nuclei']

# Check what value the missing data is changed to 
# Fill NaN with 0.0
bc_fillna0[bc['Bare.nuclei'].isna()]['Bare.nuclei']


# 2) Fill NaN with different values per column

# Use dictionary
# {'column name':4, 'column name':7}

# bc_fillna0=bc.fillna({'Bare.nuclei':10})
# bc_fillna0[bc['Bare.nuclei'].isna()]['Bare.nuclei']
# bc_fillna0=bc.fillna(0)


# 3) Fill NaN with different average values per column 

bc.mean()
# Id                 1.071704e+06
# Cl.thickness       4.417740e+00
# Cell.size          3.134478e+00
# Cell.shape         3.207439e+00
# Marg.adhesion      2.806867e+00
# Epith.c.size       3.216023e+00
# Bare.nuclei        3.544656e+00
# Bl.cromatin        3.437768e+00
# Normal.nucleoli    2.866953e+00
# Mitoses            1.589413e+00

bc_fillna_mean=bc.fillna(bc.mean()) 

# Check what value the missing data is changed to 
bc_fillna_mean[bc['Bare.nuclei'].isna()]['Bare.nuclei']

# 4) Fill NaN by reflecting the characteristics of each group

# Check if there are group-specific features in the dataset 
bc.columns

# Use groupby()
# Check the average value of each group
bc_group_mean=bc.groupby('Class').mean()

# Use apply() & lambda
bc_fillna_group_mean=\
    bc.groupby('Class').apply(lambda x: x.fillna(x.mean()))

# multiple indexes (Class+Index)   
# Remove unnecessary labels, droplevel()
bc_fillna_group_mean=bc_fillna_group_mean.droplevel(0)

# Check what value the missing data is changed to 
bc_fillna_group_mean[bc['Bare.nuclei'].isna()]['Bare.nuclei']


## 3. Handle outliers


# =============================================================================
# Check for outliers before handling missing values
# Understand data characteristics through data exploration -> descriptive statistics
# If outliers -> Remove outliers and find out values to fill the NaN
# If there is a similar pattern around the data -> Apply machine learning algorithm 
# 
# Exploration => Pre-processing => Good Quality Data 
# Good Quality Data => Good Information
# =============================================================================


# 3.1 Exploration (descriptive statistics)

bc.columns
# drop(index or column name)
# Remove non-numeric variables 
bc_drop_id=bc.drop(columns=['Id','Class'])
bc_drop_id.columns

# see Material
bc_des=bc.describe()

# boxplot graph : using quartiles for checking outliers
# see Material

# with 'ID'
bc.plot(kind='box')
# without 'ID'
bc_drop_id.plot(kind='box')

# 3.2 Check outliers by a user-defined function

# Module : define a function in a new file
# Save the file as ft_col.py

from ft_col import outlier_test
# from ft_col import *

# True : outlier
outlier_result=bc_drop_id.apply(outlier_test)

# apply() : see Material
# bc_drop_id.apply(outlier_test) -> DF.apply(Series)

# Check the number of outliers for each object/index/row/record
# True(1) : outlier, False(0) : normal data
outlier_result.sum(axis=1)

# Outlier criteria : two or more values determined as outliers (up to one is allowed) 
# Normal objects : 611
(outlier_result.sum(axis=1) < 2).sum()
outlier_result.sum(axis=1).value_counts()  
# 0    504
# 1    107
# 2     54
# 3     23
# 4      8
# 5      3

# 3.3 Remove outliers

# Save normal data
bc_outlier_drop=bc[outlier_result.sum(axis=1) < 2]

# Raw Dataset
bc.Class.value_counts()
# benign       458
# malignant    241

# Dataset without outliers
# Check the number of Class 'malignant'
bc_outlier_drop.Class.value_counts()
# benign       458
# malignant    153

# Check outliers by group

bc_drop_id=bc.drop(columns=['Id'])

# Add groupby() to outlier_result=bc_drop_id.apply(outlier_test) 
outlier_result_group=bc_drop_id.groupby('Class').apply(outlier_test)

# Check the number of outliers for each object/index/row/record
# True(1) : outlier, False(0) : normal data
outlier_result_group.sum(axis=1)

bc_outlier_drop_group=bc[outlier_result_group.sum(axis=1) < 2]

# Check the number of dropped objects using the len() function
len(bc_outlier_drop_group)
# 590

bc_outlier_drop_group.Class.value_counts()
# benign       349
# malignant    241

# Results analysis

# Raw Dataset : 
# benign       458
# malignant    241

# Dataset without outliers
# benign       458
# malignant    153

# Dataset without outliers (consider groups)
# benign       349
# malignant    241

# 3.4 Handle Missing Values with Outlier-Removed Dataset 

# Check the number of missing data

# Before removing outliers
bc.isna().sum() 

# After removing outliers
bc_outlier_drop_group.isna().sum()

# Fill NaN by reflecting the characteristics of each group 
# Use apply() & lambda
bc_pre=bc_outlier_drop_group.groupby('Class').apply(lambda x:
                                                    x.fillna(x.mean()))

# multiple indexes (Class+Index)   
# Remove unnecessary labels, droplevel()
bc_pre=bc_pre.droplevel(0)


bc_pre[bc_outlier_drop_group.isna().any(axis=1)]
# After removing outliers (groupby() result)
bc_pre[bc_outlier_drop_group.isna().any(axis=1)]['Bare.nuclei'] 
# 139    1.150442
# 145    1.150442
# 164    1.150442
# 235    1.150442
# 249    1.150442
# 275    1.150442
# 294    1.150442
# 321    1.150442
# 411    1.150442
# 617    1.150442
# 23     7.627615
# 292    7.627615

# Before removing outliers (groupby() result)
bc_fillna_group_mean[bc['Bare.nuclei'].isna()]['Bare.nuclei'] 
# 40     1.346847
# 139    1.346847
# 145    1.346847
# 158    1.346847
# 164    1.346847
# 235    1.346847
# 249    1.346847
# 275    1.346847
# 294    1.346847
# 297    1.346847
# 315    1.346847
# 321    1.346847
# 411    1.346847
# 617    1.346847
# 23     7.627615
# 292    7.627615






