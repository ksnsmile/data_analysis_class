# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Exploration - EDA(Exploratory Data Analysis) 
# =============================================================================


## 1. Load dataset ##

# =============================================================================
# Dataset Download:
# - UCI Machine Learning Repository
# - Auto MPG Data Set
# - https://archive.ics.uci.edu/ml/datasets/auto+mpg

# Attribute Information:

# 1. mpg (연비): continuous (연속값)
# 2. cylinders (실린더 수): multi-valued discrete (이산값)
# 3. displacement (배기량): continuous
# 4. horsepower (출력): continuous
# 5. weight (차중): continuous
# 6. acceleration (가속능력): continuous
# 7. model year (출시년도): multi-valued discrete
# 8. origin (제조국): multi-valued discrete (1: USA, 2: EU, 3: JPN)
# 9. car name (모델명): string (unique for each instance)

# =============================================================================

import pandas as pd

# read_csv()
autoMPG_df = pd.read_csv('auto-mpg.csv', header=None)

# Column names
autoMPG_df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']


## 2. Explore data ##

## 2.1 Basic information ##

autoMPG_df.head()
autoMPG_df.head(10)
autoMPG_df.tail()
autoMPG_df.tail(10)

autoMPG_df.columns
autoMPG_df.index

autoMPG_df.shape
# (398 -> row/index, 9 -> column)

# Check basic conditions of columns - data types, missing values
autoMPG_df.info()
 
# Check data types of all columns
autoMPG_df.dtypes
# Check data types of specific columns
autoMPG_df['mpg'].dtypes    # for one column
autoMPG_df[['mpg', 'cylinders']].dtypes    # for multiple columns

# Descriptive statistics
desc = autoMPG_df.describe()    # for numerical data
desc1 = autoMPG_df.describe(include='all')    # for all data
# =============================================================================
# unique : unique value
# top(Mode) : most frequent value
# freq : freqency
# =============================================================================

# Check the number of data
# count() : the number of available data (DataFrame, Series)
autoMPG_df.count()
autoMPG_df.origin.count()
# value_counts() : the number by unique values (Series)
autoMPG_df.origin.value_counts()
autoMPG_df.origin.value_counts(dropna = True)   # Except NaN, dropna = False (default)
# unique() : types of unique values (Series)
autoMPG_df.origin.unique()
# nunique() : the number of types of unique values (DataFrame, Series)
autoMPG_df.nunique()
autoMPG_df.origin.nunique()

# Basic Statistics
# Mean value (DataFrame, Series)
autoMPG_df.mean()   # Only for numeric data, Except horsepower & car_name (String Object)
autoMPG_df.mpg.mean()
# Median value (DataFrame, Series)
autoMPG_df.median()    # Only for numeric data     
autoMPG_df.mpg.median()
# Maximum value (DataFrame, Series)
autoMPG_df.max()    # ASCII for String data
autoMPG_df.mpg.max()
# Minimum value (DataFrame, Series)
autoMPG_df.min()    # ASCII for String data
autoMPG_df.mpg.min()
# Standard deviation (DataFrame, Series)
autoMPG_df.std()    # Only for numeric data
autoMPG_df.mpg.std()

# Correlation coefficient - correlation analysis
# corr() (DataFrame, Series)
# Only for numeric data, Except horsepower & car_name (String Object) 
autoMPG_df.corr()
auto_corr = autoMPG_df.corr()
autoMPG_df[['mpg', 'weight']].corr()
autoMPG_df['mpg'].corr(autoMPG_df['weight'])
abs(auto_corr['mpg']).nlargest(3)

## 2.2 Visualize data ##

import seaborn as sns

# Scatter plot - pairplot()
sns.pairplot(autoMPG_df)
# Scatter plot - plot()
autoMPG_df.plot(x='weight', y='mpg', kind='scatter')

# boxplot
autoMPG_df[['mpg', 'cylinders']].plot(kind='box')


## 3. Pre-processing ##

## 3.1 Pre-processing for data types ##

# Data Types
autoMPG_df.info()

# =============================================================================
#  0   mpg           398 non-null    float64
#  1   cylinders     398 non-null    int64  
#  2   displacement  398 non-null    float64
#  3   horsepower    398 non-null    object   (-> int or float)
#  4   weight        398 non-null    float64
#  5   acceleration  398 non-null    float64
#  6   model_year    398 non-null    int64    (-> category) 
#  7   origin        398 non-null    int64    (-> category)   
#  8   car_name      398 non-null    object 
# =============================================================================

# Column : horsepower
# Types of unique values
autoMPG_df.horsepower.unique()

import numpy as np

# '?' -> NaN
autoMPG_df.horsepower.replace('?', np.NaN, inplace=True)
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna
# DataFrame.dropna : delete rows containing missing values 
autoMPG_df.dropna(subset=['horsepower'], axis=0, inplace=True)   
# string -> float 
autoMPG_df['horsepower'] = autoMPG_df['horsepower'].astype('float') 

autoMPG_df.horsepower.dtypes    # float64

# Column : origin
autoMPG_df.origin.unique()

# 1-> USA, 2 -> EU, 3 -> JAPAN
autoMPG_df['origin'].replace({1:'USA', 2:'EU', 3:'JAPAN'}, inplace=True)
autoMPG_df['origin'].unique()
autoMPG_df['origin'].dtypes

# string -> category
autoMPG_df['origin'] = autoMPG_df['origin'].astype('category')     
autoMPG_df['origin'].dtypes

# category -> string
autoMPG_df['origin'] = autoMPG_df['origin'].astype('str')     
autoMPG_df['origin'].dtypes

# Column : model_year
autoMPG_df['model_year'].sample(3)
autoMPG_df['model_year'] = autoMPG_df['model_year'].astype('category') 

autoMPG_df.info()

## 3.2 Pre-processing for categorical data ##

# Binning 
# horsepower : 'poor', 'good', 'high' 
# np.histogram(input data, the number of bins)
# return : the number of values belonging to each bin & bin_edges
# count : the number of values belonging to each bin & bin_dividers : bin edges
count, bin_dividers = np.histogram(autoMPG_df['horsepower'], bins=3)
print(bin_dividers) 

# pd.cut()
# Use cut() when you need to segment and sort data values into bins. 
# This function is also useful for going from a continuous variable to a categorical variable. 

# https://pandas.pydata.org/docs/reference/api/pandas.cut.html#pandas.cut

bin_names = ['poor', 'good', 'high']

# Append a new cloumn - 'hp_bin'
autoMPG_df['hp_bin'] = pd.cut(x=autoMPG_df['horsepower'],     # target series for binning
                              bins=bin_dividers,              # bin edges
                              labels=bin_names,               # bin names
                              include_lowest=True)            # include left boundary

autoMPG_df[['horsepower', 'hp_bin']].head(15)

# One-hot encoding - dummy variable (0, 1)
# 0 : False, do not exist, 1: True, exist
# get_dummies()

# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies

horsepower_dummies = pd.get_dummies(autoMPG_df['hp_bin'])
horsepower_dummies.head(15)

autoMPG_df1 = pd.concat([autoMPG_df, horsepower_dummies], axis=1)
autoMPG_df1.drop('hp_bin', axis=1, inplace=True)

autoMPG_df = pd.get_dummies(data=autoMPG_df, columns=['hp_bin', 'origin'])
