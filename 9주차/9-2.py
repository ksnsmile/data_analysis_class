# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Analysis(1)
# =============================================================================

import numpy as np
import pandas as pd


## 1. Load dataset ##

bc=pd.read_csv('BreastCancer.csv')


# =============================================================================
# 5) Applying machine learning algorithms
# 
# Types of machine learning
#  
# Types of algorithm application for handling missing values
# 
# (1) Use relationships among data
#     - Supervised Learning : regression(y-numerical value) , classification(y-categorical value))
#       * linear regression/non-linear regression
# (2) Refer to similar data
#     - Supervised Learning : regression(y-numerical value) , classification(y-categorical value))
#       * tree, knn(find surrounding data)
# 
# scikit-learn.org : https://scikit-learn.org/stable/
# =============================================================================

# Train/Validation/Test dataset (see Material)

# In our case (do not evaluate) :
# (Train datset) x : no missing data, y : no missing data  
# (Test dataset) x : no missing data, y : missing data


## 2. Pre-processing - Use relationships among data ##
# Supervised learning : liner regression

# seaborn package
import seaborn as se

## 2.1 Correlation analysis for configuring dataset ##

# drop() : ID & Class
# corr() : correlation analysis -> calculation of correlation between variables 
bc_corr=bc.drop(columns=['Id', 'Class']).corr()

# Visualization of correlation analysis
# pairplot() : Scatterplot
se.pairplot(bc.drop(columns=['Id', 'Class']))

# Including regression line
se.pairplot(bc.drop(columns=['Id', 'Class']), kind="reg")

# Including density plot
# kde : Kernel Density Estimation
se.pairplot(bc.drop(columns=['Id', 'Class']), kind="reg", diag_kind="kde") #데이터를 좀 더 부드럽게 보여줄 수 있다 히스토그램 부분 없애고

# Correlation coefficients of other variables related to Bare.nuclei 
bc_corr['Bare.nuclei']

# abs() : absolute value function 
# nlargest(n) : Find the n largest values 
abs(bc_corr['Bare.nuclei']).nlargest(3)
# Bare.nuclei    1.000000
# Cell.shape     0.713878
# Cell.size      0.691709

abs(bc_corr['Bare.nuclei']).nlargest(3).index

var_list=abs(bc_corr['Bare.nuclei']).nlargest(3).index

## 2.2 Dataset configuration for training ##

# Variable with missing data : Bare.nuclei -> y
# Variables highly correlated with Bare.nuclei : Cell.shape, Cell.size -> x

# (Train datset) x : no missing data, y : no missing data  
# loc() : cell-by-cell execution 
train=bc.loc[bc['Bare.nuclei'].notna(), var_list]
# (Test dataset) x : no missing data, y : missing data
test=bc.loc[bc['Bare.nuclei'].isna(), var_list]

# =============================================================================
# DataFrame data access : 
#     
# (1) bc['variable name'] : correct name,
#                           batch execution is not possible with cell-by-cell access 
# (2) bc.variable name : correct name,
#                        batch execution is not possible with cell-by-cell access 
# (3) bc.loc['index name', 'column name'] :
#                                       batch execution possible by accessing the cell unit 
# (4) bc.iloc[index position, column position] :
#                                       batch execution possible by accessing the cell unit 
# =============================================================================

# Check missing data
train.isna().sum()
test.isna().sum()

## 2.3 Learning (Define the algorithm/model) ##

# sklearn : scikit-learn 
# linear regression (see Material)
from sklearn.linear_model import LinearRegression

# ols : instance of LinearRegression Class
ols=LinearRegression()

# fit(x,y) method (supervised learning) : Learn rules/patterns -> fitting
# fit(x,y) -> train dataset
# x variable (Cell.shape, Cell.size): iloc[all rows, column 1~end]
# y variable (Bare.nuclei): iloc[all rows, column 0]
ols.fit(train.iloc[:,1:], train.iloc[:,0]) 

# List of available properties/functions 
dir(ols) 

# y=a+b1*x1+b2*x2
# a : constant
ols.intercept_
# b1, b2 : coefficient
ols.coef_ 

# y=0.6990081320249253+0.5949618*X1+0.29602292*X2

# predict() : prediction using the model created by the fit() method
# predict(x) : test dataset
ols.predict(test.iloc[:,1:])

# copy() 
bc_fillna_da = bc.copy()

# fill missing data with the predicted result
bc_fillna_da.loc[bc['Bare.nuclei'].isna(), 'Bare.nuclei']=ols.predict(test.iloc[:,1:])

# Check missing data
bc[bc['Bare.nuclei'].isna()]['Bare.nuclei']

# Result
bc_fillna_da[bc['Bare.nuclei'].isna()]['Bare.nuclei']
# 23     4.857909
# 40     6.044916
# 139    1.589993
# 145    2.779916
# 158    2.184955
# 164    1.589993
# 235    3.374878
# 249    1.589993
# 275    2.779916
# 292    7.826886
# 294    1.589993
# 297    3.667985
# 315    5.449955
# 321    1.589993
# 411    1.589993
# 617    1.589993
