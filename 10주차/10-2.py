# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Analysis(2)
# =============================================================================

import pandas as pd
import os

# listgdir() : Return the file names in the path
# Relative path
os.listdir('.') 
# Save the file names to file_list
file_list=os.listdir('.')

## 1. Load dataset ##

bc=pd.read_csv('BreastCancer.csv')


## 2. Pre-processing - Refer to similar data ##

# =============================================================================
# Supervised learning : KNN
# 
# What to do
# 
# 1. Dataset
# - train : reference data (X : independent variable, Y : dependent variable [target])
# - test : missing data (X : independent variable, Y : dependent variable[missing data])
# 
# 2. Learning 
# KNN algorithm (regression/classification) (see Material)
# Euclidean distance : sklearn default option
# 
# https://scikit-learn.org/stable/
# neighbors -> algorithms that refer to nearby neighbors  
# neighbors.KNeighborsRegressor : regression 
# neighbors.KNeighborsClassifier : classification
# =============================================================================

from sklearn.neighbors import KNeighborsRegressor 


## 2.1 Dataset ##

# Check missing data
bc.isna().sum()

# Check column name
bc.columns
# ['Id', 'Cl.thickness', 'Cell.size', 'Cell.shape', 'Marg.adhesion',
#        'Epith.c.size', 'Bare.nuclei', 'Bl.cromatin', 'Normal.nucleoli',
#        'Mitoses', 'Class']

# Dataset without missing data
train=bc[bc['Bare.nuclei'].notna()] 

# Dataset with missing data
test=bc[bc['Bare.nuclei'].isna()] 


## 2.2 Learning ##

# 3-NN : n_neighbors=3
knn=KNeighborsRegressor(n_neighbors=3) 

# fit(x,y) -> train data
# x : only numerical data -> drop ID, Class
# y : Bare.nuclei
knn.fit(train.drop(columns=['Bare.nuclei', 'Id', 'Class']), 
        train['Bare.nuclei'])

bc_knn=bc.copy()

# Check the y variable (missing data)
bc_knn.loc[bc_knn['Bare.nuclei'].isna(), 'Bare.nuclei']

# predict(x) -> test data
bc_knn.loc[bc_knn['Bare.nuclei'].isna(), 'Bare.nuclei']=\
    knn.predict(test.drop(columns=['Bare.nuclei', 'Id', 'Class'])) 
 
# no missing data
bc_knn.loc[bc_knn['Bare.nuclei'].isna(), 'Bare.nuclei']

# Result
bc_knn.loc[bc['Bare.nuclei'].isna(), 'Bare.nuclei']

# Check missing data
bc_knn.isna().sum()
    
dir(knn)

# train dataset
# x[0] -> 3 nearest neighbors : [197, 521,  95] distance : [0., 0., 0.]
knn.kneighbors()


# =============================================================================
# Additional information
# 
# PYPI -> https://pypi.org/ ->  python packages homepage
# : verified packages 
# 
# Github -> https://github.com/
# : various packages
# =============================================================================


## 2. Pre-processing - Refer to similar data ##

## 2.1 Dataset ##
# Same dataset as KNN 


## 2.2 Learning ##

# Supervised learning : decision-tree (see Material)

# Visualization : export_graphviz
from sklearn.tree import DecisionTreeRegressor, export_graphviz

# Class instance
dt_r=DecisionTreeRegressor()

# fix(x,y)
X_train = train.drop(columns=['Bare.nuclei', 'Id', 'Class'])
Y_train = train['Bare.nuclei']  
                 
dt_r.fit(X_train,Y_train)

bc_dt=bc.copy()

# Check missing data
bc_dt.loc[bc_dt['Bare.nuclei'].isna(), 'Bare.nuclei']

# predict(x)
bc_dt.loc[bc_dt['Bare.nuclei'].isna(), 'Bare.nuclei']=\
    dt_r.predict(test.drop(columns=['Bare.nuclei', 'Id', 'Class'])) 

# Decision-tree result
bc_dt.loc[bc['Bare.nuclei'].isna(), 'Bare.nuclei']
# 23     10.000000
# 40     10.000000
# 139     1.000000
# 145     1.000000
# 158     1.000000
# 164     1.000000
# 235     1.000000
# 249     1.250000
# 275     1.000000
# 292     6.000000
# 294     1.045455
# 297     3.000000
# 315     6.000000
# 321     1.250000
# 411     1.000000
# 617     1.000000
 
# KNN result
bc_knn.loc[bc['Bare.nuclei'].isna(), 'Bare.nuclei']
# 23      6.666667
# 40      9.333333
# 139     1.000000
# 145     1.000000
# 158     1.333333
# 164     1.000000
# 235     1.000000
# 249     2.000000
# 275     1.000000
# 292     6.000000
# 294     1.333333
# 297     1.666667
# 315    10.000000
# 321     2.000000
# 411     1.000000
# 617     1.000000

## 3. Visualization : export_graphviz ##


# Install graphviz-2.38 
# https://graphviz.org/download/
 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz   
# export_graphviz(instance object, file name)
# gveedit.ext
export_graphviz(dt_r, 'tree.dot')

# max_depth = 3 : visualize up to depth 3
# feature_names : variable name, train.columns.drop(['Bare.nuclei','Id','Class'])
# gveedit.ext
export_graphviz(dt_r, 'tree.dot', max_depth=3,
           feature_names=train.columns.drop(['Bare.nuclei','Id','Class']))





















