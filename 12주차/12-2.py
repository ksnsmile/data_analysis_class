# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Practice of Manufacturing Data (1)
# =============================================================================


## 1. Load dataset ##


# 1) Load the tdms file (.tdms)

# TDMS (Technical Data Management Streaming File) : 
# The NI TDMS file format is an NI platform-supported file format. 
# All NI software development environments interface with TDMS files.
# Link : https://www.ni.com/ko-kr/support/documentation/supplemental/06/the-ni-tdms-file-format.html


# npTDMS package : 
# https://nptdms.readthedocs.io/en/stable/


# Install npTDMS :
# !pip install npTDMS


# Import nptdms.TdmsFile
from nptdms import TdmsFile


# read(file name) : data loading 
tdms_file=TdmsFile.read('1_200714_150955_Reference.tdms')

dir(tdms_file)


# 2) Understand the file structure

# Check groups in the file 
tdms_file.groups()

# Save the groups
all_group=tdms_file.groups()

# Check the 1st group name - access the data in the List 
all_group[0].name

# Check what functions/attributes are available at the group level  
dir(all_group[0])


# Check channels in the group (all_group[0] => 1st group)
all_group[0].channels()

# Save the chennels
channels=all_group[0].channels()

# Check what functions/attributes are available at the chennel level
dir(channels[0])

# Check the 1st chennel name
channels[0].name     # Current Block

# data -> array type
channels[0].data

# Check what functions/attributes are available at the tdms file level
dir(tdms_file)

tdms_file_dataframe=tdms_file.as_dataframe()


## 2. Basic pre-processing ##


# 1) Check variable names

# Checek the current status
tdms_file_dataframe.columns

# Pandas library : https://pandas.pydata.org/
# Series.str.replace() : replace string 

# "/" => "_"
tdms_file_dataframe.columns=tdms_file_dataframe.columns.str.replace("/","_")
# Check the result
tdms_file_dataframe.columns

# " " => "_"
tdms_file_dataframe.columns=tdms_file_dataframe.columns.str.replace(" ","_")
# Check the result
tdms_file_dataframe.columns

# "'" => "" => remove
tdms_file_dataframe.columns=tdms_file_dataframe.columns.str.replace("'","")
# Check the result
tdms_file_dataframe.columns

# "_" in the front
# slice(1,) : from 1 to the end
tdms_file_dataframe.columns=tdms_file_dataframe.columns.str.slice(1,)
# Check the result
tdms_file_dataframe.columns


# 2) Handle missing data

# Check the data type by checking only the top 5 data for each variable 
tdms_file_dataframe.head()

# Check for meaningless/unusable variables filled with only missing values  
tdms_file_dataframe.isna().all()

# Check location of meaningless variables
tdms_file_dataframe.columns[tdms_file_dataframe.isna().all()]
na_var_list=tdms_file_dataframe.columns[tdms_file_dataframe.isna().all()]

# Check the top 5 data => NaN
tdms_file_dataframe[na_var_list].head()

# Dataset without missing data (all)
tdms_df=tdms_file_dataframe.drop(columns=na_var_list)


# 3) Check data types

tdms_df.dtypes
tdms_df.dtypes.value_counts()
# float64    234
# object       1

# What to do
# Numerical data -> check whether numeric categorical data or not
# Object data : determine whether to be included in analysis
# If included -> conversion using One-Hot, Dummy methods


# In our dataset,
# No numeric categorical data
# Object data :

ob_list=tdms_df.columns[tdms_df.dtypes=='object']

# Check the top 5 data
tdms_df[ob_list].head()

# Check unique values 
tdms_df[ob_list[0]].unique()

tdms_df[ob_list[0]].value_counts()

# Remove the variable of object type  
tdms_df=tdms_df.drop(columns=ob_list)
tdms_df.dtypes.value_counts()
# float64    234


## 3. Data exploration ##


# Check variables consisting only of equal values(meaningless data) : variance or standard deviation 
# var() : calculate the variance
tdms_df.var()

# 38 data : variance == 0.0
const_var_list=tdms_df.columns[tdms_df.var()==0.0]

# Dataset without 38 data
tdms_df=tdms_df.drop(columns=const_var_list)
# 294-59-1-38=196


# 1) Correlation Analysis

# Analysis Target : Predict cutting force
# Dependent variables, Y : Cutting_Force_1, Cutting_Force_2, Cutting_Force_3
# Independent variables, X : 
# Find variables that are highly correlated with Cutting_Force 

tdms_corr=tdms_df.corr()

# Visualization
import seaborn as se

# heatmap()
se.heatmap(tdms_corr)

# Absolute value 
se.heatmap(abs(tdms_corr))


# Index : 100, 101, 102 (Cutting_Force_1,_2,_3)
var_list=tdms_df.columns
cut_force=tdms_df.columns[100:103]

# Correlation coefficients related to Cutting_Force_1,_2,_3 
tdms_corr[cut_force]

# 10 variables that are highly correlated with Cutting_Force_1
abs(tdms_corr['Raw_Cutting_Force_1']).nlargest(10)

# Apply the function applied to Raw_Cutting_Force_1 up to Raw_Cutting_Force_1,_2,_3 
# Index list only (variable names) 
cut_force_corr_var_list=abs(tdms_corr[cut_force]).apply(lambda x: x.nlargest(10).index)
# Value list only (values) 
cut_force_corr_var_values=abs(tdms_corr[cut_force]).apply(lambda x: x.nlargest(10).values)


# What to do
# (1) Check outliers
# (2) If outliers exist, use models less sensitive to outliers [reference models] 


## 4. Advanced pre-processing ##


# 1) Handle outliers

# Copy()
data=tdms_df.copy()

# Variables without missing data
data.notna().all()

notna_var_list=data.columns[data.notna().all()] 
# 48 data
len(notna_var_list) 


# Check outliers
# Pre-defined function
def outlier_test(x):
    Q1=x.quantile(1/4)
    Q3=x.quantile(3/4)
    IQR=Q3-Q1
    LL=Q1-(1.5*IQR)
    UU=Q3+(1.5*IQR)
    outlier=(x < LL) | (x > UU)
    return outlier

# Apply the outlier_test() to data (DataFrame)
outlier_return=data.apply(outlier_test)

# True : outliers, False : normal
outlier_return.head()

# Check the number of outliers by variable (column level) 
# True[1], False[0]
outlier_return.sum()  

# The number of outliers
outlier_return.sum().value_counts()

# Outlier ratio to total 
out_f=(outlier_return.sum()/len(data))*100

# Sorting
out_f.value_counts().sort_index()


# Check the number of outliers by index (row level) 
# True[1], False[0]
outlier_return.sum(axis=1)  
outlier_return.sum(axis=1).value_counts() 

# Indexes with 12 or more outliers in a row 
data_outlier_search=data[outlier_return.sum(axis=1) >= 12]


## 2) Re-handling missing data


# Still many missing values?
# Separate the variables to remove NaN and the variables to fill NaN 
# Remove NaN if the NaN ratio is high 
# Fill NaN if the NaN ratio is low


len(data) # 412160

# Variables with a missing value ratio of more than 80% => 148
(data.isna().sum() > (len(data) * 4/5))
(data.isna().sum() > (len(data) * 4/5)).sum()


# Read the tdms file : 
# 1. Load whole data
# 2. Load data in groups and channels
# => Create a DataFrame for each group
# => Define keys to connect the DataFrames to each other => Final Dataset
#    - Time, Identifier, Line-ID, etc.


# Total dataset :
# Total variables(196) - variables with a missing value ratio of more than 80%(148) = 48
len(data.columns)-(data.isna().sum() > (len(data) * 4/5)).sum()

len(notna_var_list) #48


# 3) Remove outliers

# Re-check outliers for variables without NaN
outlier_return=data[notna_var_list].apply(outlier_test)

outlier_return.sum().value_counts() 

# Dataset with outliers
data_pre=data[notna_var_list]  

# Check records with many outliers.
outlier_return.sum(axis=1).value_counts()
# 2     79316
# 3     73857
# 4     62802
# 1     52470
# 0     45740
# 5     45039
# 6     26935
# 7     14821
# 8      7064
# 9      2777
# 10     1010
# 11      261
# 12       60
# 13        7
# 14        1

# Dataset without outliers (2~10)
data_pre2=data_pre[outlier_return.sum(axis=1)<11]


## 5. Feature engineering (feature extraction & feature selection)


# Dataset
# X : 45
# Y : 3


# 1) Apply Random Forest 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Dataset #

data_pre.columns

# Define X, Y variables
X=data_pre.iloc[:,3:]
Y=data_pre.iloc[:,:3]

# random_state=100
tr_x, te_x, tr_y, te_y=train_test_split(X, Y,
                                        test_size=0.3,
                                        random_state=100)

# Instance of RandomForestRegressor
# n_estimators : the number of trees in the forest, default = 100
# max_depth : the maximum depth of the tree, default = None
cut_model1=RandomForestRegressor(n_estimators=10, max_depth=5) 


# Learn/Train #

# Total y : cut_model1.fit(tr_x, tr_y)
# Model by each y
cut_model1.fit(tr_x, tr_y.iloc[:,0])

dir(cut_model1)

# estimators_ : the number of trees
cut_model1.estimators_

cut_model1.estimators_[0]

dir(cut_model1.estimators_[0])

# The importance of variables
cut_model1.estimators_[0].feature_importances_
cut_model1.estimators_[1].feature_importances_
cut_model1.estimators_[2].feature_importances_
cut_model1.feature_importances_


# Predict #

# test set
cut1_pred=cut_model1.predict(te_x)


# Evaluate # 


from sklearn.metrics import mean_squared_error

# test set
# mean_squared_error(Actual, Predicted)
mean_squared_error(te_y.iloc[:,0],cut1_pred) # 9134.675592025087


# 2) Apply RFE & Random Forest (feature selection)

# RFE(Recursive Feature Elimination) :
# => A very intuitive feature selection method
# => Remove unimportant features after training for the entire feature, leaving only the desired number of features
# => minimal optimal feature subset

from sklearn.feature_selection import RFE

# Instance
RFE_cut_model1=RFE(cut_model1)

# fit()
RFE_cut_model1.fit(tr_x, tr_y.iloc[:,0])

# predict()
cut1_pred2=RFE_cut_model1.predict(te_x)

# Evaluate
mean_squared_error(te_y.iloc[:,0],cut1_pred2) # 9144.43556684862


dir(RFE_cut_model1)

# Check the order of variable removal => ranking_
# Select only variables with a value of 1
# Remove features in the order of the largest value (24, 23, 22 ~~~)
RFE_cut_model1.ranking_


import pandas as pd

# Save the order
remove_var_rank=pd.Series(RFE_cut_model1.ranking_, index=tr_x.columns)

# Sorting => Descending 
remove_var_rank.sort_values(ascending=False)


# n_features_to_select=1
RFE_cut_model1=RFE(cut_model1, n_features_to_select=1)
RFE_cut_model1.fit(tr_x, tr_y.iloc[:,0])
cut1_pred3=RFE_cut_model1.predict(te_x)
mean_squared_error(te_y.iloc[:,0],cut1_pred3) # 9772.742899570363


