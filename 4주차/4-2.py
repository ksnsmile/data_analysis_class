# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Warm-up for Data Analysis
# =============================================================================


# 0. Understand the dataset (see Material)


# 1. Load the dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris

from sklearn.datasets import load_iris

iris_dataset = load_iris()  # Bunch class 

print("Iris_dataset keys:\n", iris_dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

print(iris_dataset['DESCR'])   # Describe the iris dataset
iris_descr = iris_dataset.DESCR

print("Target names: ", iris_dataset.target_names)
# Target names: ['setosa' 'versicolor' 'virginica']

print("Feature names: ", iris_dataset.feature_names)
# Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Data : independent variable(X), Target : dependent variable(Y)
print("Data:\n", iris_dataset.data)
# Feature names: ['sepal length (cm)->0', 'sepal width (cm)->1', 'petal length (cm)->2', 'petal width (cm)->3']
type(iris_dataset.data)
iris_dataset.data.shape
iris_dataset.data[:5]

print("Target:\n", iris_dataset.target)
# Target names: ['setosa->0' 'versicolor->1' 'virginica->2']
type(iris_dataset.target)
iris_dataset.target.shape


# 2. Explore data

# 2-1. Generate the dataframe

# =============================================================================
# Dataframe - Indexing & Slicing DataFrame (see Material)
# 
# # List
# data_ex = [['S1', 25, 95],
#            ['S2', 28, 80],
#            ['S3', 22, 75]]
# 
# # Dataframe from list
# df = pd.DataFrame(data_ex,
#                   index=['row1', 'row2', 'row3'],
#                   columns=['Name', 'Age', 'Score'])
# 
# df['Name'] 
# df['Age']
# df['Score']
# df[['Name', 'Score']]
# 
# # row observation 
# df['row1']		# Error
# df.loc['row1']
# df.loc[['row1', 'row3']]
# 
# # row & column observation
# df.loc['row1', 'Name'] 
# df.loc[:, 'Name'] 
# df.loc[:, ['Name', 'Score']] 
# df.loc[:,'Name':'Score']
# 
# df.iloc[0,0]
# df.iloc[:, [0,2]]
# df.iloc[::2, [0,2]] 
# df.iloc[-1,:] 
# df.iloc[-1::-1,:]
# 
# df.head(1) 
# df.head(2)
# df.tail(1) 
# df.tail(2)
# =============================================================================

import pandas as pd

iris_df = pd.DataFrame(iris_dataset.data, columns = iris_dataset.feature_names)    # X-variables
iris_df.head()

iris_df['type'] = iris_dataset.target    ## Y-variable

# Basic information
iris_df.head()
iris_df.tail()
iris_df.shape
iris_df.describe()
iris_df.info()

# Dataframe indexing - row & column
iris_df.iloc[:,-1].value_counts()
iris_df.loc[:,'type'].value_counts()
iris_df['type'].value_counts()

# 2-2. Visualize data

import seaborn as sns

# Scatter plot
sns.pairplot(iris_df)
sns.pairplot(iris_df, hue="type")

# 2-3. Correlation analysis

iris_df_corr = iris_df.corr()
iris_df['type'].corr(iris_df['sepal length (cm)'])
abs(iris_df_corr['type']).nlargest(3)


# 3(1). Data Modeling - kNN

# 3(1)-1. Train & Test dataset

from sklearn.model_selection import train_test_split

X = iris_df.drop(columns=['type'])      # X-variable
Y = iris_df['type']     # Y-variable

# =============================================================================
# X = iris_dataset.data     # X-variable
# Y = iris_dataset.target    # Y-variable
# =============================================================================

# train dataset = 70%, test dataset = 30%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# 3(1)-2. Generate the Model

from sklearn.neighbors import KNeighborsClassifier

# knn : instance of KNeighborsClassifier Class
knn = KNeighborsClassifier(n_neighbors=1)

# 3(1)-3. Train the model

# fit(x, y) => train datasets
knn.fit(X_train, Y_train)

# 3(1)-4. Predict the Y-variable

# predict(x) => test datasets
Y_predict = knn.predict(X_test)


# 4(1). Evaluate the analysis results - kNN

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_predict)
print("kNN algorithm accuracy: %4.2f" % accuracy)
# kNN algorithm accuracy: 0.98

import numpy as np
accuracy_cal = np.mean(Y_test == Y_predict)
print("kNN algorithm accuracy: %4.2f" % accuracy_cal)
# kNN algorithm accuracy: 0.98


# 3(2). Data Modeling - decision tree

# 3(2)-1. Generate the Model

from sklearn.tree import DecisionTreeClassifier

# dc_t : instance of DecisionTreeClassifier Class
dc_t = DecisionTreeClassifier(criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1, max_features=None)

# 3(2)-2. Train the model

# fit(x, y) => train datasets
dc_t.fit(X_train, Y_train)

# 3(2)-3. Predict the Y-variable

# predict(x) => test datasets
Y_predict = dc_t.predict(X_test)


# 4(2). Evaluate the analysis results - decision tree

accuracy = accuracy_score(Y_test, Y_predict)
print("decision tree algorithm accuracy: %4.2f" % accuracy)
# decision tree algorithm accuracy: 0.98
