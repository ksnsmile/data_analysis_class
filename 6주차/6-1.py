# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Exploration - Standardization & Normalization
# =============================================================================


## 1. Load dataset ##

import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()

# X-variables
iris_df = pd.DataFrame(iris_dataset.data, columns = iris_dataset.feature_names) 

# Before feature scaling - Standardization, Normalization
iris_df.mean()
# =============================================================================
# sepal length (cm)    5.843333
# sepal width (cm)     3.057333
# petal length (cm)    3.758000
# petal width (cm)     1.199333
# =============================================================================

iris_df.std()
# =============================================================================
# sepal length (cm)    0.828066
# sepal width (cm)     0.435866
# petal length (cm)    1.765298
# petal width (cm)     0.762238
# =============================================================================


## 2. Standardization ##

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler

from sklearn.preprocessing import StandardScaler

# scaler : instance of the StandardScaler Class
scaler = StandardScaler()

# Compute the mean and std to be used for later scaling
scaler.fit(iris_df)

# Perform standardization by centering and scaling
iris_scaled = scaler.transform(iris_df)

# Fit to data, then transform it
# iris_scaled_ft = scaler.fit_transform(iris_df)

# Scaled DataFrame
iris_scaled_df = pd.DataFrame(data=iris_scaled, columns = iris_dataset.feature_names)

iris_scaled_df.mean()
# =============================================================================
# sepal length (cm)   -1.690315e-15
# sepal width (cm)    -1.842970e-15
# petal length (cm)   -1.698641e-15
# petal width (cm)    -1.409243e-15
# =============================================================================

iris_scaled_df.std()
# =============================================================================
# sepal length (cm)    1.00335
# sepal width (cm)     1.00335
# petal length (cm)    1.00335
# petal width (cm)     1.00335
# =============================================================================


## 3. Normalization ##

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

# scaler : instance of the MinMaxScaler Class
scaler = MinMaxScaler()

scaler.fit(iris_df)

iris_scaled_minmax = scaler.transform(iris_df)

iris_scaled_minmax_df = pd.DataFrame(data=iris_scaled_minmax, columns = iris_dataset.feature_names)

iris_scaled_minmax_df.min() # Minimum values = 0
# =============================================================================
# sepal length (cm)    0.0
# sepal width (cm)     0.0
# petal length (cm)    0.0
# petal width (cm)     0.0
# =============================================================================

iris_scaled_minmax_df.max() # Maximum values = 1
# =============================================================================
# sepal length (cm)    1.0
# sepal width (cm)     1.0
# petal length (cm)    1.0
# petal width (cm)     1.0
# =============================================================================

iris_scaled_minmax_df.describe()
