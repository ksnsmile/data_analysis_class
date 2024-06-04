"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Regression - Linear Regression
# =============================================================================


# 1. Load datasets

import pandas as pd
import numpy as np

from sklearn.datasets import load_boston

# =============================================================================
# https://scikit-learn.org/stable/
# scikit-learn : sklearn
# =============================================================================

## 1-1. Load the boston dataset from sklearn 

boston = load_boston()
print(boston.DESCR)   # Describe the boston dataset
boston_descr = boston.DESCR

# =============================================================================
# Attribute Information: 
# Data published in 1978 summarizes factors that affect housing prices in Boston, USA
#         - CRIM     per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#         - INDUS    proportion of non-retail business acres per town
#         - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#         - NOX      nitric oxides concentration (parts per 10 million)
#         - RM       average number of rooms per dwelling
#         - AGE      proportion of owner-occupied units built prior to 1940
#         - DIS      weighted distances to five Boston employment centres
#         - RAD      index of accessibility to radial highways
#         - TAX      full-value property-tax rate per $10,000
#         - PTRATIO  pupil-teacher ratio by town
#         - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#         - LSTAT    % lower status of the population
#         - PRICE    Median value of owner-occupied homes in $1000's
# =============================================================================

boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)   # X-variables
boston_df.head()

boston_df['PRICE'] = boston.target    ## Y-variable
boston_df.head()
boston_df.iloc[:,-1].value_counts()

boston_df.shape
boston_df.info()

# boston_df.to_csv('boston.csv', index=False)
# from sklearn.datasets import fetch_openml
# X, y = fetch_openml('boston', return_X_y=True)

# 2. Explore data - Correlation analysis

boston_df_corr = boston_df.corr()
abs(boston_df_corr['PRICE']).nlargest(5)

import seaborn as sns

sns.pairplot(boston_df)


# 3. Data Modeling - Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

## 3-1. Datasets

Y = boston_df['PRICE']    # Y-variable
X = boston_df.drop(columns=['PRICE'])    # X-variable

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 156)

## 3-2. Generate the Model

### lr : instance of LinearRegression Class
lr = LinearRegression()

## 3-3. Train the model

### fit(x, y) => train datasets
lr.fit(X_train, Y_train)

## 3-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = lr.predict(X_test)


# 4. Evaluate & interpret the analysis results

## 4-1. Evaluate

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
r2_score = r2_score(Y_test, Y_predict)

## 4-2. Interpret

# =============================================================================
# y=a+b1*x1+b2*x2+ ~ +bn*xn
# a : constant (Y 절편)
# b1, b2, ~ bn : coefficient (회귀 계수)
# =============================================================================

### Constant
lr.intercept_ 

### Coefficient
lr.coef_

coef = pd.Series(data=np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending = False)

# =============================================================================
# Y(PRICE) = - 0.11*X(CRIM) + 0.07*X(ZN) + 0.03*X(INDUS) + 3.05*X(CHAS) 
#            - 19.80*X(NOX) + 3.35*X(RM) + 0.01*X(AGE) - 1.74*X(DIS)
#            + 0.36*X(RAD) - 0.01*X(TAX) - 0.92*X(PTRATIO) + 0.01*X(B)
#            - 0.57*X(LSTAT) + 41.00
# =============================================================================

np.round(lr.coef_, 2)
np.ceil(lr.coef_)
np.trunc(lr.coef_)
np.floor(lr.coef_)
