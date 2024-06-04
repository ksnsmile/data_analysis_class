"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Regression - Linear Regression (Practice)
# =============================================================================


# 1. Load datasets

# =============================================================================
# Dataset Download:
# - UCI Machine Learning Repository
# - Auto MPG Data Set
# - https://archive.ics.uci.edu/ml/datasets/auto+mpg
# =============================================================================

import pandas as pd
import numpy as np

data_df = pd.read_csv('auto-mpg.csv', header=0)


# 2. Explore data & Pre-processing

data_df.shape
data_df.head()
data_df.info()

data_df.columns
# =============================================================================
# ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
#        'acceleration', 'model_year', 'origin', 'car_name']
# =============================================================================

# =============================================================================
# mpg: 연비
# cylinders: 실린더 개수
# displacement: 배기량
# horsepower: 마력
# weight: 무게
# acceleration: 엔진이 초당 얻을 수 있는 가속력
# model year: 출시 년도
# origin: 제조 장소(1: 미국 USA, 2: 유럽 EU, 3: 일본 JPN)
# car name: 자동차 모델명
# =============================================================================

data_df = data_df.drop(columns=['car_name', 'origin', 'horsepower'])

data_df.shape
data_df.head()

# 3. Data Modeling - Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

## 3-1. Datasets

Y = data_df['mpg']    # Y-variable
X = data_df.drop(columns=['mpg'])    # X-variables

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

## 3-2. Generate the Model

### lr : instance of LinearRegression Class
lr = LinearRegression()

## 3-3. Train the model

### fit(x, y) => train datasets
lr.fit(X_train, Y_train)

## 3-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = lr.predict(X_test)

### score(x, y) => test datasets; return R2
lr.score(X_test, Y_test)

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
print(np.round(lr.intercept_, 2))

### Coefficient
lr.coef_
print(np.round(lr.coef_, 2))

coef = pd.Series(data=np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending = False)

# =============================================================================
# Y(mpg) = - 0.14*X(cylinders) + 0.01*X(displacement) - 0.01*X(weight) + 0.20*X(acceleration) 
#          + 0.76*X(model_year) - 17.55
# =============================================================================


# 5. Visualize the analysis results

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(16,16) , ncols=3 , nrows=2)
lm_features = ['cylinders','displacement','weight','acceleration','model_year']
plot_color = ['r', 'b', 'y', 'g', 'r']

for i , feature in enumerate(lm_features):
    row = int(i/3)
    col = i%3
    sns.regplot(x=feature , y='mpg', data=data_df , ax=axs[row][col], color=plot_color[i])


# 6. Apply the analysis results

## 6-1. User input - X-variables

cyliners_1 = int(input("cylinders : "))
displacement_1 = int(input("displacement : "))
weigh_1 = int(input("weigh : "))
acceleration_1 = int(input("acceleration : "))
model_year_1 = int(input("model_year : "))

## 6-2. Predict - Y-variable (mpg)

mpg_predict = lr.predict([[cyliners_1, displacement_1, weigh_1, acceleration_1, model_year_1]])

print(mpg_predict)
