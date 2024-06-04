"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Classification - Decision Tree
# =============================================================================


# 1. Load datasets

# =============================================================================
# Dataset Download:
# - UCI Machine Learning Repository
# - Human activity recognition
# - https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
# - https://youtu.be/XOEN9W05_4A
# =============================================================================

import pandas as pd
import numpy as np

## 1-1. Load the feature name

feature_name_df = pd.read_csv('features.csv', names=['feature_name'])

## 1-2 Load the label name

label_name_df = pd.read_csv('activity_labels.csv', names=['label', 'mean'])

## 1-3. Load the X_train, X_test, Y_train, Y_test

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
Y_train = pd.read_csv('Y_train.csv')
Y_test = pd.read_csv('Y_test.csv')

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
X_train.head()

Y_train['action'].value_counts()


# 2. Data Modeling - Decision Tree

from sklearn.tree import DecisionTreeClassifier

## 2-1. Generate the Model

### dt_c : instance of DecisionTreeClassifier Class
dt_c = DecisionTreeClassifier(random_state=156)

## 2-2. Train the model

### fit(x, y) => train datasets
dt_c.fit(X_train, Y_train)

## 2-3. Predict the Y-variable

### predict(x) => test datasets
Y_predict = dt_c.predict(X_test)


# 3. Evaluate & Improve the analysis results

from sklearn.metrics import accuracy_score

## 3-1. Evaluate

accuracy = accuracy_score(Y_test, Y_predict)
print('결정트리 예측 정확도: {0: .4f}'.format(accuracy))

## 3-2. Improve by optimizing the hiperparameters 

### Check the hiperparameters
first_params = dt_c.get_params()

### Optimize the hiperparameters

from sklearn.model_selection import GridSearchCV

### Hiperparameter : max_depth, search options : [6, 8, 10, 12, 16, 20, 24]
params = {
    'max_depth' : [6, 8, 10, 12, 16, 20, 24]
    }

### grid_cv : instance of GridSearchCV Class
grid_cv = GridSearchCV(dt_c, param_grid = params, scoring='accuracy', cv=5, return_train_score = True)

### Train the seven decision tree models
grid_cv.fit(X_train, Y_train)

### Analysis results - optimized hiperparameters
cv_result_df = pd.DataFrame(grid_cv.cv_results_)

### accuracy: 0.8513 < 0.8548
cv_result_df[['param_max_depth', 'mean_test_score', 'mean_train_score']]
print('최고 평균 정확도: {0: .4f}, 최적 하이퍼 매개변수: {1}'.format(grid_cv.best_score_, grid_cv.best_params_))

### Hiperparameter : max_depth & min_samples_split, search options : [8, 16, 20] & [8, 16, 24]
params_add = {
    'max_depth' : [8, 16, 20],
    'min_samples_split' : [8, 16, 24]
    }

grid_cv_add = GridSearchCV(dt_c, param_grid = params_add, scoring='accuracy', cv=5, return_train_score = True)

### Train the nine decision tree models
grid_cv_add.fit(X_train, Y_train)
cv_add_result_df = pd.DataFrame(grid_cv_add.cv_results_)
cv_add_result_df[['param_max_depth', 'param_min_samples_split', 'mean_test_score', 'mean_train_score']]

### accuracy: 0.8549 > 0.8548, optimized hiperparameters: 'max_depth': 8, 'min_samples_split': 16
print('최고 평균 정확도: {0: .4f}, 최적 하이퍼 매개변수: {1}'.format(grid_cv_add.best_score_, grid_cv_add.best_params_))

### Best model -> best_estimator : best_opti_dt
best_opti_dt = grid_cv_add.best_estimator_
best_Y_predict = best_opti_dt.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)
print('Best 결정트리 예측 정확도: {0: .4f}'.format(best_accuracy))


# 4. Interpret the analysis results - feature importance

import seaborn as sns
import matplotlib.pyplot as plt

feature_importance_values = best_opti_dt.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)

## Ten features with high importance
feature_top10 = feature_importance_values_s.sort_values(ascending=False)[:10]

## Bar chart
plt.figure(figsize=(10,5))
plt.title('Feature Top 10')
sns.barplot(x=feature_top10, y=feature_top10.index)
plt.show()


# 5. Visualize the analysis results

from sklearn.tree import export_graphviz

## Install graphviz-2.38 
## https://graphviz.org/download/

# export_graphviz(instance object, file name)
# gveedit.ext
export_graphviz(best_opti_dt, 'best_tree.dot')

# max_depth = 3 : visualize up to depth 3
# feature_names : X-variables, X_train.columns
export_graphviz(best_opti_dt, 'best_tree_op.dot', max_depth=3,
           feature_names=X_train.columns)








