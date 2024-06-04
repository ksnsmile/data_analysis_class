"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Classification - Practice
# =============================================================================


# 1. Load datasets & Pre-processing

import pandas as pd

from sklearn.datasets import load_breast_cancer

b_cancer = load_breast_cancer()

b_cancer_df = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)   # X-variables
b_cancer_df['dianosis'] = b_cancer.target    ## Y-variable

b_cancer_df['dianosis'].value_counts()
# 1    357
# 0    212
b_cancer_df.info()
b_cancer_des = b_cancer_df.groupby('dianosis').describe()


# 2(a). Data Modeling - Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

## 2(a)-1. Datasets

Y = b_cancer_df['dianosis']    # Y-variable
X = b_cancer_df.drop(columns=['dianosis'])    # X-variables

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

## 2(a)-2. Generate the Model

### lr_b_cancer : instance of LogisticRegression Class
lr_b_cancer = LogisticRegression()

## 2(a)-3. Train the model

### fit(x, y) => train datasets
lr_b_cancer.fit(X_train, Y_train)

## 2(a)-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = lr_b_cancer.predict(X_test)


# 3(a). Evaluate

## 3(a)-1. Evaluate

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

confusion_matrix(Y_test, Y_predict)
#                 Predicted
#                 N(0) P(1)
#   Actual N(0) [[ 61,   2],
#          P(1)  [  5, 103]]

TN = 61
TP = 103
FP = 2
FN = 5

accuracy_m = (TN+TP)/(TN+FP+FN+TP)
precision_m = TP/(FP+TP)
recall_m = TP/(FN+TP)
f1_m = 2 * ((precision_m*recall_m)/(precision_m+recall_m))
FPR = FP/(FP+TN) 

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy_m, precision_m, recall_m, f1_m))
print('FPR: {0: .3f}'.format(FPR))
# 정확도:  0.959, 정밀도:  0.981, 재현율:  0.954, F1:  0.967,
# FPR:  0.032

accuracy_sk = accuracy_score(Y_test, Y_predict)
precision_sk = precision_score(Y_test, Y_predict)
recall_sk = recall_score(Y_test, Y_predict)
f1_sk = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy_sk, precision_sk, recall_sk, f1_sk))
print('ROC_AUC: {0: .3f}'.format(roc_auc))
# 정확도:  0.959, 정밀도:  0.981, 재현율:  0.954, F1:  0.967,
# ROC_AUC:  0.961

### ROC Curve

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

b_cancer_pro = lr_b_cancer.predict_proba(X_test)
Y_predict = b_cancer_pro[:,1]

model_fpr, model_tpr, threshold1 = roc_curve(Y_test, Y_predict)
random_fpr, random_tpr, threshold2 = roc_curve(Y_test, [0 for i in range(X_test.__len__())])

plt.figure(figsize = (10, 10))
plt.plot(model_fpr, model_tpr, marker = '.', label = 'Logistic Regression')
plt.plot(random_fpr, random_tpr, linestyle = '--', label = 'Random')

plt.xlabel('False Positive Rate', size = 20)
plt.ylabel('True Positive Rate', size = 20)

plt.legend(fontsize = 20)

plt.title('ROC curve', size = 20)
plt.show()


# 2(b). Data Modeling - Decision Tree

from sklearn.tree import DecisionTreeClassifier

## 2(b)-1. Datasets

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 123)

## 2(b)-2. Generate the Model

### dt_c : instance of DecisionTreeClassifier Class
dt_c = DecisionTreeClassifier(random_state=156)

## 2(b)-3. Train the model

### fit(x, y) => train datasets
dt_c.fit(X_train, Y_train)

## 2(b)-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = dt_c.predict(X_test)


# 3(b). Evaluate & Improve the analysis results

## 3(b)-1. Evaluate

confusion_matrix(Y_test, Y_predict)
#                 Predicted
#                 N(0) P(1)
#   Actual N(0) [[ 62,   6],
#          P(1)  [  3, 100]]

accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy, precision, recall, f1))
print('ROC_AUC: {0: .3f}'.format(roc_auc))
# 정확도:  0.947, 정밀도:  0.943, 재현율:  0.971, F1:  0.957,
# ROC_AUC:  0.941

### ROC Curve

b_cancer_pro = dt_c.predict_proba(X_test)
Y_predict = b_cancer_pro[:,1]

model_fpr, model_tpr, threshold1 = roc_curve(Y_test, Y_predict)
random_fpr, random_tpr, threshold2 = roc_curve(Y_test, [0 for i in range(X_test.__len__())])

plt.figure(figsize = (10, 10))
plt.plot(model_fpr, model_tpr, marker = '.', label = 'Decision Tree')
plt.plot(random_fpr, random_tpr, linestyle = '--', label = 'Random')

plt.xlabel('False Positive Rate', size = 20)
plt.ylabel('True Positive Rate', size = 20)

plt.legend(fontsize = 20)

plt.title('ROC curve', size = 20)
plt.show()

## 3(b)-2. Improve by optimizing the hiperparameters 

### Check the hiperparameters
first_params = dt_c.get_params()

### Optimize the hiperparameters

from sklearn.model_selection import GridSearchCV

### Hiperparameter : max_depth & min_samples_split, 
### search options : [2, 3, 4, 5] & [2, 4, 8]
params = {
    'max_depth' : [2, 3, 4, 5],
    'min_samples_split' : [2, 4, 8]
    }

### grid_cv : instance of GridSearchCV Class
grid_cv = GridSearchCV(dt_c, param_grid = params, scoring='roc_auc', cv=10, return_train_score = True)

### Train the decision tree models
grid_cv.fit(X_train, Y_train)

### Analysis results - optimized hiperparameters
cv_result_df = pd.DataFrame(grid_cv.cv_results_)

print('최고 평균 정확도: {0: .4f}, 최적 하이퍼 매개변수: {1}'.format(grid_cv.best_score_, grid_cv.best_params_))

### Best model -> best_estimator : best_opti_dt
best_opti_dt = grid_cv.best_estimator_
best_Y_predict = best_opti_dt.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)
print('Best 결정트리 예측 정확도: {0: .4f}'.format(best_accuracy))
# Best 결정트리 예측 정확도:  0.9708


# 2(c). Data Modeling - Randoms Forest

from sklearn.ensemble import RandomForestClassifier

## 2(c)-1. Datasets

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 123)

## 2(c)-2. Generate the Model

### rf_c : instance of RandomForestClassifier Class
rf_c = RandomForestClassifier(n_estimators=500, random_state=156)

## 2(c)-3. Train the model

### fit(x, y) => train datasets
rf_c.fit(X_train, Y_train)

## 2(c)-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = rf_c.predict(X_test)


# 3(c). Evaluate & Interpret the analysis results

## 3(c)-1. Evaluate

confusion_matrix(Y_test, Y_predict)
#                 Predicted
#                 N(0) P(1)
#   Actual N(0) [[ 66,   2],
#          P(1)  [  1, 102]]

accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy, precision, recall, f1))
print('ROC_AUC: {0: .3f}'.format(roc_auc))
# 정확도:  0.982, 정밀도:  0.981, 재현율:  0.990, F1:  0.986,
# ROC_AUC:  0.980

### ROC Curve

b_cancer_pro = rf_c.predict_proba(X_test)
Y_predict = b_cancer_pro[:,1]

model_fpr, model_tpr, threshold1 = roc_curve(Y_test, Y_predict)
random_fpr, random_tpr, threshold2 = roc_curve(Y_test, [0 for i in range(X_test.__len__())])

plt.figure(figsize = (10, 10))
plt.plot(model_fpr, model_tpr, marker = '.', label = 'Randoms Forest')
plt.plot(random_fpr, random_tpr, linestyle = '--', label = 'Random')

plt.xlabel('False Positive Rate', size = 20)
plt.ylabel('True Positive Rate', size = 20)

plt.legend(fontsize = 20)

plt.title('ROC curve', size = 20)
plt.show()


## 3(c)-2. Feature importance

feature_importance_values = rf_c.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)

## Ten features with high importance
feature_top10 = feature_importance_values_s.sort_values(ascending=False)[:10]
