"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Classification - Logistic Regression
# =============================================================================


# 1. Load datasets

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

b_cancer = load_breast_cancer()
print(b_cancer.DESCR)   # Describe the breast_cancer dataset
b_cancer_descr = b_cancer.DESCR

# =============================================================================
# Attribute Information:
#         - radius (mean of distances from center to points on the perimeter) - 세포 크기
#         - texture (standard deviation of gray-scale values) - 질감
#         - perimeter - 둘레
#         - area - 면적
#         - smoothness (local variation in radius lengths) - 매끄러움
#         - compactness (perimeter^2 / area - 1.0) - 작은 정도
#         - concavity (severity of concave portions of the contour) - 오목함
#         - concave points (number of concave portions of the contour) - 오목한 곳의 수
#         - symmetry - 대칭성
#         - fractal dimension ("coastline approximation" - 1) - 프랙탈 차원
# 
#         The mean (평균), standard error (표준오차), and "worst" or largest 
#         (mean of the three largest values) (가장 나쁜 혹은 큰 측정치) of 
#         these features were computed for each image,resulting in 30 features.  
#         For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
#         - class:
#                 - WDBC-Malignant (악성) - 0
#                 - WDBC-Benign (양성) - 1
# =============================================================================

b_cancer_df = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)   # X-variables
b_cancer_df.head()

b_cancer_df['dianosis'] = b_cancer.target    ## Y-variable
b_cancer_df.head()

b_cancer_df.shape
b_cancer_df.info()

# =============================================================================
# Modeling approach:
#     - (a) Logistic regression - basic type
#     - (b) Logistic regression - standardization
#     - (c) Logistic regression - stratify (consider class characteristics)
# =============================================================================


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


# 3(a). Evaluate & interpret the analysis results

## 3(a)-1. Evaluate

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

confusion_matrix(Y_test, Y_predict)
#                 Predicted
#                 N(0) P(1)
#   Actual N(0) [[ 61,   2],
#          P(1)  [  5, 103]]

# Y_test.value_counts()    # Check!

accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy, precision, recall, f1))
print('ROC_AUC: {0: .3f}'.format(roc_auc))

## 3(a)-2. Interpret

### Check the beta values & meaning
column_name = ['const'] + b_cancer.feature_names.tolist() 
beta = np.concatenate([lr_b_cancer.intercept_, lr_b_cancer.coef_.reshape(-1)]).round(2)
odds = np.exp(beta).round(2)
interpret= np.where(beta>0, 'protective', 'risky')

beta_analysis = pd.DataFrame(np.c_[beta, odds, interpret], index=column_name, columns=['beta', 'exp(beta)', 'interpret'])
print(beta_analysis)

### Classification threshold (임계값, default = 0.5)
b_cancer_pro = lr_b_cancer.predict_proba(X_test)    # 1st column: probability of Y=0, 2nd column: probability of Y=1

b_cancer_threshold = np.linspace(0.01, 0.99, 10)

for threshold in b_cancer_threshold:
    Y_predict_th = np.where(b_cancer_pro[:,1] >= threshold, 1, 0)
    accuracy_th = accuracy_score(Y_test, Y_predict_th)
    precision_th = precision_score(Y_test, Y_predict_th)
    recall_th = recall_score(Y_test, Y_predict_th)
    f1_th = f1_score(Y_test, Y_predict_th)
    roc_auc_th = roc_auc_score(Y_test, Y_predict_th)
    print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f}, threshold: {4: .3f}'.format(accuracy_th, precision_th, recall_th, f1_th, threshold))
    print('ROC_AUC: {0: .3f}'.format(roc_auc_th))

### ROC Curve

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

Y_predict_pro = b_cancer_pro[:,1]

model_fpr, model_tpr, threshold1 = roc_curve(Y_test, Y_predict_pro)
random_fpr, random_tpr, threshold2 = roc_curve(Y_test, [0 for i in range(X_test.__len__())])

plt.figure(figsize = (10, 10))
plt.plot(model_fpr, model_tpr, marker = '.', label = 'Logistic')
plt.plot(random_fpr, random_tpr, linestyle = '--', label = 'Random')

plt.xlabel('False Positive Rate', size = 20)
plt.ylabel('True Positive Rate', size = 20)

plt.legend(fontsize = 20)

plt.title('ROC curve', size = 20)
plt.show()


# 2(b). Data Modeling - Logistic Regression (with Standardization)

## 2(b)-0. Preprocessing - Standardization

from sklearn.preprocessing import StandardScaler

### scaler : instance of the StandardScaler Class
scaler = StandardScaler()

### standardization target : X-variables -> b_cancer.data
### fit_transform() method : Fit to data, then transform it.
b_cancer_scaled = pd.DataFrame(scaler.fit_transform(b_cancer.data), columns = b_cancer.feature_names)

b_cancer_scaled.head()
b_cancer_df.head()

## 2(b)-1. Datasets

Y = b_cancer_df['dianosis']    # Y-variable
X = b_cancer_scaled    # X-variables

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

## 2(b)-2. Generate the Model

### lr_b_cancer : instance of LogisticRegression Class
lr_b_cancer = LogisticRegression()

## 2(b)-3. Train the model

### fit(x, y) => train datasets
lr_b_cancer.fit(X_train, Y_train)

## 2(b)-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = lr_b_cancer.predict(X_test)


# 3(b). Evaluate

confusion_matrix(Y_test, Y_predict)

accuracy_sd = accuracy_score(Y_test, Y_predict)
precision_sd = precision_score(Y_test, Y_predict)
recall_sd = recall_score(Y_test, Y_predict)
f1_sd = f1_score(Y_test, Y_predict)
roc_auc_sd = roc_auc_score(Y_test, Y_predict)

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy_sd, precision_sd, recall_sd, f1_sd))
print('ROC_AUC: {0: .3f}'.format(roc_auc_sd))


# 2(c). Data Modeling - Logistic Regression (with Stratify)

b_cancer_gy_des = b_cancer_df.groupby('dianosis').describe()

## 2(c)-1. Datasets

Y = b_cancer_df['dianosis']    # Y-variable
X = b_cancer_scaled    # X-variables

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0, stratify = Y)

## 2(c)-2. Generate the Model

### lr_b_cancer : instance of LogisticRegression Class
lr_b_cancer = LogisticRegression()

## 2(c)-3. Train the model

### fit(x, y) => train datasets
lr_b_cancer.fit(X_train, Y_train)

## 2(c)-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = lr_b_cancer.predict(X_test)


# 3(c). Evaluate

confusion_matrix(Y_test, Y_predict)

accuracy_sf = accuracy_score(Y_test, Y_predict)
precision_sf = precision_score(Y_test, Y_predict)
recall_sf = recall_score(Y_test, Y_predict)
f1_sf = f1_score(Y_test, Y_predict)
roc_auc_sf = roc_auc_score(Y_test, Y_predict)

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy_sf, precision_sf, recall_sf, f1_sf))
print('ROC_AUC: {0: .3f}'.format(roc_auc_sf))

