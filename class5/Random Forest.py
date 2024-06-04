"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Classification - Random Forest
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


# 2. Data Modeling - Randoms Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

## 2-1. Datasets

Y = b_cancer_df['dianosis']    # Y-variable
X = b_cancer_df.drop(columns=['dianosis'])    # X-variables

### Train datasets & test datasets - train_test_split()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 123)

## 2-2. Generate the Model

### dt_c : instance of RandomForestClassifier Class
rf_c = RandomForestClassifier(n_estimators=1000, random_state=156)

## 2-3. Train the model

### fit(x, y) => train datasets
rf_c.fit(X_train, Y_train)

## 2-4. Predict the Y-variable

### predict(x) => test datasets
Y_predict = rf_c.predict(X_test)


# 3. Evaluate & Improve the analysis results

## 3-1. Evaluate

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

confusion_matrix(Y_test, Y_predict)

# =============================================================================
#                                    predicted result
#                               negative (0) positive(1)
# actual  negative(0, Malignant)  [[ 66(TN),    2(FP)],
#         positive(1, Benign)      [  1(FN),  102(TP)]]
# =============================================================================

accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: {3: .3f},'.format(accuracy, precision, recall, f1))
print('ROC_AUC: {0: .3f}'.format(roc_auc))


### ROC Curve

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

b_cancer_pro = rf_c.predict_proba(X_test)
Y_predict = b_cancer_pro[:,1]

model_fpr, model_tpr, threshold1 = roc_curve(Y_test, Y_predict)
random_fpr, random_tpr, threshold2 = roc_curve(Y_test, [0 for i in range(X_test.__len__())])

plt.figure(figsize = (10, 10))
plt.plot(model_fpr, model_tpr, marker = '.', label = 'Random Forest')
plt.plot(random_fpr, random_tpr, linestyle = '--', label = 'Random')

plt.xlabel('False Positive Rate', size = 20)
plt.ylabel('True Positive Rate', size = 20)

plt.legend(fontsize = 20)

plt.title('ROC curve', size = 20)
plt.show()
