# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Analysis(4)
# =============================================================================


# =============================================================================
# < Learn dataset to derive the model >
# 
# What to do
# - Sampling for train & test datasets
# - Train dataset -> algorithm/model, applying sampling methods
# - Test dataset -> prediction & performance evaluation (performance evaluation for classification)
# =============================================================================


import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_graphviz


## Load dataset ##

bc=pd.read_csv('BreastCancer.csv')


## Prepare dataset ##

train=bc[bc['Bare.nuclei'].notna()] 
test=bc[bc['Bare.nuclei'].isna()] 


## Apply decision-tree for missing data ##

dt_r=DecisionTreeRegressor()

dt_r.fit(train.drop(columns=['Bare.nuclei', 'Id', 'Class']),
        train['Bare.nuclei'])

bc_dt=bc.copy()

bc_dt.loc[bc_dt['Bare.nuclei'].isna(), 'Bare.nuclei']=\
    dt_r.predict(test.drop(columns=['Bare.nuclei', 'Id', 'Class'])) 
     
bc_dt.loc[bc['Bare.nuclei'].isna(), 'Bare.nuclei']


# =============================================================================
# < Sampling >
# 
# Sampling types : balanced sampling & unbalanced sampling
# (1) simple random sampling (단순임의샘플링)
# (2) stratified sampling (층화추출) : extraction method by group (by layer)  
# (3) systematic sampling (계통추출) : extraction in a certain pattern 
# =============================================================================

# Scikit-learn : model_selection module
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

# Sampling function : train_test_split()
# X-train, X-test, Y-train, Y-test = train_test_split(parameters)
# Parameters : 
# test_size : proportion of test set, default = 0.25 (0~1)
# train_size : proportion of train set (0~1)
# shuffle: default = True, whether or not to shuffle the data before splitting 
# stratify: default = None, important option in classification 
# random_state: control the shuffling applied to the data before applying the split


from sklearn.model_selection import train_test_split


## (1) Simple random sampling ##


# 1. Check the column/variable names 

bc_dt.columns
# =============================================================================
# ['Id', 'Cl.thickness', 'Cell.size', 'Cell.shape', 'Marg.adhesion',
#        'Epith.c.size', 'Bare.nuclei', 'Bl.cromatin', 'Normal.nucleoli',
#        'Mitoses', 'Class']
# =============================================================================


# 2. Define X & Y variables

# X :  remove Class & ID
# Y : Class (benign, malignant)
X=bc_dt.drop(columns=['Id', 'Class'])
Y=bc_dt.Class


# 3. Sampling

# train set : 70%, test set : 30%
# random_state : 1234
# train set -> (tr_x, tr_y)
# test set -> (te_x, te_y)
tr_x, te_x, tr_y, te_y =\
    train_test_split(X, Y,
                     test_size=0.3,
                     random_state=1234)


# 4. Decision tree for classification -> DecisionTreeClassifier
    
from sklearn.tree import DecisionTreeClassifier

# dt_c : instance of DecisionTreeClassifier
dt_c=DecisionTreeClassifier()


# 1) Learn/Train
# train dataset
dt_c.fit(tr_x, tr_y)

# feature_names = tr_x.columns -> names of X variables
# class_names = tr_y.unique() -> names of Y labels
export_graphviz(dt_c, 'tree_dt_c.dot', 
                feature_names=tr_x.columns,
                class_names=tr_y.unique())

dir(dt_c)

# Check variable importance 
dt_c.feature_importances_ 
pd.Series(dt_c.feature_importances_, index=tr_x.columns)
var_imp=pd.Series(dt_c.feature_importances_, index=tr_x.columns)

# sort_values() -> default : ascending=True
# Descending  -> ascending=False
var_imp.sort_values(ascending=False)


# 2) Predict
# test dataset
dt_c_pred=dt_c.predict(te_x) 


# 3) Evaluate
# score() : regression(R-square), classification(accuracy)
# test dataset
# Comparing dt_c_pred & te_y for performance results 
dt_c.score(te_x, te_y)


# 5. Evaluate - confusion matrix

# Scikit-learn : metrics module
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score

# Confusion Matrix (see Material)
confusion_matrix = confusion_matrix(te_y, dt_c_pred)
# =============================================================================
#      Predicted      Benign(N)   Malignant(P)
#  Actual
# Benign(N)           [[131(TN),  2(FP)],
# Malignant(P)        [ 13(FN), 64(TP)]]

# # accuracy - 정확도
# (131+64)/(131+2+13+64)
# # precision - 정밀도
# 64/(2+64)
# # recall - TPR, 재현율
# 64/(13+64)
# # TNR - 특이성
# 131/(131+2)
# # FPR
# 1-(131/(131+2))
# 2/(131+2)
# =============================================================================

te_y.value_counts()
# benign       133
# malignant     77

# = dt_c.score(te_x, te_y)
accuracy_ran=accuracy_score(te_y, dt_c_pred) # 0.9285714285714286
classification_report = classification_report(te_y, dt_c_pred)

precision_score(te_y, dt_c_pred, pos_label='malignant')
recall_score(te_y, dt_c_pred, pos_label='malignant')


# Algorithm comparison -> improvement 
# Sampling -> improvement  
# Improving performance by changing only the sampling method 


## (2) Stratified sampling ##

# Status of each group of the original dataset
bc.Class.value_counts()
# benign       458
# malignant    241

# Status of each group of the dataset applying simple random sampling
tr_y.value_counts()
# benign       325
# malignant    164


# 1. stratify = Y, variable with group criteria 
tr_x2, te_x2, tr_y2, te_y2 =\
    train_test_split(X, Y,
                     test_size=0.3,
                     random_state=1234,
                     stratify=Y)
    
# Status of each group of the dataset applying stratified sampling
tr_y2.value_counts()
# benign       320
# malignant    169

# Check the ratio of the train dataset
tr_y2.value_counts()/Y.value_counts()
# benign       0.698690
# malignant    0.701245

# Check the ratio of the test dataset
te_y2.value_counts()/Y.value_counts()
# benign       0.301310
# malignant    0.298755


# 2. Decision tree for classification -> DecisionTreeClassifier

# 1) Learn/Train
dt_c_stratify=DecisionTreeClassifier()

dt_c_stratify.fit(tr_x2, tr_y2)

# 2) Predict
dt_c_stratify_pred=dt_c_stratify.predict(te_x2)

# 3) Evaluate
dt_c_stratify.score(te_x2, te_y2)   # 0.9380952380952381


# 3. Evaluate - confusion matrix

confusion_matrix(te_y, dt_c_pred)
# array([[131,   2],
#        [ 13,  64]], dtype=int64)

confusion_matrix(te_y2, dt_c_stratify_pred)
# array([[131,  7],
#        [  6,  66]], dtype=int64)

te_y2.value_counts()
# benign       138
# malignant     72

accuracy_stratify = accuracy_score(te_y2, dt_c_stratify_pred)   # 0.9380952380952381
classification_report_stratify = classification_report(te_y2, dt_c_stratify_pred)


## (3) Unbalanced/imbalanced sampling ##

# imbalanced learn => imblearn
# Anaconda Prompt -> pip install --user imbalanced-learn
# https://imbalanced-learn.org/
from imblearn.over_sampling import SMOTE, SMOTENC


# =============================================================================
# imbalanced sampling (see Material) : 
# (1) under/down sampling
# (2) over/up sampling
# (3) combination sampling
# =============================================================================

# SMOTE (see Material) :
# 1. SMOTE
# 2. SMOTENC


## 1. SMOTE ##

# Instance
smote=SMOTE(random_state=1234)

# fit_resample() : re-generate dataset
# Y : criteria for classification, categorical variable
smote_X, smote_Y=smote.fit_resample(X, Y)

# Original dataset
Y.value_counts()
# benign       458
# malignant    241

# SMOTE dataset => over sampling
smote_Y.value_counts()
# malignant    458
# benign       458

len(smote_X)

tr_x3, te_x3, tr_y3, te_y3 =\
    train_test_split(smote_X, smote_Y,
                     test_size=0.3,
                     random_state=1234,
                     stratify=smote_Y) 

tr_y3.value_counts()
# benign       321
# malignant    320
tr_y3.value_counts()/smote_Y.value_counts()


# 1) Learn/Train
dt_c_smote=DecisionTreeClassifier()

dt_c_smote.fit(tr_x3, tr_y3)

# 2) Predict
dt_c_smote_pred=dt_c_smote.predict(te_x3)

# 3) Evaluate
dt_c_smote.score(te_x3, te_y3)


# 4) Confusion Matrix
confusion_matrix(te_y, dt_c_pred)
# array([[131,   2],
#        [ 13,  64]], dtype=int64)
confusion_matrix(te_y2, dt_c_stratify_pred)
# array([[131,  7],
#        [  6,  66]], dtype=int64)
confusion_matrix(te_y3, dt_c_smote_pred)
# array([[131,   6],
#       [  11,  127]], dtype=int64)

accuracy_smote = accuracy_score(te_y3,dt_c_smote_pred)   # 0.9381818181818182
classification_report_smote = classification_report(te_y3, dt_c_smote_pred)


## 2. SMOTENC ##

# numpy random module
import numpy.random as rd


# Data for practice
X.dtypes
len(X)

# numerical variables  X + numeric categorical variables X2

# rd.randint()
# randint(0,4) [0,1,2,3]
# RandomState => random_state()
X2=rd.RandomState(1234).randint(0,4, (len(X),2))
# array([[3, 3],
#        [2, 1],
#        [0, 0],
#        ...,
#        [1, 0],
#        [2, 3],
#        [2, 3]])

type(X2)

# X2 -> DataFrame, columns = D1, D2
X2=pd.DataFrame(X2, columns=['D1', 'D2'])

X2.D1.value_counts()
# 3    189
# 1    184
# 2    180
# 0    146
X2.D2.value_counts()
# 1    185
# 0    176
# 3    174
# 2    164

# Concatenate DataFrame, direction : column (axis=1)
X3=pd.concat([X, X2], axis=1)


# Instance of SMOTENC
# categorical_features : location of numeric categorical data 
smotenc=SMOTENC(random_state=1234, categorical_features=[9,10])

# fit_resample() : re-generate dataset
smotenc_x, smotenc_y=smotenc.fit_resample(X3, Y)

smotenc_y.value_counts()
# benign       458
# malignant    458
smotenc_x.D1.value_counts()
# 1    267
# 3    250
# 2    228
# 0    171
smotenc_x.D2.value_counts()
# 3    251
# 1    243
# 0    222
# 2    200