# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Practice of Manufacturing Data(2)
# =============================================================================


## 1단계. 라이브러리/데이터 불러오기

# 1-1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance

# !pip install bayesian-optimization

# 1-2. 데이터 불러오기
dc_data=pd.read_csv("DieCasting_Raw_Data.csv")

dc_data
dc_data.head(10)
dc_data.tail(10)


## 2단계. 데이터 종류 및 개수 확인

# 2-1. 칼럼명 확인
dc_data.columns

# 2-2. 데이터 정보 확인
dc_data.info()

# 2-3. 요약 통계량 확인
dc_des = dc_data.describe()

# 2-4. 칼럼 별 데이터 개수 확인
dc_data.count()


## 3단계. 데이터 정제(전처리)

# 3-1. 필요없는 feature 제거
dc_data_drop = dc_data.drop(['Shot', '_id'],axis=1)
dc_data_drop.columns

# 3-2. 결측치 확인 
dc_data_drop.isnull().sum()

# 3-3. 행의 결측치 개수 확인
dc_data_drop.isnull().sum(axis=1)
dc_data_drop.isnull().sum(axis=1).value_counts()

# 3-4. 결측치 제거
dc_data_drop = dc_data_drop.dropna()

# set_index() : 기존 행 인덱스를 제거하고 데이터 열 중 하나를 인덱스로 설정
# reset_index() : 기존 행 인덱스를 제거하고 인덱스를 데이터 열로 추가
# drop=True : 기존 인덱스를 삭제
dc_data_drop = dc_data_drop.reset_index(drop=True)

# 3-5. 결측치 제거 결과 확인
dc_data_drop.isnull().sum()


## 4단계. 데이터 특성 파악

# 4-1. 변수 간의 상관관계 파악 
plt.subplots(figsize=(25,25))
sns.heatmap(data = dc_data_drop.corr(), linewidths=0.1, annot=True,
            fmt = '.2f', cmap='Blues') 

# 4-2. Histogram을 통한 변수별 데이터 파악
plt.figure(figsize = (30,30))

# 각 변수의 막대그래프 개수 
bin = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10] 

for index, value in enumerate(dc_data_drop):
    sub = plt.subplot(5, 5, index +1) 
    sub.hist(dc_data_drop[value], bins = bin[index], 
             facecolor = (144/255,171/255,221/255), linewidth=.3, edgecolor ='black')
    plt.title(value)  
    

## 5단계. 학습 및 평가 데이터 분리

# 5-1. 설비 상태(정상/비정상 데이터) 파악

# 정상 및 비정상 데이터 파악
dc_data_drop.columns
dc_data_drop['Machine_Status'].value_counts()
dc_data_drop['Machine_Status'].dtypes
dc_data_drop['Machine_Status'] = dc_data_drop['Machine_Status'].astype(int)

# 5-2. MinMaxScaler를 통한 데이터 정규화

# MinMaxScaler를 통한 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
dc_data_scaler = scaler.fit_transform(dc_data_drop)

# 데이터프레임 형식으로 바꿔주기
dc_data_scaler = pd.DataFrame(dc_data_scaler)
dc_data_scaler.columns = dc_data_drop.columns 
dc_data_scaler.head()

# 5-3. 학습 데이터/평가 데이터 분리

# 학습 데이터/평가 데이터 분리
# 독립변수
X = dc_data_scaler.drop(columns=['Machine_Status']) 
# 종속변수
Y = dc_data_scaler['Machine_Status']

train_x, test_x, train_y, test_y = train_test_split(X, 
                                                    Y,
                                                    test_size = 0.3, 
                                                    random_state = 42)

# 5-4. 정상/비정상 데이터 불균형 해결

# 정상/비정상 데이터 불균형 해결
smote = SMOTE(random_state=42)

X_train_over, y_train_over = smote.fit_resample(train_x, train_y)

print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ', train_x.shape, train_y.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 : ', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 값의 분포 : ' + '\n', pd.Series(y_train_over).value_counts())


## 6단계. AI모델 구축 및 훈련

# 6-1. 파라미터 최적화를 위한 목적함수 생성

# 목적함수 생성
def RF(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    model = RandomForestClassifier( n_estimators =  int(round(n_estimators)),
                                max_depth = int(round(max_depth)),
                                min_samples_split = int(round(min_samples_split)),
                                min_samples_leaf = int(round(min_samples_leaf)),
                                max_leaf_nodes = int(round(max_leaf_nodes))
                               )
    scoring = {'f1_score': make_scorer(f1_score)}
    result = cross_validate(model, X_train_over, y_train_over, cv=5, scoring=scoring)
    f1_score_mean = result["test_f1_score"].mean()
    return f1_score_mean

# 6-2. 기준 하이퍼파라미터 범위 설정

# 파라미터 범위 설정
pbounds = {'n_estimators': (1, 200),
           'max_depth': (2, 25),
           'min_samples_split': (2, 20),
           'min_samples_leaf': (1, 10),
           'max_leaf_nodes': (2, 25),
          }

# 6-3. 베이지안 최적화 객체 생성 및 실행

# 베이지안 최적화
# f: 목적함수, pbounds:입력값 탐색구간
LFBO = BayesianOptimization(f = RF, pbounds = pbounds, verbose = 2, random_state = 0)
# 목적함수가 최대가 되는 최적해 찾기(acq=ei)
LFBO.maximize(init_points=5, n_iter = 20, acq='ei', xi=0.01)

# 6-4. 최적 파라미터 도출
LFBO.max

# 6-5. 랜덤포레스트 모델 구축 및 훈련

# 파라미터 정의
rfc = RandomForestClassifier(n_estimators=200, max_depth=25, max_leaf_nodes = 25,
                             min_samples_leaf = 1, min_samples_split = 2, random_state=0)

# 모델 훈련
rfc.fit(X_train_over, y_train_over)


## 7단계. 결과 분석 및 해석

# 7-1. 오차행렬 및 정확도
def get_clf_eval(y_test, pred=None):
    print('오차행렬 \n', confusion_matrix(y_test, pred))
    print('정확도 :', accuracy_score(y_test, pred))
    
def get_model_train_eval(model, ftr_train = None, ftr_test = None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    get_clf_eval(tgt_test, pred)
    
get_model_train_eval(rfc, X_train_over, test_x, y_train_over, test_y)

# 7-2. 변수 중요도 - Feature Importance

dir(rfc)
rfc_fi = rfc.feature_importances_
rfc_fi_df = pd.DataFrame(rfc_fi, index=train_x.columns)

# permutation_importance() : 모델을 학습시킨 뒤, 특정 feature 데이터를 shuffle 했을 때, 
# 검증 데이터 셋에 대한 예측성능을 확인하고 feature importance 계산
result_rfc = permutation_importance(rfc, test_x, test_y, 
                                    n_repeats=30, random_state=333)

dir(result_rfc)
# importances : 재배열/순열 결과
# importances_mean : 결과의 평균
# importances_std : 결과의 표준편차

importances_rf = pd.DataFrame(result_rfc.importances_mean, 
                              index=train_x.columns)
importances_rf.sort_values(by=0, axis=0, ascending=False, inplace=True)
importances_rf.plot.bar()


## 8단계. 주요 인자 범위 제시
dc_data_drop.columns

# 8-1. 주요 인자 boxplot
Iqr_dc_data = dc_data_drop.loc[:,['Cycle_Time', 'High_Velocity', 
                                'Rapid_Rise_Time','Velocity_2',
                                'Casting_Pressure','Machine_Status']] 
Iqr_dc_data = Iqr_dc_data[Iqr_dc_data['Machine_Status']==0]
Iqr_dc_data = Iqr_dc_data.drop(['Machine_Status'],axis=1)
Iqr_dc_data = Iqr_dc_data.reset_index(drop=True)
Iqr_dc_data 

# 8-2. 5개 주요인자 Boxplot 시각화
plt.figure(figsize = (20, 10))

for index, value in enumerate(Iqr_dc_data):
    sub = plt.subplot(2, 3, index+1)
    sub.boxplot(Iqr_dc_data[value])
    plt.title(value)

# 8-3. IQR 이상치 함수 설정
    
# 이상치 함수
def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25,75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    return np.where((data<lower_bound)|(data>upper_bound))

# 8-4. 주요 인자별 이상치 개수 확인
Cycle_Time_outlier = outliers_iqr(Iqr_dc_data['Cycle_Time'])[0]
High_Vel_outlier = outliers_iqr(Iqr_dc_data['High_Velocity'])[0]
Rapid_Rise_Time_outlier = outliers_iqr(Iqr_dc_data['Rapid_Rise_Time'])[0]
Velocity_2_outlier = outliers_iqr(Iqr_dc_data['Velocity_2'])[0]
Casting_Pressure_outlier = outliers_iqr(Iqr_dc_data['Casting_Pressure'])[0]
lead_outlier_index  = np.concatenate((Cycle_Time_outlier, High_Vel_outlier,
                                      Rapid_Rise_Time_outlier, Velocity_2_outlier, 
                                      Casting_Pressure_outlier), axis=None)

# 중복 index 제거
lead_outlier_index = list(set(lead_outlier_index))

print("CyCle_Time의 이상치 개수: ",len(Cycle_Time_outlier))
print("High_Velocity의 이상치 개수: ",len(High_Vel_outlier))
print("Rapid_Rise_Time의 이상치 개수: ",len(Rapid_Rise_Time_outlier))
print("Velocity_2의 이상치 개수: ",len(Velocity_2_outlier))
print("Casting_Pressure의 이상치 개수: ",len(Casting_Pressure_outlier))
print("전체 이상치 개수",len(lead_outlier_index))

# 8-5. 이상치 제거
not_outlier_index=[]
for j in Iqr_dc_data.index:
    if j not in lead_outlier_index: 
        not_outlier_index.append(j)
        
Iqr_dc_data_clean = Iqr_dc_data.loc[not_outlier_index]
Iqr_dc_data_clean = Iqr_dc_data_clean.reset_index(drop=True)

print(Iqr_dc_data.shape, Iqr_dc_data_clean.shape)

# 8-6. 이상치 제거 후 boxplot 시각화
plt.figure(figsize = (20, 10))

for index, value in enumerate(Iqr_dc_data_clean):
    sub = plt.subplot(2, 3, index+1)
    sub.boxplot(Iqr_dc_data_clean[value])
    plt.title(value)

# 8-7. 주요 인자 범위 확인
for i in Iqr_dc_data_clean.columns:
    a = Iqr_dc_data_clean[i].max()
    b = Iqr_dc_data_clean[i].min()
    print(f'{i}의 범위: {b} ~ {a}')


