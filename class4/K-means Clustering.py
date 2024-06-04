"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Unsupervised Learning - K-means Clustering
# =============================================================================

# 1. Load datasets

# =============================================================================
# Dataset Download:
# - UCI Machine Learning Repository
# - online retail
# - https://archive.ics.uci.edu/ml/datasets/Online+Retail
# =============================================================================

import pandas as pd

retail_df = pd.read_excel('Online Retail.xlsx')
retail_df.head()
retail_df.tail()

# =============================================================================
# Data collection : 2010. 12. 01 ~ 2011. 12. 09
# Attribute Information:
# 
# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. 
#            If this code starts with letter 'c', it indicates a cancellation. - 송장번호
# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. - 제품 품목 코드
# Description: Product (item) name. Nominal. - 제품 설명 (제품 이름)
# Quantity: The quantities of each product (item) per transaction. Numeric. - 주문 수량
# InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.-주문 날짜/시간
# UnitPrice: Unit price. Numeric, Product price per unit in sterling. - 제품 단가
# CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. - 고객 번호
# Country: Country name. Nominal, the name of the country where each customer resides. - 고객 국적
# 
# =============================================================================


# 2. Explore data & pre-processing

## 2-1. Check data (basically)

retail_df.shape
retail_df.columns

# =============================================================================
# ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'UnitPrice', 'CustomerID', 'Country']
# =============================================================================

retail_df.index
retail_df.info()
retail_df.describe()
retail_df.isna().sum()   # columns with missing values : Description, CustomerID

## 2-2. Data cleaning

### Negative values
retail_df_pre = retail_df[(retail_df['Quantity']>0) & (retail_df['UnitPrice']>0)]                      

### Missing values
retail_df_pre = retail_df_pre.dropna()

### CustomerID data type : float64 -> int64
retail_df_pre['CustomerID'] = retail_df_pre['CustomerID'].astype('int64')
retail_df_pre.head()

### Duplicated data

# =============================================================================
# 1) Check the duplicated data
# duplicated() : a function that checks for duplicate values in each row
# - False : unique data
# - True : duplicated data
# 2) Remove the duplicated data
# drop_duplicates() : a function to remove duplicate rows
# - inplace=True : Data that has been deduplicated is applied directly to the dataframe
# =============================================================================

retail_df_pre.duplicated().sum()   
retail_df_pre.drop_duplicates(inplace=True)

### Check the pre-processed data
retail_df_pre.shape
retail_df_pre.info()
retail_df_pre.isna().sum()


# 3. Generate the dataset for analysis

## 3-1. Add the column 'SaleAmount' - 주문 금액 (= UnitPrice(제품 단가) * Quantity(주문 수량))
 
retail_df_pre['SaleAmount'] = retail_df_pre['UnitPrice'] * retail_df_pre['Quantity'] 
retail_df_pre.head()

## 3-2. Generate customer dataframe

# =============================================================================
# InvoiceNo : number of orders per customer
# SaleAmount : total orders per customer
# InvoiceDate : Last order date per customer
# =============================================================================

customer_df = retail_df_pre.groupby('CustomerID').agg({'InvoiceNo':'count', 'SaleAmount':'sum', 'InvoiceDate':'max'})
customer_df = customer_df.reset_index()
customer_df.head()

customer_df = customer_df.rename(columns={'InvoiceNo': 'Freq', 'InvoiceDate': 'ElapsedDays'})
customer_df.head()

### ElaspsedDays
 
import datetime    # see Material

customer_df['ElapsedDays'] = datetime.datetime(2011,12,10) - customer_df['ElapsedDays']
customer_df.head()
customer_df.info()

customer_df['ElapsedDays'] = customer_df['ElapsedDays'].apply(lambda x: x.days+1) 
customer_df.head()

## 3-3. Adjust the distribution of data - outliers handling (이상치 처리)    # see Material

import matplotlib.pyplot as plt
import seaborn as sns

### Boxplot graph
fig, ax = plt.subplots()
ax.boxplot([customer_df['Freq'], customer_df['SaleAmount'], customer_df['ElapsedDays']], sym='rs')   # rs = red square
plt.xticks([1, 2, 3], ['Freq', 'SaleAmount','ElapsedDays'])
plt.show()

### Apply the log function (로그 함수)

import numpy as np

customer_df['Freq_log'] = np.log1p(customer_df['Freq'])
customer_df['SaleAmount_log'] = np.log1p(customer_df['SaleAmount'])
customer_df['ElapsedDays_log'] = np.log1p(customer_df['ElapsedDays'])

customer_df.head() 

### Boxplot graph
fig, ax = plt.subplots()
ax.boxplot([customer_df['Freq_log'], customer_df['SaleAmount_log'], customer_df['ElapsedDays_log']], sym='bo')
plt.xticks([1, 2, 3], ['Freq_log', 'SaleAmount_log','ElapsedDays_log'])
plt.show()


# 4. Data Modeling - K-means Clustering

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

## 4-1. Standardization

scaler = StandardScaler()
X_features = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']]
X_features_scaled = scaler.fit_transform(X_features)

## 4-2. Determine the number of clusters(k) - elbow method (엘보 방법) 

distortions = []   # store distortion values (inertia_) in a List

for i in range(1, 11):    # k : 1 ~ 10
    kmeans_i = KMeans(n_clusters=i, random_state=0)  
    kmeans_i.fit(X_features_scaled)   
    distortions.append(kmeans_i.inertia_)
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

### => Conclusion : k = 3

## 4(a)-3. Generate the Model - k=3

### kmeans_3 : instance of KMeans Class
kmeans_3 = KMeans(n_clusters=3, random_state=0)

## 4(a)-4. Train the model - compute k-means clustering

### fit(x)
kmeans_3.fit(X_features_scaled)

## 4(a)-5. Predict the closest cluster each sample in X belongs to

### predict(x)
Y_labels = kmeans_3.predict(X_features_scaled)

### Y_labels = kmeans.fit_predict(X_features_scaled) 

## 4(a)-6. Add labels to customer_df

customer_df_result_3 = customer_df.copy()
customer_df_result_3['ClusterLabel'] = Y_labels
customer_df_result_3.head()


# 5. Evaluate & Visualize the analysis results

## 5(a)-1. Evaluate  - silhouette analysis (실루엣 분석)

score_samples = silhouette_samples(X_features_scaled, Y_labels, metric='euclidean')
score_samples.shape
score_samples
customer_df_result_3['SilhoutteCoeff'] = score_samples

average_score = silhouette_score(X_features_scaled, Y_labels)
average_score
customer_df_result_3.groupby('ClusterLabel')['SilhoutteCoeff'].mean()

# =============================================================================
# 0    0.342056
# 1    0.315260
# 2    0.264004
# =============================================================================

## 5(a)-2. Visualize - SilhouetteVisualizer 

### https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html
### !pip install yellowbrick

from yellowbrick.cluster import SilhouetteVisualizer

visualizer = SilhouetteVisualizer(kmeans_3, colors='yellowbrick')
visualizer.fit(X_features_scaled)      
visualizer.show()

customer_df_result_3['ClusterLabel_vis'] = visualizer.predict(X_features_scaled)
customer_df_result_3['SilhoutteCoeff_vis'] = visualizer.silhouette_samples_

average_score_vis = customer_df_result_3['SilhoutteCoeff_vis'].mean()


# 4. Data Modeling - K-means Clustering

## 4(b)-3. Generate the Model - k=4

### kmeans_4 : instance of KMeans Class
kmeans_4 = KMeans(n_clusters=4, random_state=0)

## 4(b)-4. Train the model - compute k-means clustering

### fit(x)
kmeans_4.fit(X_features_scaled)

## 4(b)-5. Predict the closest cluster each sample in X belongs to

### predict(x)
Y_labels = kmeans_4.predict(X_features_scaled)

### Y_labels = kmeans.fit_predict(X_features_scaled) 

## 4(b)-6. Add labels to customer_df

customer_df_result_4 = customer_df.copy()
customer_df_result_4['ClusterLabel'] = Y_labels
customer_df_result_4.head()


# 5. Evaluate & Visualize the analysis results

## 5(b)-1. Evaluate  - silhouette analysis (실루엣 분석)

score_samples = silhouette_samples(X_features_scaled, Y_labels, metric='euclidean')
score_samples.shape
score_samples
customer_df_result_4['SilhoutteCoeff'] = score_samples

average_score = silhouette_score(X_features_scaled, Y_labels)
average_score
customer_df_result_4.groupby('ClusterLabel')['SilhoutteCoeff'].mean()

# =============================================================================
# 0    0.239534
# 1    0.305898
# 2    0.347801
# 3    0.324294
# =============================================================================

## 5(b)-2. Visualize - SilhouetteVisualizer 

visualizer = SilhouetteVisualizer(kmeans_4, colors='yellowbrick')
visualizer.fit(X_features_scaled)      
visualizer.show()

customer_df_result_4['ClusterLabel_vis'] = visualizer.predict(X_features_scaled)
customer_df_result_4['SilhoutteCoeff_vis'] = visualizer.silhouette_samples_

average_score_vis = customer_df_result_4['SilhoutteCoeff_vis'].mean()

## 5(b)-3. Export the dataset

customer_df_result_4.to_csv('Online_Retail_Customer_Cluster.csv', index=False)

## 5(b)-4. Visualize the clustering results

from mpl_toolkits.mplot3d import Axes3D

x = X_features_scaled[:,0]
y = X_features_scaled[:,1]
z = X_features_scaled[:,2]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, c=Y_labels, marker='o', s= 20, alpha=0.5, cmap='rainbow')

ax.set_xlabel('X Label-Freq')
ax.set_ylabel('Y Label-SaleAmount')
ax.set_zlabel('Z Label-ElapsedDays')

plt.show()