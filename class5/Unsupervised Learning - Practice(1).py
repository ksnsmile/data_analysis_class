"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Unsupervised Learning - K-means Clustering - Practice(1)
# =============================================================================


# 1. Load datasets

import pandas as pd

customer_df = pd.read_csv('customer_df.csv')


# 2. Data Modeling - K-means Clustering

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

## 2-1. Standardization

scaler = StandardScaler()
X_features = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']]
X_features_scaled = scaler.fit_transform(X_features)

## 2-2. Determine the number of clusters(k) - elbow method (엘보 방법) 

distortions = []   # store distortion values (inertia_) in a List

for i in range(1, 11):    # k : 1 ~ 10
    kmeans_i = KMeans(n_clusters=i, random_state=0)  
    kmeans_i.fit(X_features_scaled)   
    distortions.append(kmeans_i.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

## 2-3. Generate the Model - k=4

### kmeans_4 : instance of KMeans Class
kmeans_4 = KMeans(n_clusters=4, random_state=0)

## 2-4. Train the model - compute k-means clustering

### fit(x)
kmeans_4.fit(X_features_scaled)

## 2-5. Predict the closest cluster each sample in X belongs to

### predict(x)
Y_labels = kmeans_4.predict(X_features_scaled)


## 2-6. Add labels to customer_df

customer_df_result = customer_df.copy()
customer_df_result['ClusterLabel'] = Y_labels
customer_df_result.head()


# 3. Evaluate the analysis results

### silhouette analysis (실루엣 분석)

score_samples = silhouette_samples(X_features_scaled, Y_labels, metric='euclidean')
score_samples.shape
score_samples
customer_df_result['SilhoutteCoeff'] = score_samples

average_score = silhouette_score(X_features_scaled, Y_labels)
average_score
customer_df_result.groupby('ClusterLabel')['SilhoutteCoeff'].mean()

# =============================================================================
# 0    0.239534
# 1    0.305898
# 2    0.347801
# 3    0.324294
# =============================================================================


# 4. Interpret the k-means clustering results (고객 마케팅 전략)

customer_df_result.columns
# ['CustomerID', 'Freq', 'SaleAmount', 'ElapsedDays', 'Freq_log',
#        'SaleAmount_log', 'ElapsedDays_log', 'ClusterLabel', 'SilhoutteCoeff',
#        'ClusterLabel_vis', 'SilhoutteCoeff_vis']

customer_df_result.groupby('ClusterLabel')['CustomerID'].count()
# 0     891
# 1    1207
# 2    1368
# 3     872

customer_cluster_df = customer_df_result[['CustomerID', 'Freq', 'SaleAmount', 'ElapsedDays', 'ClusterLabel']]

### Average purchase amount per sale(order)
customer_cluster_df['SaleAmountAvg'] = customer_cluster_df['SaleAmount']/customer_cluster_df['Freq']

customer_cluster_des = customer_cluster_df.groupby('ClusterLabel').describe()





















