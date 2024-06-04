"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Unsupervised Learning - K-means Clustering - Practice(2)
# =============================================================================


# 1. Load datasets

from sklearn.datasets import make_blobs

# =============================================================================
# Input Parameter: 
# n_samples: the number of samples(표본 데이터의 수), default=100
# n_features: the number of features(독립 변수의 수), default=2
# centers: the number of centers to generate, or the fixed center locations(생성할 클러스터의 수 혹은 중심), default=None(3)
# cluster_std: the standard deviation of the clusters(클러스터의 표준 편차), default=1.0
# center_box: the bounding box for each cluster center when centers are generated at random(생성할 클러스터의 바운딩 박스), default=(-10.0, 10.0)
# shuffle: shuffle the samples(숫자를 랜덤으로 섞을지 여부), default=True
# 
# Return Parameter:
# X : the generated samples, (n_samples, n_features) 크기의 배열 샘플: 독립 변수
# y : the integer labels for cluster membership of each sample, (n_samples,) 크기의 배열: 종속 변수
# =============================================================================

import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='black', s=50)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# 2. Data Modeling - K-means Clustering

from sklearn.cluster import KMeans
from matplotlib import cm

kmeans = KMeans(n_clusters=3, init='random', random_state=0)
Y_labels = kmeans.fit_predict(X)

def clusterScatter(n_cluster, X_features): 
    c_colors = []
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster) 
        c_colors.append(c_color)
        plt.scatter(X_features[Y_labels == i,0], X_features[Y_labels == i,1],
                     marker='o', color=c_color, edgecolor='black', s=50, 
                     label='cluster '+ str(i))
    
    for i in range(n_cluster):
        plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], 
                    marker='*', color='red', edgecolor='black', s=200)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

clusterScatter(3, X)


# 3. Determine the number of clusters(k) - elbow method
 
distortions = []   # store distortion values (inertia_) in a List

for i in range(1, 11):    # k : 1 ~ 10
    kmeans_i = KMeans(n_clusters=i, random_state=0)  
    kmeans_i.fit(X)   
    distortions.append(kmeans_i.inertia_)

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# 4. Evaluate the analysis results - silhouette analysis

from sklearn.metrics import silhouette_score, silhouette_samples

score_samples = silhouette_samples(X, Y_labels, metric='euclidean')
score_samples.shape
score_samples

average_score = silhouette_score(X, Y_labels)
average_score
