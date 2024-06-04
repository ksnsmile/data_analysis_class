"""
Spyder Editor

This is a temporary script file.
"""


# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Visualization(1)
# =============================================================================

# 1. The importance of visualization

## 1-1. Load datasets

import seaborn as sns 
import matplotlib.pyplot as plt

anscombe = sns.load_dataset("anscombe") 
anscombe.head()

dataset_1 = anscombe[anscombe['dataset'] == 'I']
dataset_2 = anscombe[anscombe['dataset'] == 'II'] 
dataset_3 = anscombe[anscombe['dataset'] == 'III'] 
dataset_4 = anscombe[anscombe['dataset'] == 'IV']

an_stat = anscombe.groupby('dataset').agg(['mean', 'std']).round(2)
an_des = anscombe.groupby('dataset').describe()

## 1-2. Graphs

plt.plot(dataset_1['x'], dataset_1['y'])
plt.plot(dataset_1['x'], dataset_1['y'], 'o')

# =============================================================================
## How to plot graphs with matplot library

# 1) Create a basic frame to place the entire graph
# 2) Create a graph grid on which to draw graphs
# 3) Add one graph to each grid
#    - left -> right
#    - first row -> second row -> ...
# 4) Additional work - Title, Style, etc.
# 5) Reflect the graph
# =============================================================================

# =============================================================================
### 1) Create a basic frame to place the entire graph
### 2) Create a graph grid on which to draw graphs
#      - fig : basic frame
#      - ax : graph grid
# =============================================================================

fig, ax = plt.subplots(nrows=2, ncols=2) 
# =============================================================================
# fig = plt.figure()
# axes1 = fig.add_subplot(2, 2, 1)
# axes2 = fig.add_subplot(2, 2, 2)
# axes3 = fig.add_subplot(2, 2, 3)
# axes4 = fig.add_subplot(2, 2, 4)
# =============================================================================

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7,7))

### 3) Add one graph to each grid

ax[0, 0].plot(dataset_1['x'], dataset_1['y'], 'o', color ="red") 
ax[0, 1].plot(dataset_2['x'], dataset_2['y'], 'o', color ="green") 
ax[1, 0].plot(dataset_3['x'], dataset_3['y'], 'o', color ="blue") 
ax[1, 1].plot(dataset_4['x'], dataset_4['y'], 'o', color ="yellow")

# plt.show()

# =============================================================================
# axes1.plot(dataset_1['x'], dataset_1['y'], 'o', color ="red") 
# axes2.plot(dataset_2['x'], dataset_2['y'], 'o', color ="green") 
# axes3.plot(dataset_3['x'], dataset_3['y'], 'o', color ="blue") 
# axes4.plot(dataset_4['x'], dataset_4['y'], 'o', color ="yellow")
# =============================================================================

### 4) Additional work - Title, Style, etc.

ax[0, 0].set_title("dataset_1") 
ax[0, 1].set_title("dataset_2")
ax[1, 0].set_title("dataset_3") 
ax[1, 1].set_title("dataset_4")

fig.suptitle("Anscombe Data")

### 5) Reflect the graph & save it as a file

plt.show()
fig.savefig('fig1.png')


# 2. Matplotlib library

# https://matplotlib.org/
# https://wikidocs.net/137778

## 2-1. Load datasets

tips = sns.load_dataset("tips") 
tips.head()

## 2-2. Histogram (히스토그램)

# =============================================================================
### 1) Create the basic frame and grid of the graph
# 
# fig = plt.figure()
# axes1 = fig.add_subplot(1,1,1)
# 
# fig, axes1 = plt.subplots(1,1)
# 
# fig, axes1 = plt.subplots()
# =============================================================================

hist_fig = plt.figure()
axes1 = hist_fig.add_subplot(1,1,1)

# =============================================================================
### 2) hist()

# one variable => univariate graph (단변량 그래프)
# bins : the number of equal-width bins in the range - 막대 구간(bins)의 개수 (default = 10)
# cumulative : 누적 그래프 (default = False)
# histtype : The type of histogram to draw, {'bar', 'barstacked', 'step', 'stepfilled'} (default = 'bar')
# color : (default = None)
# label : String, or sequence of strings to match multiple datasets - 라벨 (default = None)
# stacked : If True, multiple data are stacked on top of each other (default = False)
# =============================================================================

#### Basic    
axes1.hist(tips['total_bill'], bins=12) 
axes1.set_title('Histogram of Total Bill')
axes1.set_xlabel('Frequency')
axes1.set_ylabel('Total Bill') 
plt.show()

#### cumulative, label, color & plt.hist()
plt.hist(tips['total_bill'], cumulative=True, label='cumulative=True', color = "green")
plt.hist(tips['total_bill'], cumulative=False, label='cumulative=False', color = "red")
plt.legend(loc='upper left')    # legend = 범례
plt.title('Histogram of Total Bill')
plt.xlabel('Frequency')
plt.ylabel('Total Bill')
plt.show()

## 2-3. Scatter plot (산점도)

# =============================================================================
### scatter()

# Two variables => bivariate graph (이변량 그래프)
# s : marker size (size**2)
# c : marker color
# marker : marker style (default = 'o'), https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
# alpha : transparency (투명도), 0 ~ 1 (default = None)
# cmap : color map
# =============================================================================

import numpy as np

scatter_plot = plt.figure() 
axes1 = scatter_plot.add_subplot(1, 1, 1) 
axes1.scatter(tips['total_bill'], tips['tip'], s = 10**2, c = np.random.rand(244), marker = '*', alpha = 0.5, cmap = 'Spectral') 
axes1.set_title('Scatterplot of Total Bill vs Tip') 
axes1.set_xlabel('Total Bill') 
axes1.set_ylabel('Tip')
plt.show()

## 2-4. Box plot (박스 그래프)

# =============================================================================
### boxplot()

# Categorical & continuous variables
# notch : Whether to draw a notched boxplot (True), or a rectangular boxplot (False) (default = False)
# sym : symbol 
# whis : The position of the whiskers (default = 1.5)
# labels : Labels for each dataset
# =============================================================================

boxplot, axes1 = plt.subplots()

axes1.boxplot( 
    [tips[tips['sex'] == 'Female']['tip'], 
     tips[tips['sex'] == 'Male']['tip']], 
     labels=['Female', 'Male'],
     notch=True, sym = 'bo', whis = 1.5)   # whis : 1.5 -> 2.5

axes1.set_xlabel('Sex') 
axes1.set_ylabel('Tip') 
axes1.set_title('Boxplot of Tips by Sex')
plt.show()

# =============================================================================

# boxplot, axes1 = plt.subplots()
# axes1.boxplot( 
#     [tips[tips['sex'] == 'Female']['tip'], 
#       tips[tips['sex'] == 'Male']['tip']], 
#       notch=True, sym = 'rs', whis = 1.5)   # whis : 1.5 -> 2.5

# axes1.set_xlabel('Sex') 
# axes1.set_ylabel('Tip') 
# axes1.set_title('Boxplot of Tips by Sex')
# plt.xticks([1, 2], ['Female', 'Male'])
# plt.show()


# Week #12 - clustering

# fig, ax = plt.subplots()
# ax.boxplot([customer_df['Freq_log'], customer_df['SaleAmount_log'], customer_df['ElapsedDays_log']], sym='bo')
# plt.xticks([1, 2, 3], ['Freq_log', 'SaleAmount_log','ElapsedDays_log'])
# plt.show()

# =============================================================================


# 3. Pandas - DataFrame & Series
# https://pandas.pydata.org/docs/reference/index.html

import pandas as pd

fig, ax = plt.subplots()
ax = tips['total_bill'].plot.hist()
tips['total_bill'].plot(kind='hist')

fig, ax = plt.subplots() 
tips[['total_bill', 'tip']].plot.hist(alpha=0.5, bins=20, ax=ax) 

fig, ax = plt.subplots() 
ax = tips['tip'].plot.kde()

fig, ax = plt.subplots() 
tips.plot.scatter(x='total_bill', y='tip', ax=ax) 

fig, ax = plt.subplots() 
tips.plot.hexbin(x='total_bill', y='tip', gridsize=10, ax=ax) 

fig, ax = plt.subplots() 
tips.plot.box(ax=ax) 