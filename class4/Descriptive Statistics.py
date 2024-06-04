"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Descriptive Statistics
# =============================================================================


# 1. Load datasets

import pandas as pd

red_df = pd.read_csv("winequality-red.csv", sep=";")
red_df.to_csv("winequality-red2.csv", index=False)

white_df = pd.read_csv("winequality-white.csv", sep=";")
white_df.to_csv("winequality-white2.csv", index=False)

## 1-1. Check data (basically)

red_df.shape
red_df.columns
red_df.index
red_df.info()
red_df.describe()

white_df.shape
white_df.columns
white_df.index
white_df.info()
white_df.describe()

## 1-2. Join datasets (red wine + white wine)

### Labeling for red wine
red_df.head()
red_df.insert(0,"type", value="red")    # see Material
red_df.head()
red_df.shape

### Labeling for white wine
white_df.head()
white_df.insert(0,"type", value="white")
white_df.head()
white_df.shape

### concat()
wine = pd.concat([red_df, white_df])
wine.shape
wine.to_csv("wine.csv", index=False)


# 2. Explore data

wine.info()
wine.head()
wine.tail()
wine_des = wine.describe()

wine.quality.unique()
sorted(wine.quality.unique())
wine.quality.value_counts()
wine.quality.nunique()
wine.quality.count()


# 3. Descriptive statistics / Summary statistics

## 3-1. Compare grouped data using describe()

wine.groupby('type')['quality'].describe()

wine.groupby('type')['quality'].mean()
wine.groupby('type')['quality'].min()
wine.groupby('type')['quality'].max()
wine.groupby('type')['quality'].count()
wine.groupby('type')['quality'].std()
wine.groupby('type')['quality'].quantile(0.25)    # see Material
wine.groupby('type')['quality'].quantile(0.5)
wine.groupby('type')['quality'].quantile(0.75)

## 3-2. Compare grouped data using statistical hypothesis Test

from scipy import stats

red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']

### 3-2-1. Normality test

# =============================================================================
#  H0: 해당 연속형 데이터 분포는 정규분포를 따른다
#      (The continuous data distribution follows a normal distribution)
#  H1: 해당 연속형 데이터 분포는 정규분포를 따르지 않는다
#      (The continuous data distribution does not follow a normal distribution)
# P-value (0.0) < 0.05 => Reject H0 / Accept H1
# =============================================================================

stats.normaltest(wine['quality'])    # NormaltestResult(statistic=50.358972216153944, pvalue=1.1606148581928246e-11)
stats.normaltest(red_wine_quality)    # NormaltestResult(statistic=17.26240081635554, pvalue=0.0001784503033385499)
stats.normaltest(white_wine_quality)    # NormaltestResult(statistic=27.683238421253613, pvalue=9.74229231258065e-07)

### 3-2-2. One sample t-test

# =============================================================================
# H0: 특정 값은 집단의 평균과 같다
#     (A certain value is equal to the mean of the group)
# H1: 특정 값은 집단의 평균과 다르다
#     (A certain value is not equal to the mean of the group)
# P-value (0.0) < 0.05 => Reject H0 / Accept H1
# =============================================================================

mean = wine.quality.mean()
stats.ttest_1samp(red_wine_quality, mean)    # Ttest_1sampResult(statistic=-9.029475106074619, pvalue=4.834747461832767e-19)
stats.ttest_1samp(white_wine_quality, mean)    # Ttest_1sampResult(statistic=4.704361617180321, pvalue=2.6166935092731884e-06)

### 3-2-3. two samples t-test

# =============================================================================
# H0: 두 집단의 평균이 유의미한 차이가 없다
#     (There is no significant difference between the means of the two groups)
# H1: 두 집단의 평균이 유의미한 차이가 있다
#     (There is a significant difference between the means of the two groups)
# P-value (0.0) < 0.05 => Reject H0 / Accept H1
# =============================================================================

#### Homogeneity of variance test

# =============================================================================
# H0: 두 집단의 분산이 같다 (차이가 없다)
#     (Both groups have the same variance)
# H1: 두 집단의 분산이 다르다 (차이가 있다)
#     (The variances of the two groups are different)
# P-value (0.126) > 0.05 => Accept H0 / Reject H1
# =============================================================================
    
stats.levene(red_wine_quality, white_wine_quality)    # LeveneResult(statistic=2.3327077520087762, pvalue=0.1267300410918103)

stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var = True)    # Ttest_indResult(statistic=-9.685649554187696, pvalue=4.888069044201508e-22)

### 3-2-4. chi-squre test()

# =============================================================================
# H0: A 항목과 B 항목은 서로 독립적이다 (연관성이 없다)
#     (A and B are independent of each other)
# H1: A 항목과 B 항목은 서로 독립적이지 않다 (연관성이 있다)
#     (A and B are not independent of each other)
# P-value (0.0) < 0.05 => Reject H0 / Accept H1
# =============================================================================

#### Contigency Table / Cross Table
wine_cross_table = pd.crosstab(wine['type'], wine['quality'])

stats.chi2_contingency(wine_cross_table)








