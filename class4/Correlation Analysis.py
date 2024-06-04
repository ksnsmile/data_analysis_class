"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Correlation Analysis
# =============================================================================


# 1. Load datasets

import pandas as pd
import seaborn as sns

titanic = sns.load_dataset("titanic")
titanic.to_csv("titanic.csv", index=False)

## 1-1. Check data (basically)

titanic.shape
titanic.columns
titanic.index
titanic.info()
t_des = titanic.describe()


# 2. Data Pre-processing / Handle missing values

## 2-1. Check missing values

titanic.isna().sum()

# =============================================================================
# Columns with missing values : age, embarked, deck, embark_town
# - age : continuous variable => fill missing values with median (중앙값)
# - deck : categorical variable => fill missing values with mode (최빈값)
# - embarked, embark_town : categorical variable => delete missing values
# =============================================================================

## 2-2. Fill missing values with median

titanic_fillna = titanic.copy()
t_age_median = titanic_fillna.age.median()
titanic_fillna['age'] = titanic_fillna.age.fillna(titanic_fillna.age.median())
# titanic_fillna.age.fillna(titanic.age.median(), inplace=True)

titanic[titanic['age'].isna()]['age']
titanic_fillna.loc[titanic['age'].isna(), :]['age']
titanic_fillna[titanic['age'].isna()]['age']

## 2-3. Fill missing values with mode

titanic_fillna.deck.value_counts()    # mode = 'C'
titanic_fillna['deck'] = titanic_fillna.deck.fillna('C')
# titanic_fillna.deck.fillna('C', inplace=True)

titanic[titanic['deck'].isna()]['deck']
titanic_fillna.loc[titanic['deck'].isna(), :]['deck']

## 2-4. Delete missing values

titanic_fillna.isna().sum()
titanic_fillna = titanic_fillna.dropna()
# titanic_fillna.dropna(inplace=True)
titanic_fillna.isna().sum()


# 3. Explore data / Visualization

import matplotlib.pyplot as plt

## Pie chart - survival rates by passenger gender

titanic_fillna.survived.value_counts()

f,ax = plt.subplots(1, 2, figsize = (10, 5))
titanic_fillna['survived'][titanic_fillna['sex'] == 'male'].value_counts().plot.pie(explode = [0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
titanic_fillna['survived'][titanic_fillna['sex'] == 'female'].value_counts().plot.pie(explode = [0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)

ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (Female)')

plt.show()

## Frequency graph - number of survivors by room class category

sns.countplot(x ='pclass', hue ='survived', data = titanic)
plt.title('Pclass vs Survived')
plt.show()


# 4. Data Modeling - Correlation Analysis

# =============================================================================
# Analysis of correlations between attributes on Titanic passengers and their survival
# - corr() -> calculation of correlation between variables 
# - Correlation coefficient -> Pearson correlation coefficient
# - Correlation analysis -> only for continuous data
# =============================================================================

titanic_fillna.info()

# =============================================================================
#  0   survived     889 non-null    int64 (applied)   
#  1   pclass       889 non-null    int64 (applied)  
#  2   sex          889 non-null    object  
#  3   age          889 non-null    float64 (applied) 
#  4   sibsp        889 non-null    int64 (applied)   
#  5   parch        889 non-null    int64 (applied)    
#  6   fare         889 non-null    float64 (applied) 
#  7   embarked     889 non-null    object  
#  8   class        889 non-null    category
#  9   who          889 non-null    object  
#  10  adult_male   889 non-null    bool (applied)    
#  11  deck         889 non-null    category
#  12  embark_town  889 non-null    object  
#  13  alive        889 non-null    object  
#  14  alone        889 non-null    bool (applied) 
# =============================================================================

titanic_fillna_corr = titanic_fillna.corr(method='pearson')
print(titanic_fillna_corr)
titanic_fillna_corr.to_csv("titanic_corr.csv", index=False)

titanic_fillna['survived'].corr(titanic_fillna['adult_male'])
titanic_fillna['survived'].corr(titanic_fillna['fare'])

# =============================================================================
# abs() : absolute value function 
# nlargest(n) : Find the n largest values 
# =============================================================================
abs(titanic_fillna_corr['survived']).nlargest(3)


# 5. Result - Visualization

## 5-1. Scatter plot - pairplot()

titanic_fillna_drop = titanic_fillna.drop(columns=['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive', 'alone', 'adult_male'])

sns.pairplot(titanic_fillna_drop)
plt.show()

## 5-2. catplot() - visualize the correlation between two variables (room class and survival)

sns.catplot(x='pclass', y='survived', hue='sex', data=titanic_fillna, kind='point')
plt.show()

