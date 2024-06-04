"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Data Handling(2) - Missing Value
# =============================================================================


# 1. What is missing values?

import numpy as np
import pandas as pd

## Missing values in numpy : NaN = NAN = nan

## Reasons for missing values

### When joining datasets with missing values
visited = pd.read_csv('survey_visited.csv') 
survey = pd.read_csv('survey_survey.csv')

print(visited)
print(survey)

vs = visited.merge(survey, left_on='ident', right_on='taken') 
print(vs)

### When entering data
num_legs = pd.Series({'goat': 4, 'amoeba': np.NaN}) 
print(num_legs)
print(type(num_legs))

scientists = pd.DataFrame({ 
    'Name': ['Rosaline Franklin', 'William Gosset'], 
    'Occupation': ['Chemist', 'Statistician'], 
    'Born': ['1920-07-25', '1876-06-13'], 
    'Died': ['1958-04-16', '1937-10-16'], 
    'missing': [np.NaN, np.NaN]}) 

print(scientists)
print(type(scientists))

### When extracting data by specifying a range
gapminder = pd.read_csv('gapminder.tsv', sep='\t')

life_exp = gapminder.groupby('year')['lifeExp'].mean() 
print(life_exp)

print(life_exp.loc[range(2000, 2010)])    # 2000~2009

y2000 = life_exp.loc[life_exp.index > 2000] 
# y2000 = life_exp[life_exp.index > 2000]
print(y2000)
y2005 = life_exp[(life_exp.index > 2000) & (life_exp.index < 2005)] 


# 2. Check missing values

print(np.NaN == True)
print(np.NaN == False)
print(np.NaN == 0)
print(np.NaN == '')
print(np.NaN == np.NaN)

ebola = pd.read_csv('country_timeseries.csv') 

ebola.count()

## For column
ebola.isna()    # True = NaN
ebola.isnull()    # isna() = isnull()

ebola.isna().sum()    # True(1) - missing values, False(0) - normal values

ebola.isna().all()    # True - variables that all values are NaN 

ebola.isna().any()    # True - variables with at least one NaN 
ebola.columns[ebola.isna().any()]    # Column names with missing values

# columns = ebola.columns
# ebola.columns[0]

## For row/index

ebola.isna().sum(axis=1)

ebola.isna().all(axis=1)    # True - rows(indexes) where all values are NaN 

ebola.isna().any(axis=1)    # True - rows(indexes) with at least one NaN
ebola.index[ebola.isna().any(axis=1)]    # index names with missing values

## Number of missing values
np.count_nonzero(ebola.isnull())
np.count_nonzero(ebola['Cases_Guinea'].isnull())

ebola.Cases_Guinea.value_counts(dropna=False)
# ebola['Cases_Guinea'].value_counts(dropna=False)


# 3. Deal with missing values

## Delete
print(ebola.shape)

ebola_dropna = ebola.dropna() 
print(ebola_dropna.shape)

## Change - fillna()
ebola_fillna0 = ebola.fillna(0)
ebola_fillna_iloc = ebola.fillna(0).iloc[0:10, 0:5]

### fillna() with method='ffill' or method='pad'
ebola.fillna(method='ffill').iloc[0:10, 0:5]    # Fill missing values forward
ebola.fillna(method='pad').iloc[0:10, 0:5]
# ebola_ffill = ebola.fillna(method='ffill')

### fillna() with method='bfill' or method='backfill'
ebola.fillna(method='bfill').iloc[0:10, 0:5]    # Fill missing values backward
ebola.fillna(method='backfill').iloc[0:10, 0:5]
# ebola_bfill = ebola.fillna(method='backfill')

## Change - interploate()
ebola.interpolate().iloc[0:10, 0:5]


# 4. Data calculation with missing values

ebola['Cases_multiple'] = ebola['Cases_Guinea'] + ebola['Cases_Liberia'] + ebola['Cases_SierraLeone']
ebola_subset = ebola.loc[:, ['Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone', 'Cases_multiple']] 
print(ebola_subset.head(10))

ebola['Cases_multiple'] = ebola[['Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone']].sum(axis=1, skipna=True)
ebola_subset = ebola.loc[:, ['Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone', 'Cases_multiple']] 
print(ebola_subset.head(10))

ebola.Cases_Guinea.sum(skipna = False)
ebola.Cases_Guinea.sum(skipna = True)
