"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Data Handling(3) - Data Aggregation & Filtering
# =============================================================================


# 1. Data aggregation 

## Mean value using groupby()

import pandas as pd 

df = pd.read_csv('gapminder.tsv', sep='\t')

avg_life_exp_by_year = df.groupby('year').lifeExp.mean()    
# avg_life_exp_by_year = df.groupby('year')['lifeExp'].mean() 
print(avg_life_exp_by_year)

## Split - Apply - Combine process of groupby()

### Split
years = df.year.unique() 
print(years)
# [1952 1957 1962 1967 1972 1977 1982 1987 1992 1997 2002 2007]

### Apply
y1952 = df.loc[df.year == 1952, :]    # For 1952
# y1952 = df[df.year == 1952] 
print(y1952.head())

y1952_mean = y1952.lifeExp.mean()    
# y1952_mean = y1952['lifeExp'].mean()
print(y1952_mean)

y1957 = df.loc[df.year == 1957, :] 
y1957_mean = y1957.lifeExp.mean( )
print(y1957_mean)

y1962 = df.loc[df.year == 1962, :] 
y1962_mean = y1962.lifeExp.mean( )
print(y1962_mean)

y2007 = df.loc[df.year == 2007, :] 
y2007_mean = y2007.lifeExp.mean( )
print(y2007_mean)

### Combine
df2 = pd.DataFrame({"year":[1952, 1957, 1962, 2007], 
                    "":[y1952_mean, y1957_mean,y1962_mean,y2007_mean]}) 
print(df2)
print(avg_life_exp_by_year)

## Groupby() and user defined functions

### User defined function for mean value
def my_mean(values):
    n = len(values)
    sum = 0 
    for value in values:
        sum += value  
    return sum / n

### agg() for a function   # see Material
agg_my_mean = df.groupby('year').lifeExp.agg(my_mean)   
# agg_my_mean = df.groupby('year').lifeExp.apply(my_mean)
# agg_my_mean = df.groupby('year').lifeExp.mean()

print(agg_my_mean)

### agg() for multiple functions

import numpy as np

gdf = df.groupby('year').lifeExp.agg([np.count_nonzero, np.mean, np.std])    # using List
print(gdf)

gdf_dict = df.groupby('year').agg({'lifeExp': 'mean', 'pop': 'median', 'gdpPercap': 'median'})    # using Dictionary
# df.groupby('year').agg({'lifeExp': ['mean', 'median'], 'pop': 'median', 'gdpPercap': 'median'})
print(gdf_dict)


# 2. Data filtering

## filter()    # see Material

import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.shape)

tips['size'].value_counts()

# if apply()?
tips_applied = tips.groupby('size').apply(lambda x: x['size'].count() >= 30)    # see Material
tips_applied

tips_filtered = tips.groupby('size').filter(lambda x: x['size'].count() >= 30)
print(tips_filtered.shape)
print(tips_filtered['size'].value_counts())
    

# 3. Handling missing values using groupby()

## Dataset preparation
tips_10 = tips.sample(10)
tips_10.iloc[2:6,0] = np.NaN
print(tips_10)

## Filling missing values with mean values
count_sex = tips_10.groupby('sex').count()    # Imbalanced data 
count_des = tips_10.groupby('sex').describe()
print(count_sex)
print(count_des)

def fill_na_mean(x):
    avg = x.mean() 
    return x.fillna(avg)

tips_10.groupby('sex').total_bill.mean()
total_bill_group_mean = tips_10.groupby('sex').total_bill.apply(fill_na_mean)
total_bill_group_mean_lambda = tips_10.groupby('sex').total_bill.apply(lambda x: x.fillna(x.mean()))

print(total_bill_group_mean)
print(total_bill_group_mean_lambda)

tips_10['fill_total_bill'] = total_bill_group_mean

print(tips_10)
