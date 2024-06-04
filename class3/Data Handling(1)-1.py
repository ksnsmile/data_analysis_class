# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Data Handling(1) - Basic
# =============================================================================


# 1. Load dataset

import pandas as pd

## Relative path
df = pd.read_csv('gapminder.tsv' , sep ='\t')
## Absolute path
df = pd.read_csv('C:/Class/gapminder.tsv', delimiter='\t')

# 2. Explore dataset

print(df)

## Check the data type by checking only the top 5 data for each variable 
print(df.head())
print(df.head(1))

## Check the data type by checking only the last 5 data for each variable 
print(df.tail())
print(df.tail(1))

## Basic information

### Size
print(df.shape) 
print(df.shape[0])
print(df.shape[1])

### Column, index/row name
print(df.columns)
print(df.index)

### Data types
df.values
print(df.dtypes)
print(df.info())


# 3. Extract data

## Extract columns
country_df = df['country']
print(type(country_df))
print(country_df.head())
print(country_df.tail())

subset = df[['country', 'continent', 'year']]
print(type(subset))
print(subset.head())
print(subset.tail())
 
## Extract rows

### Using loc (index)
print(df.loc[0])
print(df.loc[99])

number_of_rows = df.shape[0]
last_row_index = number_of_rows - 1
print(df.loc[last_row_index])    # Extract the last row

print(df.tail(1))    # Extract the last row

subset_tail = df.tail(1)    # Type = DataFrame
subset_loc = df.loc[last_row_index]    # Type = Series

multiple_rows = df.loc[[0, 99, 999]]
print(multiple_rows)

### Using iloc (row number)
print(df.iloc[1])
print(df.iloc[-1])
print(type(df.iloc[-1]))
print(df.loc[[0, 99, 999]])

## Extract rows & columns - Slicing, Range()

### Using Slicing
subset1 = df.loc[:, ['year', 'pop']]    # Column = string
print(subset1.head())

subset2 = df.iloc[:, [2, 4, -1]]    # Column = int
print(subset2.head())

### Using Range()
small_range = list(range(5))    # range(5) = range(0,5)
print(small_range)
subset3 = df.iloc[:, small_range]
# subset3 = df.iloc[:, list(range(5))]  
# subset3 = df.iloc[:, range(5)] 
print(subset3.head())

small_range = list(range(0, 6, 2))    # range(start, stop, step) 
subset4 = df.iloc[:, small_range] 
print(subset4.head())

### Slicing vs. Range()
subset5_S = df.iloc[:, :3]     # Slicing
print(subset5_S.head())

subset5_R = df.iloc[:, list(range(3))]    # Range()
print(subset5_R.head())

subset6_S = df.iloc[:, 0:6:2]     # Slicing
print(subset6_S.head())

subset6_R = df.iloc[:, list(range(0, 6, 2))]    # Range()
print(subset6_R.head())

print(df.loc[10:13, ['country', 'lifeExp', 'gdpPercap']])


# 4. Basic Statistics

df_des = df.describe()    # see Material

## Statistics of grouped data - groupby

df['year'].count()
df['year'].value_counts()

grouped_year_df = df.groupby('year')
print(grouped_year_df)

df.groupby('year').mean()
df.groupby('year').describe()
df.groupby('year').count()

grouped_year_df_lifeExp = df.groupby('year')['lifeExp']
grouped_year_df_lifeExp = grouped_year_df['lifeExp']
print(grouped_year_df_lifeExp)

df.groupby('year')['lifeExp'].mean()
df.groupby('year')['lifeExp'].describe()
df.groupby('year')['lifeExp'].count()

## Advanced groupby
multi_group_var = df.groupby(['year', 'continent'])[['lifeExp', 'gdpPercap']].mean() 
print(multi_group_var)

## nunique() - except for NaN, NA
print(df.groupby('continent')['country'].nunique())
print(df.groupby('continent')['country'].count())
check_data = df.groupby('continent')['country'].value_counts()


# 5. Graph

global_yearly_life_expectancy = df.groupby('year')['lifeExp'].mean() 
print(global_yearly_life_expectancy)

## Using plot()
global_yearly_life_expectancy.plot()

































