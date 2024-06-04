# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Data Handling(1) - Basic (Practice)
# =============================================================================


# 1. Load dataset
# File name: gapminder.csv

import pandas as pd

df = pd.read_csv('gapminder.csv')


# 2. Explore dataset

df.head()
df.tail()

df.shape
df.columns
df.index
df.values
df.dtypes

df.info()
df.describe()


# 3. Extract data

df.columns
# ['country', 'continent', 'year', 'lifeExp', 'pop', 'gdpPercap']

df['continent']
sub_df = df[['continent', 'pop', 'gdpPercap']]

sub_df.loc[0]
sub_df.loc[df.shape[0]-1]

sub_df.iloc[0]
sub_df.iloc[-1]

df.loc[:, ['pop', 'gdpPercap']] 
df.iloc[:, :3]
df.iloc[:, list(range(3))]    
# df.iloc[:, range(3)]


# 4. Basic statistics

sub_df['continent'].count()
sub_df['continent'].nunique()
sub_df['continent'].unique()
sub_df['continent'].value_counts()

sub_df.groupby('continent').mean()
sub_df.groupby('continent').describe()

sub_df.groupby('continent')['gdpPercap'].mean()


# 5. Export results

result = sub_df.groupby('continent').mean()
result.to_csv('continent_result.csv')
# result.to_csv('continent_result.csv', index=False)
