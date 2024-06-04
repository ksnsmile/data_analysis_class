# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Data Handling(1) - Join Datasets
# =============================================================================


# 1. Join datasets using concat()

## Multiple datasets 
import pandas as pd

df1 = pd.read_csv('concat_1.csv') 
df2 = pd.read_csv('concat_2.csv') 
df3 = pd.read_csv('concat_3.csv')

## Join dataframes - axis=0
row_concat = pd.concat([df1, df2, df3])    # see Material
print(row_concat)

## Join dataframes - axis=1
column_concat = pd.concat([df1, df2, df3], axis=1) 
print(column_concat)

## Join dataframe with series
new_row_series = pd.Series(['n1', 'n2', 'n3', 'n4'])
series_concat = pd.concat([df1, new_row_series])

## Join one-row dataframe
new_row_df = pd.DataFrame([['n1', 'n2', 'n3', 'n4']], columns=['A', 'B', 'C', 'D']) 
print(new_row_df)

print(pd.concat([df1, new_row_df]))
print(pd.concat([df1, new_row_df], axis=1))

print(df1.append(new_row_df))    # using append()


# 2. Advanced options of concat()

## ignore_index
row_concat_i = pd.concat([df1, df2, df3], ignore_index=True) 
print(row_concat_i)

## axis =1
col_concat = pd.concat([df1, df2, df3], axis=1) 
print(col_concat)
print(col_concat['A'])

col_concat['new_col_list'] = ['n1', 'n2', 'n3', 'n4']    # add the column
print(col_concat)

col_concat = pd.concat([df1, df2, df3], axis=1, ignore_index=True) 
print(col_concat)

## Only common columns and common indexes 

### Rename columns

df1.columns
df2.columns
df3.columns

df1.columns = ['A', 'B', 'C', 'D'] 
df2.columns = ['E', 'F', 'G', 'H'] 
df3.columns = ['A', 'C', 'F', 'H']

print(df1)
print(df2)
print(df3)

### concat()
row_concat = pd.concat([df1, df2, df3]) 
print(row_concat)

### Common columns
print(pd.concat([df1, df2, df3], join='inner'))
print(pd.concat([df1,df3], ignore_index=False, join='inner'))

### Rename indexes

df1.index
df2.index
df3.index

df1.index = [0, 1, 2, 3] 
df2.index = [4, 5, 6, 7] 
df3.index = [0, 2, 5, 7]

print(df1)
print(df2)
print(df3)

### concat()
col_concat = pd.concat([df1, df2, df3], axis=1) 
print(col_concat)

### Common rows
print(pd.concat([df1, df3], axis=1, join='inner'))


# 3. merge()

## Multiple datasets 
person = pd.read_csv('survey_person.csv') 
site = pd.read_csv('survey_site.csv') 
survey = pd.read_csv('survey_survey.csv') 
visited = pd.read_csv('survey_visited.csv')

print(person)
print(site)
print(visited)
print(survey)

## merge() - site & visited
visited_subset = visited.loc[[0, 2, 6]]

o2o_merge = site.merge(visited_subset, left_on='name', right_on='site') 
print(o2o_merge)

m2o_merge = site.merge(visited, left_on='name', right_on='site') 
print(m2o_merge)

## merge() - person & survey, visited & survey
ps = person.merge(survey, left_on='ident', right_on='person') 
vs = visited.merge(survey, left_on='ident', right_on='taken')

print(ps)
print(vs)

ps_vs = ps.merge(vs, left_on=['ident', 'taken', 'quant', 'reading'], right_on=['person', 'ident', 'quant', 'reading'])

print(ps_vs.loc[0])
