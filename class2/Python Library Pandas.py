# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Python Library, Pandas
# =============================================================================


### pandas

import numpy as np
import pandas as pd

data = {'Name':['S1','S2','S3'],
        'Age':[25, 28, 22],
        'Score':np.array([95, 80, 75])}
print(data)

df=pd.DataFrame(data)
print(df)

df=pd.DataFrame(data, index=['row1', 'row2', 'row3'])
print(df)

data2=[['S1', 25, 95],
       ['S2', 28, 80],
       ['S3', 22, 75]]
print(data2)

df=pd.DataFrame(data2)
print(df)

df=pd.DataFrame(data2,
                index=['row1', 'row2', 'row3'],
                columns=['Name', 'Age', 'Score'])
print(df)

df['Name'] 
df['Age']
df['Score']
df[['Name', 'Score']]

df['row1']
df.loc['row1']
df.loc[['row1', 'row3']]

df.loc['row1', 'Name'] 
df.loc[:, 'Name'] 
df.loc[:, ['Name', 'Score']] 
df.loc[:,'Name':'Score']


df.iloc[0,0]
df.iloc[:, [0,2]]
df.iloc[::2, [0,2]] 
df.iloc[-1,:] 
df.iloc[-1::-1,:]

df.head(1)  
df.head(2)

df.tail(1) 
df.tail(2)

df.info()
df.describe()
 
df2=df.copy()
df2.loc['row2','Score'] = np.NaN
df2
df2['Age'].nunique()   
df2['Age'].unique()
df2['Score'].nunique()   # dropna=True
df2['Score'].unique()   # include NA values

df2['Score'].value_counts()   # dropna=True, ascending=False
df3=df2.copy()
df3.loc['row3', 'Score'] = df3.loc['row1', 'Score']
df3['Score'].value_counts()

df3['Score'].count()
df3['Age'].count()

df['Score'].sum()
df.max()
df['Score'].std()
df.describe()


# =============================================================================
# nunique() : 고유한 값 개수
# unique() : 고유한 값
# value_counts() : 고유한 값과 그 개수 (내림차순, ascending=False)
# count() : 전체 고유한 값 개수 (Number of non-null values in the Series)
# =============================================================================


df4=df.copy()
df4=df4.iloc[:,[0,2,1]]
df4

data={'Class':['A','B','C','A','B','C','C'],
      'Name':['S1','S2','S3','S4','S5','S6','S7'],
      'Age':[20, 19, 21, 22, 24, 25, 26],
      'Score': [90, 95, 75, 80, 70, 85, 90]}

df=pd.DataFrame(data)
df

df['Score'] >= 80
df.loc[df['Score'] >= 80]
df.loc[df['Score'] >= 80, 'Name']
df.loc[df['Score'] >= 80, ['Name', 'Age']]

df['Result']=None
df
df.loc[df['Score'] >= 80, 'Result'] = 'Pass'
df.loc[df['Score'] < 80, 'Result'] = 'Fail'

idx=(df['Result']=='Pass')
df.loc[idx]
df.loc[idx].sort_values('Score')
df_sorted = df.loc[idx].sort_values('Score', ascending=False)
df_sorted

df_sorted.to_excel('data_sorted.xlsx', index=False)
# df_sorted.to_excel('data_sorted.xlsx') 

df_import = pd.read_excel('data_sorted.xlsx')
df_import

df.groupby(by='Class').mean()
df.groupby(by='Class').count()
df.groupby(by='Class').std() 
df.groupby(by='Class').describe()

df.plot.bar('Name', 'Score') 
df.plot.bar('Name', ['Score','Age'])


df.loc[[0,2], 'Score'] = np.NaN
df
df.isnull()    # isna()
df.dropna()

value=0
df.fillna(value)

df.replace(np.NaN, -1)

df
df.interpolate()
inter_df = df.interpolate()

def add_one(x):
    return x+1

add_one(10)
df
df['Age'].apply(add_one)
df['Score'].apply(np.square)

df.filter(regex='[NR]') 
df.filter(regex='[NRS]')
df.filter(regex='[S]')

df_vertial = pd.concat([df, df]) 
df_vertial
df_horizontal = pd.concat([df, df], axis=1) 
df_horizontal

df.to_excel('data_excel.xlsx', index=False)
df.to_csv('data_text.txt', sep='\t', index=False)    # sep=','
df.to_pickle('data_pickle.pkl') 

df_read_excel=pd.read_excel('data_excel.xlsx')
df_read_excel

df_read_text=pd.read_csv('data_text.txt', delimiter='\t') 
df_read_text

df_read_pickle=pd.read_pickle('data_pickle.pkl')
df_read_pickle


















