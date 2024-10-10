# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Exploration - File Handling
# =============================================================================


## Import CSV file ##

# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

import pandas as pd

# Relative path - check your working directory!
file_path = 'read_csv_sample.csv'

# Absolute path type 1 : 'C:/Class_Practice/read_csv_sample.csv'
# Absolute path type 2 : 'C:\\Class_Practice\\read_csv_sample.csv'

# read_csv() 
df1 = pd.read_csv(file_path)    # header=0
print(df1)

# read_csv(), header=None
df2 = pd.read_csv(file_path, header=None)
print(df2)

# read_csv(), with column names & header=None
df3 = pd.read_csv(file_path, names=['col_1', 'col_2', 'col_3', 'col_4'])
print(df3)

# read_csv(), with column names & header=0
df4 = pd.read_csv(file_path, header=0, names=['col_1', 'col_2', 'col_3', 'col_4'])
print(df4)

# What if header=2?

# Column names
df4.columns

# Change column names #1 - for all columns
df4.columns = ['changed_col_1', 'changed_col_2', 'changed_col_3', 'changed_col_4']

# Change column names #2 - for specific columns
df4.columns.values[0] = 'only_changed_col_1'

# Change column names #3 - for specific columns using rename()
df4.rename(columns={'changed_col_2':'column2'}, inplace=True)
# df4.rename(columns={'changed_col_3':'column3'})

# read_csv(), index_col=None
df5 = pd.read_csv(file_path, index_col=None)    # index_col=False
print(df5)

# read_csv(), index_col='c0'
df6 = pd.read_csv(file_path, index_col='c0')
print(df6)

# Index names
df5.index

# Change index names #1 - for all indexes
df5.index = ['row_1', 'row_2', 'row_3']

# Change index names #2 - for specific indexes
df5.index.values[0] = 'only_changed_row_1'

# Change index names #3 - for specific indexes using rename()
df5.rename(index={'only_changed_row_1':'index1'}, inplace=True)


## Export CSV file ##

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html

# Dictionary
data = {'name'  : [ 'Jerry', 'Riah', 'Paul'],
        'algol' : [ "A", "A+", "B"],
        'basic' : [ "C", "B", "B+"],
        'c++'   : [ "B+", "C", "C+"],
        }

df = pd.DataFrame(data)
df.set_index('name', inplace=True)   
# set_index(["column name"] or "column name")
# set_index(): Function to specify a specific column as an index
# If multiple columns are specified, group them into a list => Multi-Index
print(df)

# to_csv()
df.to_csv("df_sample.csv")

# to_csv(), without index
# Generally, it is stored without an index due to multi-index issues later 
# In particular, if it is an automatically assigned number, the index is not saved
df.to_csv("df_sample.csv", index=False)    # default = True

# to_csv(), without column names
df.to_csv("df_sample.csv", header=False)

# to_csv(), with column names
df.to_csv("df_sample.csv", header=['col_1', 'col_2', 'col_3'])


## Import Excel file ##

# https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

# read_excel()
df1 = pd.read_excel('남북한발전전력량.xlsx', engine='openpyxl')    # header=0

# engine types: Library that supports extraction of Excel file data
# - .xls : xlrd
# - .xlsx : openpyxl

print(df1)

# read_excel(), with sheet names (or numbers, start from 0)
df1 = pd.read_excel('남북한발전전력량_sheets.xlsx', engine='openpyxl', sheet_name="데이터-1")

# read_excel(), with sheet names, [sheet names]
# Return dictionary type
df2 = pd.read_excel('남북한발전전력량_sheets.xlsx', engine='openpyxl', sheet_name=["데이터-1", "데이터-2", "데이터-3"])

# read_excel(), with sheet names, None: All worksheets
df3 = pd.read_excel('남북한발전전력량_sheets.xlsx', engine='openpyxl', sheet_name=None)

df3["데이터-1"]

# Concatenate/Merge multiple DataFrames into one DataFrame
df_concat = pd.concat([df3["데이터-1"], df3["데이터-2"], df3["데이터-3"]], axis=0)


## Export Excel file ##

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html

# to_excel()
df.to_excel("df_sample.xlsx", engine="openpyxl")


## Export Excel file for Multiple DataFrames ## 
data1 = {'name' : [ 'Jerry', 'Riah', 'Paul'],
         'algol' : [ "A", "A+", "B"],
         'basic' : [ "C", "B", "B+"],
          'c++' : [ "B+", "C", "C+"]}

data2 = {'c0':[1,2,3], 
         'c1':[4,5,6], 
         'c2':[7,8,9], 
         'c3':[10,11,12], 
         'c4':[13,14,15]}

df1 = pd.DataFrame(data1)
df1.set_index('name', inplace=True)
print(df1)

df2 = pd.DataFrame(data2)
df2.set_index('c0', inplace=True)


# https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html?highlight=excelwriter#pandas.ExcelWriter

# ExcelWriter : Class that creates Excel workbook objects. Excel file itself
# to_excel
# - excel_writer : path-like, file-like, or ExcelWriter object => File path or existing ExcelWriter


# Export Excel file with multiple sheets for multiple DataFrames #1
# df1 -> 'sheet1', df2 -> 'sheet2'
writer = pd.ExcelWriter("df_excelwriter1.xlsx", engine="openpyxl")
df1.to_excel(writer, sheet_name="sheet1")
df2.to_excel(writer, sheet_name="sheet2")
writer.save()

# Export Excel file with multiple sheets for multiple DataFrames #2
with pd.ExcelWriter("df_excelwriter2.xlsx", engine="openpyxl") as writer:
    df1.to_excel(writer, sheet_name="sheet1")  
    df2.to_excel(writer, sheet_name="sheet2") 

# Append sheets to an existing Excel file 
# df -> 'sheet3'
# mode: 'a' -> append, 'w' -> write (default)
with pd.ExcelWriter("df_excelwriter2.xlsx", mode="a", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="sheet3")  
