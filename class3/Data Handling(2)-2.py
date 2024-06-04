"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Data Handling(2) - Good Datasets
# =============================================================================


# 1. Pivot - melt()   see Material

import pandas as pd

pew = pd.read_csv('pew.csv')
print(pew.head())

##  Fix only one column

### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html?highlight=melt#pandas-dataframe-melt 

pew_long = pd.melt(pew, id_vars='religion')
pew_long_other = pew.melt(id_vars='religion')
print(pew_long.head())

pew_long = pd.melt(pew, id_vars='religion', var_name='income', value_name='count')
print(pew_long.head())

##  Fix  2 or more columns
billboard = pd.read_csv('billboard.csv')

billboard.head()
billboard.iloc[0:5, 0:7]

billboard_long = pd.melt(billboard, id_vars=['year', 'artist', 'track', 'time', 'date.entered'], var_name='week', value_name='rating')

print(billboard_long.head())


# 2. Handle duplicated data

print(billboard_long.shape)

## Duplicated data
print(billboard_long[billboard_long.track == 'Loser'].head())

billboard_songs = billboard_long[['year', 'artist', 'track', 'time', 'date.entered']] 
print(billboard_songs.shape)

## drop_duplicates()
billboard_songs = billboard_songs.drop_duplicates() 
print(billboard_songs.shape)

## Add 'id' column
billboard_songs['id'] = range(len(billboard_songs)) 
print(billboard_songs.head(10))

## merge()
billboard_ratings = billboard_long.merge(billboard_songs, on=['year', 'artist', 'track', 'time', 'date.entered']) 
print(billboard_ratings.shape)
billboard_ratings_dropna = billboard_ratings.dropna()


# 3. Column names

ebola = pd.read_csv('country_timeseries.csv')
print(ebola.columns)

## Extract data
print(ebola.iloc[:5, [0, 1, 2, 3, 10, 11]])

## Pivot using melt() - fix 'Date' and 'Day' columns
ebola_long = pd.melt(ebola, id_vars=['Date', 'Day'])
print(ebola_long.head())

## Split column names - split() for string data

### https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html#pandas.Series.str.split

variable_split = ebola_long.variable.str.split('_')

print(variable_split[:5])
print(type(variable_split))    # Type = Series
print(type(variable_split[0]))    # Type = list

### https://pandas.pydata.org/docs/reference/api/pandas.Series.str.get.html#pandas.Series.str.get

status_values = variable_split.str.get(0)    # extract & save status
country_values = variable_split.str.get(1)    # extract & save country

print(status_values[:5])
print(status_values[-5:])

print(country_values[:5])
print(country_values[-5:])

## Add new columns for status & country
ebola_long['status'] = status_values 
ebola_long['country'] = country_values
print(ebola_long.head())

### using 'expand=True' parameter
variable_split_df = ebola_long.variable.str.split('_', expand=True)
variable_split_df.columns = ['status-df', 'country-df']
ebola_long_concat = pd.concat([ebola_long, variable_split_df], axis=1)
print(ebola_long_concat.head())

ebola_long_concat.columns
# ['Date', 'Day', 'variable', 'value', 'status', 'country', 'status-df',
#        'country-df']
ebola_long_concat.drop(columns='variable' , inplace=True)
ebola_long_concat.drop(columns=['status', 'country'] , inplace=True)
ebola_long_concat.rename(columns={'value':'count'}, inplace=True)


# 4. Combine multiple columns into one

weather = pd.read_csv('weather.csv') 
print(weather.iloc[:5, :11])

## Pivot - melt()
weather_melt = pd.melt(weather, id_vars=['id', 'year', 'month', 'element'], var_name='day', value_name='temp') 
print(weather_melt.head())

### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html#pandas.DataFrame.pivot_table

## Pivot_table()    see Material
weather_tidy = weather_melt.pivot_table(
    index=['id', 'year', 'month', 'day'], 
    columns='element', 
    values='temp',
    dropna = False)    # with NaN

weather_tidy = weather_melt.pivot_table(
    index=['id', 'year', 'month', 'day'], 
    columns='element', 
    values='temp')    # without NaN

weather_tidy = pd.pivot_table(
    weather_melt,
    index=['id', 'year', 'month', 'day'], 
    columns='element', 
    values='temp')

print(weather_tidy)



