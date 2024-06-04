"""
Spyder Editor

This is a temporary script file.
"""


# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Visualization(2)
# =============================================================================

# =============================================================================
# Seaborn library
# https://seaborn.pydata.org/

# Graph types :
# 1. Relational plots (관계형 그래프)
# 2. Distribution plots (분산/분포형 그래프)
# 3. Categorical plots (범주형 그래프)
# 4. Regression plots (회귀형 그래프)
# 5. Matrix plots (매트릭스형 그래프)
# 6. Multi-plot grids (복수형 그래프)
# =============================================================================

import seaborn as sns 
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
tips.head()


# 1. Relational plots

# =============================================================================
## 1-1.  scatterplot()

# x, y : Variables that specify positions on the x and y axes
# hue : Grouping variable that will produce points with different colors
# data : Input data structure
# style : Grouping variable that will produce points with different markers
# =============================================================================

### x, y, data - Passing long-form data and assigning x and y will draw a scatter plot between two variables
sns.scatterplot(data=tips, x="total_bill", y="tip")

### hue - Assigning a variable to hue will map its levels to the color of the points
tips['time'].unique()    # ['Dinner', 'Lunch']
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")

### style - Assigning the same variable to style will also vary the markers and create a more accessible plot
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="time")
tips['day'].unique()    # ['Sun', 'Sat', 'Thur', 'Fri']
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time")

### Mulitple graphs
fig, axes = plt.subplots(2,2, figsize=(10,10))

sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[0, 0])
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", ax=axes[0, 1])
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="time", ax=axes[1, 0])
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time", ax=axes[1, 1])

plt.show()


## 1-2.  lineplot()

fmri = sns.load_dataset('fmri')
fmri.head()

### blue area = Confidence interval (신뢰구간)
### blue line = Estimated regression line (추정 회귀선)
sns.lineplot(data=fmri, x="timepoint", y="signal")
sns.lineplot(data=fmri, x="timepoint", y="signal", ci=70)
sns.lineplot(data=fmri, x="timepoint", y="signal", ci='sd')
sns.lineplot(data=fmri, x="timepoint", y="signal", ci=None)
sns.lineplot(data=fmri, x="timepoint", y="signal", estimator=None)
sns.lineplot(data=fmri, x="timepoint", y="signal", estimator='std', ci=None)

### hue
fmri['event'].unique()    # ['stim', 'cue']
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")

### hue & style
fmri['region'].unique()    # ['parietal', 'frontal']
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="region", style="event")

### markers & dashes
sns.lineplot(
    data=fmri,
    x="timepoint", y="signal", hue="event", style="event",
    markers=True, dashes=False)

# =============================================================================
## 1-3. relplot()

# kind : Kind of plot to draw, corresponding to a seaborn relational plot. Options are {“scatter”, “line”}
#        (default = scatter)
# row, col : Variables that define subsets to plot on different facets
# Easily draw multiple graphs at once (within relational plots)
# =============================================================================

sns.relplot(data=tips, x="total_bill", y="tip", hue="day")

### col - Assigning a col variable creates a faceted figure with multiple subplots arranged across the columns of the grid
sns.relplot(data=tips, x="total_bill", y="tip", hue="day", col="time")
 
### row - Different variables can be assigned to facet on both the columns and rows
sns.relplot(data=tips, x="total_bill", y="tip", hue="day", col="time", row="sex")

### kind = line
sns.relplot(
    data=fmri, x="timepoint", y="signal", col="region",
    hue="event", style="event", kind="line")


# 2. Distribution plots

# =============================================================================
# ## 2-1. histplot()

# bins
# binwidth : Width of each bin
# stat : Aggregate statistic to compute in each bin (default = 'count')
# - count: show the number of observations in each bin 
# - frequency: show the number of observations divided by the bin width
# - probability: or proportion: normalize such that bar heights sum to 1
# - percent: normalize such that bar heights sum to 100
# - density: normalize such that the total area of the histogram equals 1
# kde : If True, compute a kernel density estimate to smooth the distribution and show on the plot as (one or more) line(s)
# multiple : Approach to resolving multiple elements when semantic mapping creates subsets
#            Options are {“layer”, “dodge”, “stack”, “fill”}    
# =============================================================================
### pip install seaborn --upgrade 
sns.__version__    # '0.12.1'

### Assign a variable to x to plot a univariate distribution along the x axis
sns.histplot(data=tips, x="total_bill")

### Flip the plot by assigning the data variable to the y axis
sns.histplot(data=tips, y="total_bill")

### bins & binwidth
sns.histplot(data=tips, x="total_bill", bins=10)
sns.histplot(data=tips, x="total_bill", binwidth=10)

### stat
sns.histplot(data=tips, x="total_bill", bins=10, stat = "probability")
sns.histplot(data=tips, x="total_bill", bins=10, stat = "frequency")

### kde
sns.histplot(data=tips, x="total_bill", bins=10, kde=True)

### hue & multiple
sns.histplot(data=tips, x="total_bill", bins=10, hue = "day", multiple = "stack")
sns.histplot(data=tips, x="total_bill", bins=10, hue = "day", multiple = "layer")
sns.histplot(data=tips, x="total_bill", bins=10, hue = "day", multiple = "fill")
sns.histplot(data=tips, x="total_bill", bins=10, hue = "day", multiple = "dodge", kde=True)

# =============================================================================
## 2-2. displot()

# kind : Kind of plot to draw, corresponding to a seaborn distribution plot. Options are {“hist”, “kde”, “ecdf”}
#        hist => histplot(), kde => kdeplot(), ecdf => ecdfplot()
#        (default = hist)
# rug : If True, show each observation with marginal ticks => rugplot()
# Easily draw multiple graphs at once (within distribution plots)
# =============================================================================

sns.displot(data=tips, x="total_bill", hue="time", col="day")

### kind & rug
sns.displot(data=tips, x="total_bill", col="time", kind = "kde", rug=True)


# 3. Categorical plots

# =============================================================================
## 3-1. barplot()

# x : categorical variable
# y : continuous (numeric) variable
# ci : errorbar - size of confidence intervals to draw around estimated values
#      Options are {“float”, “sd”, “None”}
# =============================================================================

sns.barplot(data=tips, x ="day", y ="tip")
sns.barplot(data=tips, y ="day", x ="tip")
sns.barplot(x="day", y="tip", hue="smoker", data=tips)

### errorbar - None, float = confidence interval, sd = standard deviation
sns.barplot(x="day", y="tip", errorbar=None, data=tips)
sns.barplot(x="day", y="tip", errorbar=('ci', 80), data=tips)
sns.barplot(x="day", y="tip", errorbar="sd", data=tips)

# =============================================================================
## 3-2. countplot()

# Count the number of identical data (same as histplot)
# histplot : continuous data, countplot : categorical data
# =============================================================================

titanic = sns.load_dataset("titanic")
titanic["class"].unique()    # ['First', 'Second', 'Third']
titanic["class"].value_counts()
# Third     491
# First     216
# Second    184
sns.countplot(x="class", data=titanic)
titanic["who"].unique()    # ['man', 'woman', 'child']
sns.countplot(x="class", hue="who", data=titanic)
sns.countplot(y="class", hue="who", data=titanic)

# =============================================================================
## 3-3. boxplot()

# One variable (x or y): only continuous variable
# Two variables (x and y): continuous variable and categorical variable
# =============================================================================

### one variable
sns.boxplot(y="total_bill", data=tips)
sns.boxplot(x="total_bill", data=tips)

### two variables
sns.boxplot(x="day", y="total_bill", data=tips)
sns.boxplot(x="day", y="total_bill", hue="time", data=tips,  linewidth=2.5)
sns.boxplot(x="time", y="tip", data=tips, order=["Dinner", "Lunch"]) 

# =============================================================================
## 3-4. violinplot()

# boxplot + kde => Data distribution
# split : When using hue nesting with a variable that takes two levels, 
#         if True, draw half of a violin for each level (default = False)
# =============================================================================

sns.violinplot(x="day", y="total_bill", data=tips)
sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips)

### split - Draw split violins to compare the across the hue variable
sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, split=True)

# =============================================================================
## 3-5. stripplot()

# scatterplot : relationship between continuous and continuous variables
# stripplot : relationship between continuous and categorical variables
# =============================================================================

sns.stripplot(x="day", y="total_bill", data=tips)
sns.stripplot(x="sex", y="total_bill", hue="day", data=tips)

### Draw boxplot() and stripplot() together
### Draw strips of observations on top of a boxplot
fig, ax = plt.subplots()
ax = sns.boxplot(x="tip", y="day", data=tips)
ax = sns.stripplot(x="tip", y="day", data=tips, color=".3")

### Draw strips of observations on top of a violinplot
ax = sns.violinplot(x="day", y="total_bill", data=tips, inner=None, color=".8")
ax = sns.stripplot(x="day", y="total_bill", data=tips)

sns.swarmplot(x="day", y="total_bill", data=tips)

# =============================================================================
## 3-6. catplot()

# kind : Kind of plot to draw, corresponding to a seaborn categorical plot 
#        Options are {“strip”, “swarm”, “box”, “violin”, “boxen”, “point”, “bar”, “count”}
#        (default = strip)
# Easily draw multiple graphs at once (within categorical plots)
# =============================================================================

sns.catplot(x="day", y="tip", col="sex", data=tips)
sns.catplot(x="day", y="tip", col="sex", data=tips, kind="bar")


# 4. Regression plots

# =============================================================================
## 4-1. regplot()

# Plot data and a linear regression model fit
# scatterplot() + lineplot()
# =============================================================================

sns.regplot(x="total_bill", y="tip", data=tips)
sns.regplot(x="total_bill", y="tip", data=tips, marker = "+", color="g")
 
# =============================================================================
## 4-2. lmplot()

# Plot data and regression model fits across a FacetGrid
# Better than regplot()
# hue, col, row
# Easily draw multiple graphs at once (within regression plots)
# ci : int in [0, 100] or None, Size of the confidence interval for the regression estimate (default = 95)
# col_wrap : “wrap” the column variable at this width, so that the column facets span multiple rows
# height : height (in inches) of each facet
# =============================================================================

sns.lmplot(x = "total_bill", y = "tip", hue = "smoker", data = tips)

### ci
sns.lmplot(x = "total_bill", y = "tip", hue = "smoker", data = tips, ci=50)

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, markers=["o", "x"])
sns.lmplot(x = "total_bill", y = "tip", col= "smoker", hue = "smoker", data = tips)
sns.lmplot(x = "total_bill", y = "tip", col= "smoker", data = tips)

### col_wrap & height
sns.lmplot(x="total_bill", y="tip", col="day", hue="day", data=tips, col_wrap=2, height=3)
sns.lmplot(x="total_bill", y="tip", row="sex", col="time", data=tips, height=3)


# 5. Matrix plots

# =============================================================================
## 5-1. heatmap()

# data : rectangular dataset, 2D dataset that can be coerced into an ndarray
# vmin, vmax : values to anchor the colormap
# annot : If True, write the data value in each cell (default = None)
# =============================================================================

tips['size'].unique()    # [2, 3, 4, 1, 6, 5]
pivot = tips.pivot_table(index='day', columns='size', values='tip')
print(pivot)
sns.heatmap(pivot, cmap='Blues', annot=True) 
sns.heatmap(pivot, cmap='Blues', annot=True, vmin=6, vmax=0)

titanic_corr = titanic.corr()
sns.heatmap(titanic_corr, annot=True, cmap="YlGnBu")    # cmap = "YlGnBu", "Spectral", "RdBu", etc.
sns.heatmap(titanic_corr, annot=True, cmap="Spectral", vmin=-1, vmax=1) 


# 6. Multi-plot grids

# =============================================================================
## 6-1. pairplot() & PairGrid()

# Plot pairwise relationships in a dataset
# corner : If True, don’t add axes to the upper (off-diagonal) triangle of the grid
# kind : Kind of plot to make, Options are {‘scatter’, ‘kde’, ‘hist’, ‘reg’}
#        (default = scatter)
# =============================================================================

sns.pairplot(data = tips)

### corner
sns.pairplot(data = tips, corner=True)
sns.pairplot(data = tips, hue="sex")
sns.pairplot(data = tips, hue="sex", markers=["o", "s"])

### kind = kde & hist
sns.pairplot(data = tips, kind="kde")
sns.pairplot(data = tips, kind="hist")
sns.pairplot(data = tips, kind="reg")

### PairGrid()
pair_grid = sns.PairGrid(tips)
pair_grid.map_upper(sns.regplot) 
pair_grid.map_lower(sns.kdeplot) 
pair_grid.map_diag(sns.histplot) 

### FacetGrid()
sns.FacetGrid(tips, col="time", row="sex")    # Assign column and/or row variables to add more subplots to the figure

g = sns.FacetGrid(tips, col="time",  row="sex")
g.map(sns.scatterplot, "total_bill", "tip")    # To draw a plot on every facet, pass a function and the name of one or more columns in the dataframe to FacetGrid.map()

g = sns.FacetGrid(tips, col="time")
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.refline(y=tips["tip"].median(), color='red')    # method: refline()

g = sns.FacetGrid(tips, col="time",  row="sex")
g.map_dataframe(sns.histplot, x="total_bill")