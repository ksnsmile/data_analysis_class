"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Data Handling(2) - Missing Value & Good Datasets (Practice)
# =============================================================================


# 1. Load dataset

import pandas as pd

tips = pd.read_csv("tips.csv")
reg_c = pd.read_csv("reg_customer.csv")


# 2. Join datasets

merged_tips = tips.merge(reg_c, on='customerID', how='outer')
# merged_tips = tips.merge(reg_c, on='customerID')


# 3. Check missing values

merged_tips.isna()
merged_tips.isna().sum()
merged_tips.isna().all()
merged_tips.isna().any()
merged_tips.isna().sum(axis=1)
merged_tips.isna().all(axis=1)
merged_tips.isna().any(axis=1)

merged_tips.columns[merged_tips.isna().all()]
merged_tips.index[merged_tips.isna().all(axis=1)]

merged_tips.columns[merged_tips.isna().any()]
# ['name', 'age']

merged_tips.name.value_counts(dropna=False)
merged_tips.age.value_counts(dropna=False)


# 4. Deal with missing values

merged_tips_drop = merged_tips.dropna()

merged_tips_f0 = merged_tips.fillna(0)
merged_tips.name.fillna("abc00")

merged_tips_ff = merged_tips.fillna(method='ffill')
merged_tips_bf = merged_tips.fillna(method='bfill')


# 5. Re-structure Dataframe - Pivot

tips.columns
# ['customerID', 'total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
tips.drop("customerID", axis=1, inplace=True)

tips['tip_pct'] = tips['tip'] / tips['total_bill']

# tips.smoker.unique()
tips.pivot_table(values = "tip_pct", index = "sex", columns = "smoker", aggfunc="count", margins=True)
# tips.pivot_table(values = "tip_pct", index = "sex", columns = "smoker", aggfunc="mean", margins=True)
tips.pivot_table(values = ['tip_pct', 'size'], index = ['sex', 'day'], columns = 'smoker')
# tips.pivot_table(values = ['tip_pct', 'size'], index = ['sex', 'day'], columns = 'smoker', aggfunc="count", margins=True)

tips.pivot_table(values = "tip_pct", index = "sex")
tips.pivot_table(values = "tip_pct", index = ["sex", "smoker"])
tips.pivot_table(values = "tip_pct", index = "sex", columns = "smoker")

tips.groupby(["sex", "smoker"])[["tip_pct"]].describe()

