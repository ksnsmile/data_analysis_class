"""
Spyder Editor

This is a temporary script file.
"""


# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Practice Project(1)
# =============================================================================


# 1. Load Datasets

import pandas as pd
import numpy as np
import seaborn as sns

titanic = sns.load_dataset("titanic")


# 2. Exploratory Data Analysis (EDA) 
#    - an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods 

titanic.head()

# =============================================================================
## Feature Information:
# -survived : 0 = No, 1 = Yes/survived
# -pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# -sex : male, female
# -age : age
# -sibsp : number of siblings/spouses aboard the Titanic
# -parch : number of parents/children aboard the Titanic
# -fare : fare
# -embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# -class : Ticket class First = 1st, Second = 2nd, Third = 3rd
# -who : man, woman, child
# -adult_male : True = male, False = female 
# -deck : A, B, C, D, E, F, G
# -embark_town : Cherbourg, Queenstown, Southampton
# -alive : no, yes = alive/survived
# -alone : False, True = alone
# =============================================================================

titanic.shape
titanic.info()

# =============================================================================
#  #   Column       Non-Null Count  Dtype   
# ---  ------       --------------  -----   
#  0   survived     891 non-null    int64   
#  1   pclass       891 non-null    int64   
#  2   sex          891 non-null    object  
#  3   age          714 non-null    float64  => missing values
#  4   sibsp        891 non-null    int64   
#  5   parch        891 non-null    int64   
#  6   fare         891 non-null    float64 
#  7   embarked     889 non-null    object  => missing values
#  8   class        891 non-null    category
#  9   who          891 non-null    object  
#  10  adult_male   891 non-null    bool    
#  11  deck         203 non-null    category  => missing values
#  12  embark_town  889 non-null    object  => missing values 
#  13  alive        891 non-null    object  
#  14  alone        891 non-null    bool   
# =============================================================================

titanic.isnull()
titanic.isnull().sum()

# =============================================================================
# survived         0
# pclass           0
# sex              0
# age            177
# sibsp            0
# parch            0
# fare             0
# embarked         2
# class            0
# who              0
# adult_male       0
# deck           688
# embark_town      2
# alive            0
# alone            0
# =============================================================================


## Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

### function for bar chart
def bar_chart(feature):
    survived = titanic[titanic['survived'] == 1][feature].value_counts()
    dead = titanic[titanic['survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['survived','dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))

# =============================================================================
# Bar Chart for Categorical Features : 
# -pclass/class
# -sex/who/adult_male
# -sibsp/alone  
# -parch/alone 
# -embarked/embark_town
# -deck
# =============================================================================

# survived = titanic[titanic['survived'] == 1]['pclass'].value_counts()
# dead = titanic[titanic['survived']==0]['pclass'].value_counts()
# df = pd.DataFrame([survived, dead])
# df.index = ['survived','dead']
    
bar_chart('pclass')
### => 1st class more likely survivied than other classes
### => 3rd class more likely dead than other classes

bar_chart('sex')    
### = > women more likely survivied than men

bar_chart('sibsp')
### a person aboarded with more than 2 siblings or spouse more likely survived
### a person aboarded without siblings or spouse more likely dead
bar_chart('parch')
### a person aboarded with more than 2 parents or children more likely survived
bar_chart('alone')
### a person aboarded alone more likely dead

bar_chart('embarked')
### a person aboarded from C slightly more likely survived
### a person aboarded from Q more likely dead
### a person aboarded from S more likely dead

bar_chart('deck')


# 3. Pre-processing - Feature Engineering

## Delete unnecessary columns - class, who, adult_male, deck, embarked_town, alive, alone
titanic_pre = titanic.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'])

## Handle missing values
titanic_pre.isnull().sum()    # age, embarked

### age
titanic_pre['age'] = titanic.age.fillna(titanic.age.median())    # median = 28.0
titanic_pre.loc[titanic['age'].isna()]['age']

### embarked
titanic.embarked.value_counts()    # mode = 'S'
titanic_pre['embarked'] = titanic.embarked.fillna('S')
titanic_pre.loc[titanic['embarked'].isna()]['embarked']

## Handle categorical columns

titanic_pre.head()
titanic_pre.dtypes

titanic_pre['pclass'].value_counts()
titanic_pre['sex'].value_counts()
titanic_pre['embarked'].value_counts()

### sex - label encoding
titanic_pre['sex']=titanic_pre['sex'].replace(['female','male'],[0,1])

### pclass & embarked - one-hot encoding
titanic_pre['pclass'] = titanic_pre['pclass'].astype('category') 
titanic_pre['embarked'] = titanic_pre['embarked'].astype('category')
titanic_pre.dtypes

titanic_pre = pd.get_dummies(titanic_pre)

## Handle continuous columns - binning

### fare
titanic_pre['fare'].describe()
                   
titanic_pre.loc[titanic_pre['fare'] <= 14, 'fare'] = 0
titanic_pre.loc[(titanic_pre['fare'] > 14) & (titanic_pre['fare'] <= 31), 'fare'] = 1
titanic_pre.loc[(titanic_pre['fare'] > 31) & (titanic_pre['fare'] <= 100), 'fare'] = 2
titanic_pre.loc[titanic_pre['fare'] > 100, 'fare'] = 3

### age
titanic_pre['age'].describe()
                   
titanic_pre.loc[titanic_pre['age'] <= 20, 'age'] = 0
titanic_pre.loc[(titanic_pre['age'] > 20) & (titanic_pre['age'] <= 30), 'age'] = 1
titanic_pre.loc[(titanic_pre['age'] > 30) & (titanic_pre['age'] <= 40), 'age'] = 2
titanic_pre.loc[(titanic_pre['age'] > 40) & (titanic_pre['age'] <= 50), 'age'] = 3
titanic_pre.loc[(titanic_pre['age'] > 50) & (titanic_pre['age'] <= 60), 'age'] = 4
titanic_pre.loc[titanic_pre['age'] > 60, 'age'] = 5

### X, Y variables
X = titanic_pre.drop(columns=['survived'])
Y = titanic_pre.iloc[:,0]


# 4. Modeling

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

sk_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scoring = 'accuracy'

## Decision Tree

dt_m = DecisionTreeClassifier(random_state=0)
dt_score = cross_val_score(dt_m, X, Y, cv=sk_fold, n_jobs=1, scoring=scoring)
print(dt_score)
round(np.mean(dt_score)*100, 2)   

## Logistic Regression

lr_m = LogisticRegression(random_state=0)
lr_score = cross_val_score(lr_m, X, Y, cv=sk_fold, n_jobs=1, scoring=scoring)
print(lr_score)
round(np.mean(lr_score)*100, 2)    

## RandomForest

rf_m = RandomForestClassifier(n_estimators=25, random_state=0)
rf_score = cross_val_score(rf_m, X, Y, cv=sk_fold, n_jobs=1, scoring=scoring)
print(rf_score)
round(np.mean(rf_score)*100, 2)   

## Support Vector Machines (SVM)

sv_m = SVC(random_state=0)
sv_score = cross_val_score(sv_m, X, Y, cv=sk_fold, n_jobs=1, scoring=scoring)
print(sv_score)
round(np.mean(sv_score)*100, 2)    

## Naive Bayes

nb_m = GaussianNB()
nb_score = cross_val_score(nb_m, X, Y, cv=sk_fold, n_jobs=1, scoring=scoring)
print(nb_score)
round(np.mean(nb_score)*100, 2)    

## KNN

knn_m = KNeighborsClassifier()
knn_score = cross_val_score(knn_m, X, Y, cv=sk_fold, n_jobs=1, scoring=scoring)
print(knn_score)
round(np.mean(knn_score)*100, 2)   

## AdaBoost

ada_m = AdaBoostClassifier(n_estimators=10, random_state=0)
ada_score = cross_val_score(ada_m, X, Y, cv=sk_fold, n_jobs=1, scoring=scoring)
print(ada_score)
round(np.mean(ada_score)*100, 2)    
