# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:52:06 2024

@author: ketan

Problem Statement: - 
A film distribution company wants to target audience based on their 
likes and dislikes, you as a Chief Data Scientist Analyze the data and 
come up with different rules of movie list so that the business 
objective is achieved.

Business Objective 
Minimize : dislikes movies or dissimilar movies recommandation
Maximaze : likes movies and similar type movies
Business constraints

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/11-Association Rules/my_movies.csv.xls")
df.head()
df.tail()
#describle-5 number summary
df.describe()

df.shape
#10 rows and 10 columns

df.columns
#The columns name are (['Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
#'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile'],dtype='object')

#check the NUll value
df.isnull()
#False
df.isnull().sum()
#0 value

#box plot
#box plot on Gladiator Column
sns.boxplot(df.Gladiator)

#box plot on LOTR1 Column
sns.boxplot(df.LOTR1)
#There is outliers on the LOTR1

#box plot on all the Columns
sns.boxplot(df)
#There is some outliers on various columns

# mean
df.mean()
# mean of Gladiator  is 0.7 and highest

# median
df.median()
# median of 3 columns is 1 

# mode
df.mode()

# standard deviation
df.std()

#Data preprocessing
df.dtypes
#All the data is in integer data type

duplicated=df.duplicated()
duplicated
# if there is duplicate records output- True
# if there is no duplicate records output-False

sum(duplicated)
#The output is 3
df.drop_duplicates(inplace=True)
duplicate=df.duplicated()
sum(duplicate)
#Now sum of duplicate is 0
#duplicated is removed

# Outliers treatment on LOTR
IQR=df.LOTR.quantile(0.75)-df.LOTR.quantile(0.25)
IQR
lower_limit=df.LOTR.quantile(0.25)-1.5*IQR
upper_limit=df.LOTR.quantile(0.75)+1.5*IQR

# Trimming

outliers_df=np.where(df.LOTR>upper_limit,True,np.where(df.LOTR<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df1=df.loc[~outliers_df] 
df.shape 
# (7, 10)
df1.shape
# (6, 10)

sns.boxplot(df1)
#The outliers has remove 

# Normalization

# Normalization function

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df1) 
# you can check the df_norm dataframe which is scaled between values from 0 and 1
b=df_norm.describe()
# Data is normalize in 0-1

#We are going to use apriori Algormithm
from mlxtend .frequent_patterns import apriori,association_rules

#is 0.0075 it must me between 0 and 1
frequent_itemsets = apriori(df,min_support=0.0075,max_len=4,use_colnames=True)

#Sort this support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#The support value have soted in descending order

#we will generate  association rules, This association rule will calculate all the matrix of each and every combination 
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)

rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

plt.bar(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs. Confidence')
plt.show()

# After applying the apriori and association_rules we found insights
# where the same movies showing in columns with its 
# antecedents and consequents and its value of conviction.


