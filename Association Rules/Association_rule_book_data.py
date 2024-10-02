# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:58:45 2024

@author: ketan

Problem Statement: -
Kitabi Duniya, a famous book store in India, which was 
established before Independence, the growth of the company was
incremental year by year, but due to online selling of books 
and wide spread Internet access its annual growth started to 
collapse, seeing sharp downfalls, you as a Data Scientist 
help this heritage book store gain its popularity back and 
increase footfall of customers and provide ways the business can
improve exponentially, apply Association RuleAlgorithm, explain 
the rules, and visualize the graphs for clear understanding of 
solution.

Business Objective
Minimize : Increase sale of Online book shopping
Maximaze : Sales of books


Data Dictionary

Name of features     Type Relevance      Description
0          ChildBks  Nominal  Relevant      Child Related books
1          YouthBks  Nominal  Relevant      Youth Related books
2           CookBks  Nominal  Relevant    Cooking Related books
3          DoItYBks  Nominal  Relevant      DoItY Related books
4            RefBks  Nominal  Relevant  Reference Related books
5            ArtBks  Nominal  Relevant        Art Related books
6           GeogBks  Nominal  Relevant  Geography Related books
7          ItalCook  Nominal  Relevant   ItalCook Related books
8         ItalAtlas  Nominal  Relevant  ItalAtlas Related books
9           ItalArt  Nominal  Relevant    ItalArt Related books
10         Florence  Nominal  Relevant   Florence Related books

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/11-Association Rules/book.csv.xls")
df.head()
df.tail()
#describle-5 number summary
df.describe()

df.shape
#2000 rows and 11 columns

df.columns
#columns name ['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
#'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence'],dtype='object')

#check the NUll value
df.isnull()
#False
df.isnull().sum()
#0 value

#box plot
#box plot on ChildBks Column
sns.boxplot(df.ChildBks)

#box plot on YouthBks column
sns.boxplot(df.YouthBks)
#we can see the outliers 

#box plot on all the Columns
sns.boxplot(df)
#There is some outliers on various columns


#Data preprocessing
df.dtypes
#All the data is in integer data type

duplicated=df.duplicated()
duplicated
# if there is duplicate records output- True
# if there is no duplicate records output-False

sum(duplicated)
#The output is 1680

# Outliers treatment on YouthBks
IQR=df.YouthBks.quantile(0.75)-df.YouthBks.quantile(0.25)
IQR
lower_limit=df.YouthBks.quantile(0.25)-1.5*IQR
upper_limit=df.YouthBks.quantile(0.75)+1.5*IQR


# Trimming

outliers_df=np.where(df.YouthBks>upper_limit,True,np.where(df.YouthBks<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df1=df.loc[~outliers_df] 
df.shape 
# (2000, 11)
df1.shape
# (1505, 11)

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

# We handled outliers, normalized data, analysis and EDA.
# Apriori algorithm extracted association rules  business objectives are needed.
