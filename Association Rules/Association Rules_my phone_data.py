# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:37:48 2024

@author: ketan

Problem Statement: - 
A Mobile Phone manufacturing company wants to launch its three 
brand new phone into the market, but before going with its traditional 
marketing approach this time it want to analyze the data of its
previous model sales in different regions and you have been hired 
as an Data Scientist to help them out, use the Association rules 
concept and provide your insights to the company’s marketing team to 
improve its sales.
"""
"""
Business Objective
Minimize : unliked color phones
Maximaze : Recommandation of good Saleing  colors phones

Data Dictionary

  Name of Features     Type Relevance   Description
0              red  Nominal  Relevant     red color
1            white  Nominal  Relevant   white color
2            green  Nominal  Relevant   green color
3           yellow  Nominal  Relevant  yellow color
4           orange  Nominal  Relevant  orange color
5             blue  Nominal  Relevant    blue color

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/11-Association Rules/myphonedata.csv.xls")
df.head()
df.tail()
#describle-5 number summary
df.describe()

df.shape
#11 rows,6 columns 

df.columns
#The columns Names(['red', 'white', 'green', 'yellow', 'orange', 'blue'], dtype='object')

#check the NUll value
df.isnull()
#False
df.isnull().sum()
#0 value

#box plot
#box plot on red Column
sns.boxplot(df.red)

#box plot on green Column
sns.boxplot(df.green)
#There is outliers on the green

#box plot on all the Columns
sns.boxplot(df)
#There is some outliers on various column

# mean
df.mean()
# mean of White  is 0.6363 and highest

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
sum(duplicated)
#The output is 3
df.drop_duplicates(inplace=True)
duplicate=df.duplicated()
sum(duplicate)

# Outliers treatment on green
IQR=df.green.quantile(0.75)-df.green.quantile(0.25)
IQR
lower_limit=df.green.quantile(0.25)-1.5*IQR
upper_limit=df.green.quantile(0.75)+1.5*IQR

# Trimming

outliers_df=np.where(df.green>upper_limit,True,np.where(df.green<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df1=df.loc[~outliers_df] 
df.shape 
# (8, 6)
df1.shape
# (6, 6)

sns.boxplot(df1)
#The outliers has remove 

# Normalization
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
# where the same colors showing in same columns with its 
# antecedents and consequents and its value of conviction.
