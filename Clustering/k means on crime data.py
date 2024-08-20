# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:31:55 2024

@author: ketan
"""
'''
Perform clustering for the crime data and identify the number of clusters

formed and draw inferences. Refer to crime data.csv dataset.

1. Business Problem

1.1. What is the business objective?
Crime Pattern Identification: Group states with similar crime rates to identify regional patterns or trends, which could help in allocating resources or targeting interventions.

Safety Indexing: Cluster states into groups based on safety levels (e.g., high-crime vs. low-crime states) for use in policy-making or public awareness campaigns.

Resource Allocation: Identify clusters of states with similar crime profiles to optimize the allocation of law enforcement resources.

Urbanization Impact Analysis: Analyze how the percentage of the urban population correlates with different crime rates by clustering states based on these factors.

Policy Effectiveness Evaluation: Group states to compare the effectiveness of different crime prevention policies across similar clusters.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

df=pd.read_csv("C:/7-clustering/crime_data.csv.xls")
df.describe()
df.head()
df.shape
df.info()
df.dtypes
null=df.isnull().sum()
print(null) #There is no Null value 

duplicated=df.duplicated() #The output is false means no duplicate

#To check the outliers we use boxplot
sns.boxplot (df['Murder']) #There is no outlier

sns.boxplot (df)#To check all the coloumn haveing outliers or not

sns.boxplot(df['Rape']) #There is outliers present

#To remove the outlier 
IQR=df.Rape.quantile(0.75)-df.Rape.quantile(0.25)
IQR #11.10001 is outof range that's y giving outliers

# Let apply winsorizer technique for outliers tratment on Rape columns

from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Rape'])

df_t=winsor.fit_transform(df[['Rape']])

# Check prious with outliers
sns.boxplot(df[['Rape']])
# Check the new outliers
sns.boxplot(df_t['Rape'])

#We see the all columns
df.columns
#Here columns Unnamed:0 is not required we drop the column
df=df.drop(['Unnamed: 0'],axis=1) #axis 1 for first column 
df.columns#To check the column is deleted or not
df.head()


# Normalize the data using norm function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(df.iloc[:,:])
info=df_norm.describe()

#We use now k-means algorithm
a=[]
k=list(range(2,8)) #which will be used as the number of clusters.2,3,4,5,6,7.
for i in k:
    kmeans=KMeans(n_clusters=i) #nitializes the k-means algorithm
    kmeans.fit(df_norm) #Fits the k-means model to the normalized data.
    a.append(kmeans.inertia_)

# total within sum of square

print(a)

# As k value increases the a the a value decreases
plt.plot(k,a,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_ss")
'''
How to select value of k from elbow curve
when k changes from 2 to 3 , then decrease
in a is higher than 
when k chages from 3 to 4
when k changes from 3 to 4.
Whwn k value changes from 5 to 6 decreases
in a is higher than when k chages 3 to 4 .
When k values changes from 5 to 6 decrease
in a is considerably less , hence considered k=3
'''

model=KMeans(n_clusters=3)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

# clust group

df['clust']=mb
df.head()
df=df.iloc[:,[7,0,1,2,3,4,5,6]]
df
df.iloc[:,2:8].groupby(df.clust).mean()
 
#To save the file in folder we use this code. at top right give the exact path.
df.to_csv("kcrime.csv",encoding='utf-8')
import os
os.getcwd()

