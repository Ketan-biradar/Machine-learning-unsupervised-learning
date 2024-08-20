# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:25:18 2024

@author: ketan
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib .pyplot as plt
from sklearn.cluster import KMeans
'''
Business Objective 
Minimize :To reduce the airlines which have less engagement route
Maximaze :To increase the airlines where customer are more engageing route with the loyalty program
          To increase customer satisfaction
'''
df=pd.read_excel("C:/7-clustering/EastWestAirlines.xlsx")
df.describe()
df.head()
df.shape
#There are 3999 rows and 12 Columns
df.info()
df.dtypes
#There are all integer data types
df.isnull()
#False 
null=df.isnull().sum()
null
#0
df.duplicated()
#False
df.columns
#The colums are ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
#Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
#Days_since_enroll', 'Award?

#To check the outliers 
sns.boxplot(df['Balance'])
#There are many outliers in the colums Balance

sns.boxplot(df['Qual_miles'])
#There are many outliers in the colums Qual_miles

sns.boxplot(df)
#To check all the outliers in the data sets

#use Scatterplot on column to see the relation in between two coloum
#In column "Balance" and "Days_since_enroll" we see the data is more between 0 to 1
sns.set_style("whitegrid");
sns.FacetGrid(df,) \
   .map(plt.scatter, "Balance", "Days_since_enroll") \
   .add_legend();
plt.show();

#To remove the outlier 
IQR=df.Balance.quantile(0.75)-df.Balance.quantile(0.25)
IQR 

# Let apply winsorizer technique for outliers tratment Balance

from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Balance'])

df_t=winsor.fit_transform(df[['Balance']])

# Check prious with outliers
sns.boxplot(df[['Balance']])
# Check the new outliers
sns.boxplot(df_t['Balance'])

#drop the columns
df=df.drop(["ID#"],axis=1)
df.columns
df.head()

# Normalize the data using norm function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(df.iloc[:,:])
info=df_norm.describe()

#We can use k-means algorithm
a=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    a.append(kmeans.inertia_)
print(a)
plt.plot(k,a,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_ss")

model=KMeans(n_clusters=3)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

# clust group

df['clust']=mb
df.head()

#To save the file in folder we use this code. at top right give the exact path.
df.to_csv("kAirlines.csv",encoding='utf-8')
import os
os.getcwd()

'''
starting,Scatterplot,IQR, Winsorizer ,Normalize,K-means
'''