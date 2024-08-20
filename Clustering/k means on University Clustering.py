# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:33:49 2024

@author: ketan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Generate random numbers in the range 0 to 1
#and with uniform probability of 1/30
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
#crreate a empty dataframe with 0 rows and 2 columns
df_xy = pd.DataFrame(columns=["X","Y"])

#assign the values of the X and Y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)
'''

with dataX and Y, apply Kmeans model,
generate scatter plot
with scale/font=10
cmap=plt.cm.coolwarm:cool clor combination
'''
model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)
#############################################

Univ1=pd.read_excel("C:/7-clustering/University_Clustering.xlsx")
Univ1.describe()
Univ=Univ1.drop(['State'],axis=1)

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(Univ.iloc[:,1:])
'''
what will be the ideal cluster number,will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_)#total within sum of square
    '''
    Kmeans inertia , also known as Sum of Squares Errors(or SSE), Calculates the
    sum of the distances of all points within a cluster from the centroid off the
    point. It is the difference between the observed value and the predicted values
    '''
TWSS
#ask k value increases the TWSS value decreases
plt.plot(k.TWSS,'ro-');
plt.xlabel("No_of_cluster");
plt.ylabel("Total_within_SS")
'''
How to select value of k from elbow curve
when k chages from 2 to 3,the decrease 
in twss is higher than
when k chages from 3 to 4.
when k values changes from 5 to 6 decrease
'''
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()

#To save the file in folder we use this code. at top right give the exact path.
Univ.to_csv("kmeans_University.csv",encoding='utf-8')
import os
os.getcwd()


