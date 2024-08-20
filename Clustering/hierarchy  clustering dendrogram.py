# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:19:41 2024

@author: ketan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Univ1=pd.read_excel("C:/7-clustering/University_Clustering.xlsx")
Univ1.describe()
Univ=Univ1.drop(['State'],axis=1) # we use to delect the state coloumns Here axis=1 for coloum

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(Univ.iloc[:,1:])
b=df_norm.describe()

#we want to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#here linkage function give us hierarachy
#We use complete linkage method
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("hierarchy Clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
 
