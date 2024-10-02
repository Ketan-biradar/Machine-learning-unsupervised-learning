# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:16:06 2024

@author: ketan
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori,association_rules 

#Here we are going to use transactional data where in the size of each row is not consistent
#We can not use pandas to load this unstructured data 
#here function called open() is used 
#Create an empty list 
groceries=[]
with open("C:/11-Association Rules/groceries.csv.xls") as f:groceries=f.read()
