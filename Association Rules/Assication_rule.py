# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:36:23 2024

@author: ketan
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#Sample dataset
transactions=[
    ['milk','bread','butter'],
    ['bread','eggs'],
    ['milk','bread','eggs','butter'],
    ['bread','eggs','butter'],
    ['milk','bread','eggs']
    ]
#Setp1:convert the dataset into a formate sutiable foe Apirori
tr=TransactionEncoder()
te_ary=tr.fit(transactions).transform(transactions)
df=pd.DataFrame(te_ary,columns=tr.columns_)

#step2: Apply the apirori algorthim to find frequent itemset
frequent_itemsets=apriori(df,min_support=0.5,use_colnames=True)

#Setp3:Generate association reules from frequent itemset
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)

#setp4: out of result
print('Frequent itensets:')
print(frequent_itemsets)

print('\nAssociation rules')
print(rules[['antecedents','consequents','support','confidence','lift']])



################################Example2#################################
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
healtcare_data=[
    ['Fever','Cough','COVID-19'],
    ['cough','sore Throat','Flu'],
    ['Fever','Cough','Shortness of breath','COVID-19'],
    ['Cough','Sore throat','Flue','Headache'],
    ['Fever','Body Ache','flu'],
    ['Fever','Cough','COVID-19','Shortness of breath'],
    ['Sore throat','cough','Headache'],
    ['Fatigue','Body Ache','flu'],
    ]
#Setp1:convert the dataset into a formate sutiable foe Apirori
tr=TransactionEncoder()
te_ary=tr.fit(healtcare_data).transform(healtcare_data)
df=pd.DataFrame(te_ary,columns=tr.columns_)

#step2: Apply the apirori algorthim to find frequent itemset
frequent_itemsets=apriori(df,min_support=0.3,use_colnames=True)

#Setp3:Generate association reules from frequent itemset
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)

#setp4: out of result
print('Frequent itensets:')
print(frequent_itemsets)

print('\nAssociation rules')
print(rules[['antecedents','consequents','support','confidence','lift']])

#output you can see 
#10         (COVID-19)     (Fever, Cough)    0.375        1.00  2.666667
#11            (Cough)  (Fever, COVID-19)    0.375        0.75  2.000000
#it means fever, cough means covid 19 and vice varese which has high changes of COVID-19

###################example3#################################
#Step 1: Simulate e-commerce transactions (products purchased per c

transactions=[
['Laptop", "Mouse', 'Keyboard'],
['Smartphone', 'Headphones'],
['Laptop', 'Mouse', 'Headphones'], 
['Smartphone', 'Charger', 'Phone Case'],
["Laptop", "Mouse", 'Monitor'],
['Headphones', 'Smartwatch'], 
['Laptop', 'Keyboard', 'Monitor'],
['Smartphone', 'Charger', 'Phone Case', 'Screen Protector'],
['Mouse', 'Keyboard', 'Monitor'],
['Smartphone', 'Headphones', 'Smartwatch']
]
#Setp1:convert the dataset into a formate sutiable foe Apirori
tr=TransactionEncoder()
te_ary=tr.fit(transactions).transform(transactions)
df=pd.DataFrame(te_ary,columns=tr.columns_)

#step2: Apply the apirori algorthim to find frequent itemset
frequent_itemsets=apriori(df,min_support=0.3,use_colnames=True)

#Setp3:Generate association reules from frequent itemset
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.5)

#setp4: out of result
print('Frequent itensets:')
print(frequent_itemsets)

print('\nAssociation rules')
print(rules[['antecedents','consequents','support','confidence','lift']])

#output you can see 











