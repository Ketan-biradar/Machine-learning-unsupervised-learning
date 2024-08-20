# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:59:56 2024

@author: ketan
""" 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from feature_engine.outliers import Winsorizer

# Load dataset
df = pd.read_excel("C:/7-clustering/Telco_customer_churn.xlsx")

# Five number summary
print(df.describe())

# Shape of DataFrame
print(df.shape)

# Display the first few rows
print(df.head())

# Check data types
print(df.dtypes)

# Convert 'Total Charges' to numeric (considering it might have been read as a string)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Scatter plots
sns.set_style("whitegrid")

# Scatter plot for 'Offer' vs 'Quarter'
sns.FacetGrid(df).map(plt.scatter, "Offer", "Quarter").add_legend()
plt.show()

# Scatter plot for 'Offer' vs 'Total Charges'
sns.FacetGrid(df).map(plt.scatter, "Offer", "Total Charges").add_legend()
plt.show()

# Histograms
sns.histplot(df['Total Revenue'], kde=True)
plt.show()

sns.histplot(df['Total Charges'], kde=True)
plt.show()

# Box plot for the entire DataFrame
sns.boxplot(data=df)
plt.show()

# Check for duplicate records
duplicate = df.duplicated()
print(f"Number of duplicate rows: {sum(duplicate)}")

# Drop unwanted columns
df = df.drop(['Customer ID', 'Offer'], axis=1)

# Convert categorical columns to dummy/indicator variables
df1 = pd.get_dummies(df, drop_first=True)

# Check the shape after dummy variable conversion
print(df1.shape)

# Rename columns to be more descriptive
df1 = df1.rename(columns={
    'Referred a Friend_No': 'Referred a Friend',
    'Phone Service_No': 'Phone Service',
    'Multiple Lines_No': 'Multiple Lines',
    'Internet Service_No': 'Internet Service',
    'Online Security_No': 'Online Security',
    'Online Backup_No': 'Online Backup',
    'Device Protection Plan_No': 'Device Protection Plan',
    'Premium Tech Support_No': 'Premium Tech Support',
    'Streaming TV_No': 'Streaming TV',
    'Streaming Movies_No': 'Streaming Movies',
    'Streaming Music_No': 'Streaming Music',
    'Unlimited Data_No': 'Unlimited Data',
    'Paperless Billing_No': 'Paperless Billing'
})


# Determine the optimal number of clusters using the Elbow method
inertia = []
k_range = list(range(2, 8))
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df1)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Fit the KMeans model with the chosen number of clusters (e.g., 3)
model = KMeans(n_clusters=3)
df1['clust'] = model.fit_predict(df1)

# Add cluster labels to the original DataFrame
df2 = df.copy()
df2['clust'] = df1['clust']

# Save the result to a CSV file
df2.to_csv("ktelco.csv", encoding="utf-8", index=False)

import os
print(os.getcwd())
