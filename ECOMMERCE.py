# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:20:58 2025

@author: naray
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#TASK 1
# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Preview data
print("Customers:\n", customers.head())
print("Products:\n", products.head())
print("Transactions:\n", transactions.head())

# Check for missing values
print("\nMissing Values:")
print("Customers:", customers.isnull().sum())
print("Products:", products.isnull().sum())
print("Transactions:", transactions.isnull().sum())

# EDA: Customers by Region
region_counts = customers["Region"].value_counts()
print("\nCustomer Distribution by Region:\n", region_counts)
sns.barplot(x=region_counts.index, y=region_counts.values, palette="viridis")
plt.title("Customer Distribution by Region")
plt.show()

# EDA: Top Product Categories
top_categories = products["Category"].value_counts().head(5)
print("\nTop 5 Product Categories:\n", top_categories)
top_categories.plot(kind="bar", color="orange")
plt.title("Top 5 Product Categories")
plt.show()

# EDA: Monthly Transaction Trends
transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
monthly_trends = transactions.groupby(transactions["TransactionDate"].dt.month).size()
print("\nMonthly Transaction Trends:\n", monthly_trends)
monthly_trends.plot(kind="bar", color="blue")
plt.title("Monthly Transactions")
plt.xlabel("Month")
plt.ylabel("Transaction Count")
plt.show()

# EDA: Top 5 Customers by Total Spending
merged = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")
top_customers = merged.groupby("CustomerID")["TotalValue"].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Customers by Total Spending:\n", top_customers)

#TASK2
from sklearn.metrics.pairwise import cosine_similarity

# Prepare customer profile dataset
customer_profile = merged.groupby("CustomerID").agg({
    "TotalValue": "sum",  # Total spending
    "TransactionID": "count"  # Transaction count
}).rename(columns={"TotalValue": "Total_Spent", "TransactionID": "Transaction_Count"})

# Normalize data
normalized_data = (customer_profile - customer_profile.min()) / (customer_profile.max() - customer_profile.min())

# Compute similarity matrix
similarity_matrix = cosine_similarity(normalized_data)

# Recommendations for C0001-C0020
for i in range(20):
    customer_id = customer_profile.index[i]
    similarity_scores = similarity_matrix[i]
    similar_customers = sorted(zip(customer_profile.index, similarity_scores), key=lambda x: x[1], reverse=True)[1:4]
    print(f"Customer {customer_id}: Top 3 Similar Customers:")
    for similar in similar_customers:
        print(f"  Customer {similar[0]} with similarity score {similar[1]:.2f}")
#TASK3
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_profile)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add clusters to the profile
customer_profile["Cluster"] = clusters
print("\nCluster Assignments:\n", customer_profile)

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(scaled_features, clusters)
print(f"\nDavies-Bouldin Index: {db_index:.2f}")

# Visualize clusters
sns.scatterplot(
    x=scaled_features[:, 0], y=scaled_features[:, 1],
    hue=clusters, palette="viridis"
)
plt.title("Customer Segmentation")
plt.xlabel("Normalized Total Spending")
plt.ylabel("Normalized Transaction Count")
plt.show()

