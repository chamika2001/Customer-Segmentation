# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and Explore the Dataset
# Use a CSV file named "customer_data.csv" for this example.
# Replace the file path with your dataset location.
try:
    data = pd.read_csv("customer_data.csv")  # Add your dataset file here
except FileNotFoundError:
    print("Dataset not found! Please check the file path.")

# Display basic info about the dataset
print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# Data Preprocessing
# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Selecting relevant features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Determine the Optimal Number of Clusters using Elbow Method
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()