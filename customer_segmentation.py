# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Step 2: Load and Explore the Dataset
# Replace the file path with the actual location of your dataset
dataset_path = "customer_data.csv"  # Make sure this file is in your project folder

try:
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Dataset not found at {dataset_path}. Please check the file path.")
    exit()

# Display the first few rows of the dataset
print("\nDataset Head:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# Step 3: Data Preprocessing
# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Select relevant features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data_features = data[features]

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_features)

# Step 4: Determine the Optimal Number of Clusters using Elbow Method
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Create an output folder for saving images
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# Save the Elbow Method plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.savefig(os.path.join(output_folder, "elbow_method.png"))
plt.show()

# Step 5: Apply K-Means Clustering
# Choose the optimal number of clusters (e.g., 3 or 4 based on the elbow plot)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Step 6: Visualize the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='viridis',
    s=100,
)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid()

# Save the cluster plot
plt.savefig(os.path.join(output_folder, "cluster_plot.png"))
plt.show()

# Step 7: Interpret the Results
# Display the data with cluster assignments
print("\nClustered Data Sample:\n", data.head())

# Save the clustered data as a CSV file
output_csv_path = os.path.join(output_folder, "clustered_data.csv")
data.to_csv(output_csv_path, index=False)
print(f"\nClustered data saved to {output_csv_path}")
