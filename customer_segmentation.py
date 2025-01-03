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