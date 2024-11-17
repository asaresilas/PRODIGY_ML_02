# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuring warnings and styles
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Load and display dataset
data = pd.read_csv('Mall_Customers.csv')
print(data.head())
print(f"Dataset shape: {data.shape}")
print("Null values in each column:\n", data.isnull().sum())

# Set up continuous columns for analysis
continuous_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Visualization: Box plots, histograms, and violin plots for each continuous feature
plt.figure(figsize=(18, 15))
for i, column in enumerate(continuous_columns):
    # Box plot by Gender
    plt.subplot(3, 3, i * 3 + 1)
    sns.boxplot(x='Gender', y=column, data=data)
    plt.title(f'Box Plot of {column}')

    # Histogram
    plt.subplot(3, 3, i * 3 + 2)
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f'Histogram of {column}')
    
    # Violin plot by Gender
    plt.subplot(3, 3, i * 3 + 3)
    sns.violinplot(x='Gender', y=column, data=data)
    plt.title(f'Violin Plot of {column}')

plt.tight_layout()
plt.show()

# KDE plots: Gender-based distributions for each feature
def plot_gender_kde(column, xlabel):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data.loc[data['Gender'] == 'Male', column], label='Male', shade=True)
    sns.kdeplot(data.loc[data['Gender'] == 'Female', column], label='Female', shade=True)
    plt.title(f'Distribution of {xlabel} by Gender')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

plot_gender_kde('Annual Income (k$)', 'Annual Income')
plot_gender_kde('Age', 'Age')
plot_gender_kde('Spending Score (1-100)', 'Spending Score (1-100)')

# Data preprocessing: Drop CustomerID and encode Gender
data.drop(columns=['CustomerID'], inplace=True)
data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='Pastel1')
plt.title('Correlation Heatmap')
plt.show()

# Clustering: Using KMeans to find optimal clusters with Elbow Method
from sklearn.cluster import KMeans

# Define range for k and calculate inertias
inertias = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['Age', 'Spending Score (1-100)']])
    inertias.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid()
plt.show()

# Applying KMeans with optimal k
optimal_k = 5
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans_optimal.fit_predict(data[['Age', 'Spending Score (1-100)']])

# Visualization of clusters
plt.figure(figsize=(8, 5))
plt.scatter(data['Age'], data['Spending Score (1-100)'], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], kmeans_optimal.cluster_centers_[:, 1], 
            c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title('Clusters of Data Points')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Silhouette Score for evaluation
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data[['Age', 'Spending Score (1-100)']], y_kmeans)
print(f'Silhouette Score for {optimal_k} clusters: {silhouette_avg}')
