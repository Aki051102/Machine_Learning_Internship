import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r'C:\Users\Akash\Desktop\Machine_Learning_Internship\Machine_Learning_Internship\PRODIGY_ML_02\dataset\Shopping_Mall_Customer_Segmentation_Data.csv')

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

X = data[['Age', 'Annual Income', 'Spending Score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_

print(data.head())

plt.figure(figsize=(8,6))
plt.scatter(data['Annual Income'], data['Spending Score'], c=data['Cluster'], cmap='rainbow', s=50)
plt.title('Customer Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
