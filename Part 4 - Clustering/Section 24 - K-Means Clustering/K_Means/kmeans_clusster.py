# Kmeans Clustering

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the datasets
dataset = pd.read_csv('Mall_Customers.csv')
dataset.head()
X = dataset.iloc[:,[3,4]].values

# using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Nuber of clusters')
plt.ylabel('wcss')
plt.show()
    
# Applyig k-means to the mall dataset
Kmeans = KMeans(n_clusters = 5,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
y_kmeans = Kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s = 100,c = 'red',label = 'Carefull')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s = 100,c = 'blue',label = 'Standard')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s = 100,c = 'green',label = 'Target')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s = 100,c = 'cyan',label = 'careless')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s = 100,c = 'magenta',label = 'sensible')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1], s = 300,c = 'yellow',label = 'centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()