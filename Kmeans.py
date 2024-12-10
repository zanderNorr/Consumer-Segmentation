from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('preprocessed_marketing_campaign.csv')

x1 = df['Income']
x2 = df['MntTotalPurchased']

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 20)

for k in K:
	kmeanModel = KMeans(n_clusters=k).fit(X)
	kmeanModel.fit(X)
	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])
	inertias.append(kmeanModel.inertia_)

	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
								'euclidean'), axis=1)) / X.shape[0]
	mapping2[k] = kmeanModel.inertia_


plt.plot(K, inertias, 'rx-', label = 'Inertia Value')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.legend()
plt.show()

# KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s = 50, cmap='plasma')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s=200, alpha = 0.75, marker = 'X', label = 'Centroid')
plt.title(f'K-means Clustering of Income vs Total Purchases with k = {k}')
plt.xlabel('Income (Dollars)')
plt.ylabel('Total Purchases (Dollars)')
plt.legend(loc = 'best')
plt.show()