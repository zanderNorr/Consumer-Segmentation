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

# plt.plot()
# plt.xlim([min(x1), max(x1)])
# plt.ylim([min(x2), max(x2)])
# plt.title('Income Vs Mnt Total Purchased')
# plt.scatter(x1, x2)
# plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 50)

for k in K:
	kmeanModel = KMeans(n_clusters=k).fit(X)
	kmeanModel.fit(X)
	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])
	inertias.append(kmeanModel.inertia_)

	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
								'euclidean'), axis=1)) / X.shape[0]
	mapping2[k] = kmeanModel.inertia_


fig, axs = plt.subplots(1, 2)

axs[0].plot(K, distortions, 'bx-')
axs[0].set_xlabel('Values of K')
axs[0].set_ylabel('Distortion')
axs[0].set_title('The Elbow Method using Distortion')

axs[1].plot(K, inertias, 'rx-')
axs[1].set_xlabel('Values of K')
axs[1].set_ylabel('Inertia')
axs[1].set_title('The Elbow Method using Inertia')
plt.show()

#KMeans

# k = 10
# kmeans = KMeans(n_clusters=k, random_state=0)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s = 50, cmap='plasma')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s=200, alpha = 0.75, marker = 'X')
# plt.title(f'K-means Clustering of Income vs Total Purchases with k = {k}')
# plt.xlabel('Income (Dollars)')
# plt.ylabel('Total Purchases (Dollars)')
# plt.show()

n = 6
# gaussian mixture model (GMM)
gmm = GaussianMixture(n_components=n, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s = 20, cmap='plasma', marker = 'o', alpha = 0.25)
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s=200, alpha = 0.75, marker = 'X')
plt.xlabel('Income')
plt.ylabel('Total Purchases')
plt.title('Gaussian Mixture Model (GMM) Hierarchical Clustering')
plt.show()