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

n_components_range = range(1, 21)
BIC = []
AIC = []
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    BIC.append(gmm.bic(X))
    AIC.append(gmm.aic(X))

print(BIC, AIC)
# Plot log-likelihood vs number of components
plt.plot(n_components_range, BIC, label= 'BIC', color ='blue')
plt.plot(n_components_range, AIC, label= 'AIC', color = 'red')
plt.xlabel('Number of Components')
plt.ylabel('BIC and AIC Values')
plt.title('BIC and AIC Scores')
plt.legend(loc='best')
plt.show()
n = 5
# gaussian mixture model (GMM)
gmm = GaussianMixture(n_components=n, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s = 20, cmap='plasma', marker = 'o', alpha = 0.25)
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s=200, alpha = 0.75, marker = 'X', label = 'Centroid')
plt.xlabel('Income')
plt.ylabel('Total Purchases ($)')
plt.title('Gaussian Mixture Model (GMM)')
plt.legend(loc= 'best')
plt.show()