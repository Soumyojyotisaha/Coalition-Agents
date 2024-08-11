import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('test_case_20.csv')
print("The shape of data is",data.shape)
print("data",data)
data.head()


plt.scatter(data['res1'],data['res2'])
plt.show()

wcss = []
for i in range(1,11):
   km = KMeans(n_clusters=i)
   km.fit_predict(data)
   wcss.append(km.inertia_)
print (wcss)

#x = data.iloc[:,1:3] # 1t for rows and second for columns
#print(x)


plt.plot(range(1,11),wcss)


X = data.iloc[:,:].values
km = KMeans(n_clusters=4)
y_means = km.fit_predict(X)
print(y_means)
print(X[y_means == 3,1])

#plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color='blue')
plt.scatter(data['res1'],data['res2'],cmap="rainbow",c=y_means)

#plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color='red')
#plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color='green')
#plt.scatter(X[y_means == 3,0],X[y_means == 3,1],color='yellow')
plt.show()


# Display data points in each cluster
clusters = {}
for cluster_id in range(4):  # Assuming 4 clusters
    clusters[cluster_id] = np.where(y_means == cluster_id)[0]

print("Data points in each cluster:")
for cluster_id, indices in clusters.items():
    print(f"Cluster {cluster_id}: {list(indices)}")

