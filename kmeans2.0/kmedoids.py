import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
import matplotlib.pyplot as plt
import numpy as np

# Define the new data
data = np.array([
    [3409.68625885183, 3555.7411521427807, 4725.484360017625],
    [14875.338686201683, 15512.52810468253, 20615.732643765743],
    [4728.819136940927, 4931.379467131155, 6553.670683029681],
    [2672.050186684577, 2786.508226297787, 3703.194489985727],
    [13212.715221790171, 13778.685685140694, 18311.50269965336],
    [5938.341356668357, 6192.711919626104, 8229.947589028774],
    [6419.893366233976, 6694.891348265416, 8897.330543642218],
    [9858.764122400617, 10281.067124074076, 13663.261699976683],
    [1679.7157339172354, 1751.6668413364337, 2327.918121291977],
    [3283.085203100151, 3423.717103691896, 4550.027950395429],
    [1623.9413476304023, 1693.5033431433824, 2250.620396492874],
    [5508.390239058449, 5744.343722016587, 7634.078313208108],
    [7388.72769215224, 7705.225971673294, 10240.038085337455],
    [5581.016624188403, 5820.081079278574, 7734.731224063249],
    [570.6432176908828, 595.0868843334296, 790.8544644973977],
    [6537.738068961495, 6817.783962787485, 9060.651523161703],
    [10406.533217585798, 10852.30006627382, 14422.41491692698],
    [4069.199776691528, 4243.50512153706, 5639.504178023152],
    [2531.467746779583, 2639.903897075088, 3508.3612793524794]
])

# Initial medoids indices (can be chosen randomly or based on specific criteria)
initial_medoids = [0, 4, 8]  # Adjust as needed

# Create K-Medoids algorithm object
metric = distance_metric(type_metric.EUCLIDEAN)
kmedoids_instance = kmedoids(data, initial_medoids, metric=metric)

# Run cluster analysis
kmedoids_instance.process()

# Get clusters and medoids
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Assign cluster labels to data points
labels = np.zeros(len(data))
for i, cluster in enumerate(clusters):
    for index in cluster:
        labels[index] = i

# Create a DataFrame to simulate the original one and add cluster labels
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
df['Cluster'] = labels

# Print the cluster centers (medoids)
print("Medoids:")
print(df.iloc[medoids])

# Plot the clusters (optional)
plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', label='Data Points')
plt.scatter(df.iloc[medoids, 0], df.iloc[medoids, 1], color='red', marker='x', s=100, label='Medoids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Medoids Clustering')
plt.legend()
plt.show()

# Get clusters (indices of the data points)
clusters = kmedoids_instance.get_clusters()
final_clusters = []
# Get the indices of the points in each cluster
for i, cluster in enumerate(clusters):
    print(f"Indices of points in Cluster {i}:\n", cluster)
    final_clusters.append(cluster)
    print("The final coalition structure is: ", final_clusters)

# Optional: Get the medoid indices
medoids = kmedoids_instance.get_medoids()
print("Indices of Medoids (Cluster Centers):\n", medoids)

# Calculation of Coalition Structure value V(CS)
################################################################
P = 14.3
Q = 5400
W = 83683
agent = 19  # Updated to match the number of data points

def value_calc(a_list):
    land1 = [
        3409.68625885183, 14875.338686201683, 4728.819136940927,
        2672.050186684577, 13212.715221790171, 5938.341356668357,
        6419.893366233976, 9858.764122400617, 1679.7157339172354,
        3283.085203100151, 1623.9413476304023, 5508.390239058449,
        7388.72769215224, 5581.016624188403, 570.6432176908828,
        6537.738068961495, 10406.533217585798, 4069.199776691528,
        2531.467746779583
    ]
    index = [i for i in a_list]
    coalition_land = [land1[x] for x in index]
    return sum(coalition_land)

def discount(b_list):
    coal_val_final = []
    for i in b_list:
        if i < 1:
            val = P * Q * i - W * i
        elif i < 1.5:
            val = P * Q * i - (W * 0.9) * i
        elif i < 2:
            val = P * Q * i - (W * 0.85) * i
        elif i <= 3:
            val = P * Q * i - (W * 0.75) * i
        else:
            val = P * Q * i - (W * 0.50) * i
        coal_val_final.append(val)
    return coal_val_final

list_of_all_vcs = [value_calc(i) for i in final_clusters]
final_cs_value = discount(list_of_all_vcs)
print("Value of each coalition:", final_cs_value)
print("Value of the entire coalition structure:", sum(final_cs_value))
