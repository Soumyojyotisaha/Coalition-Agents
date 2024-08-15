import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample
from pyclustering.cluster.kmedoids import distance_metric, type_metric
import matplotlib.pyplot as plt
import numpy as np

# Load the data directly from the given values
# data = np.array([
#     [6631.735273211839, 6915.807564364208, 9190.922253325587],
#     [1487.62556505517, 1551.348434746467, 2061.700346470519],
#     [7780.68434351365, 8113.972199125657, 10783.250828446824],
#     [2084.07596952923, 2173.347964144454, 2888.320992446744],
#     [10119.002772391848, 10552.4531716172, 14023.926458270242],
#     [2726.6165430899327, 2843.411948301332, 3778.818006110487],
#     [2583.567308579545, 2694.2351586155887, 3580.566064707832],
#     [7402.4292828441185, 7719.514473950139, 10259.027120576036],
#     [7320.412912439996, 7633.984908689337, 10145.360628650586],
#     [12942.482748537974, 13496.877726037816, 17936.987501192798],
#     [13676.1398429566, 14261.961233475904, 18953.76290575997],
#     [6654.642547686674, 6939.6960785411775, 9222.669446188404],
#     [5375.430720450536, 5605.688844121033, 7449.809709589212],
#     [2536.1243627132544, 2644.759980493934, 3514.8148836124155],
#     [356.5679800856975, 371.84167224643113, 494.1675817837351]
# ])

# Initial medoids indices (you can also choose random points)
initial_medoids = [2, 5, 11]

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

# Add the cluster labels to the original dataframe for easier inspection
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
df['Cluster'] = labels

# Print the cluster centers (medoids)
print("Medoids:")
print(df.iloc[medoids])

# Plot the clusters (optional)
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(df.iloc[medoids, 0], df.iloc[medoids, 1], color='red', marker='x', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Medoids Clustering')
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
res1 = 7844
res2 = 8180
res3 = 10871
# W=(7844+8180+10871)
W = 83683
# agent=int(input("enter no of agents: "))
agent = 15

def value_calc(a_list):
    # List of agent indices
    ag = list(range(0, agent))
    land1 = []

    # Read land values from file
    with open('land_value_shuffeled_15.txt') as f:
        lines = f.read().splitlines()
    
    # Convert the land values from strings to floats
    for line in lines:
        land1.append(float(line))

    # Get the indices of the agents in the coalition
    index = [ag.index(j) for j in a_list]

    # Map indices to land values
    coalition_land = list(map(lambda x: land1[x], index))

    return sum(coalition_land)

def discount(b_list):
    coal_val_final = []
    for i in b_list:
        if i < 1:
            val = P * Q * i - W * i
            coal_val_final.append(val)
        elif i < 1.5 and i >= 1:
            val = P * Q * i - (W * 0.9) * i
            coal_val_final.append(val)
        elif i < 2 and i >= 1.5:
            val = P * Q * i - (W * 0.85) * i
            coal_val_final.append(val)
        elif i <= 3 and i:
            val = P * Q * i - (W * 0.75) * i
            coal_val_final.append(val)
        else:
            val = P * Q * i - (W * 0.50) * i
            coal_val_final.append(val)
    return coal_val_final

list_of_all_vcs = []
for i in final_clusters:
    list_of_all_vcs.append(value_calc(i))
    final_cs_value = discount(list_of_all_vcs)
print("Value of each coalition:", final_cs_value)
print("Value of the entire coalition structure:", sum(final_cs_value))
