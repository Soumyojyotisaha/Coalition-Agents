import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample
from pyclustering.cluster.kmedoids import distance_metric, type_metric
from pyclustering.utils.metric import distance_metric, type_metric
import matplotlib.pyplot as plt
import numpy as np
import timeit

# Load the data from a CSV file
df = pd.read_csv('test_case.csv')


# the timer starts from here
t1 = timeit.default_timer()
# Assuming the CSV file has three columns
data = df.iloc[:, :].values  # Extract the first three columns
#X = data.iloc[:,:].values
# Initial medoids indices (you can also choose random points)
initial_medoids = [ 3, 7, 13, 18, 22]

# Create K-Medoids algorithm o1ject
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
df['Cluster'] = labels

# Print the cluster centers (medoids)
print("Medoids:")
print(df.iloc[medoids])

# Plot the clusters (optional)
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
# plt.scatter(df.iloc[medoids, 0], df.iloc[medoids, 1], color='red', marker='x', s=100)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('K-Medoids Clustering')
# plt.show()

# Get clusters (indices of the data points)
clusters = kmedoids_instance.get_clusters()
final_clusters=[]
# Get the indices of the points in each cluster
for i, cluster in enumerate(clusters):
    print(f"Indices of points in Cluster {i}:\n", cluster)
    final_clusters.append(cluster)
    print ("The final coalition structure is: ",final_clusters)

# Optional: Get the medoid indices
medoids = kmedoids_instance.get_medoids()
print("Indices of Medoids (Cluster Centers):\n", medoids)

#Calculation of Coalition Structure value V(CS)
################################################################
P = 14.3
Q = 5400
res1 = 7844
res2 = 8180
res3 = 10871
# W=(7844+8180+10871)
W = 83683
#agent=int(input("enter no of agents: "))
agent=475
def value_calc(a_list):
    # all_agent = [i for i in range(agent)]
    ag = list(range(0, agent))
    land1 = []
    index = []
    with open('land_value_shuffeled_475.txt') as f:
        lines = f.read().splitlines()
    for i in lines:
        land1.append(i)
        for i in range(0, len(land1)):
            land1[i] = float(land1[i])
    # print("coalition values:", land1)
    for j in a_list:
        index.append(ag.index(j))
    # print ("index of the agents: ",index)
    coalition_land = list(map(lambda x: land1[x], index))
    # print("land of the corresponding lands: ",coalition_land)

    # print("combined land of a coalition: ",sum(coalition_land))
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
    # print ("the coalition values are v(C): ", coal_val_final)
    return coal_val_final

list_of_all_vcs=[]
for i in final_clusters:
    list_of_all_vcs.append(value_calc(i))
    final_cs_value=discount(list_of_all_vcs)
print("value of each coalitions:",final_cs_value)
print("value of the entire coalition structure:",sum(final_cs_value))

t2 = timeit.default_timer()
print("time required is:",t2 - t1)