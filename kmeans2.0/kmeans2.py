import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('test_case.csv')
print("The shape of data is",data.shape)
print("data",data)
df = pd.DataFrame(data)
data.head()

plt.scatter(data['res1'],data['res2'])
plt.show()

wcss = []
for i in range(1,11):
   km = KMeans(n_clusters=i)
   km.fit_predict(df)
   wcss.append(km.inertia_)
print (wcss)

#x = data.iloc[:,1:3] # 1t for rows and second for columns
#print(x)
plt.plot(range(1,11),wcss)

X = data.iloc[:,:].values
km = KMeans(n_clusters=10, random_state=0) #modified the number of clusters
y_means = km.fit_predict(X)
print(y_means)
print(X[y_means == 3,1])

plt.show()

# Add cluster labels to the dataframe
df['Cluster'] = km.labels_

clusters = [list(df[df['Cluster'] == cluster].index) for cluster in range(km.n_clusters)]
#final_cluster = [list(item) for item in clusters]
print(clusters)

for cluster in range(km.n_clusters):
    print(f"Cluster {cluster}:")
    print(df[df['Cluster'] == cluster])

#list_of_lists = [list(item) for item in frozen_set]
################################################################
P = 14.3
Q = 5400
res1 = 7844
res2 = 8180
res3 = 10871
# W=(7844+8180+10871)
W = 83683
agent = int(input("Enter the number of agents: "))
filenum = int(input("Enter the file number: "))

def value_calc(a_list):
    """Calculate the combined land value for a given coalition."""
    ag = list(range(agent))
    land1 = []
    index = []
    
    # Construct the filename using the file number
    filename = f'land_value_shuffeled_{filenum}.txt'
    
    # Read the land values from the file
    with open(filename) as f:
        lines = f.read().splitlines()
    
    # Convert land values to float
    land1 = [float(line) for line in lines]
    
    # Get the indices of the agents in the coalition
    index = [ag.index(j) for j in a_list]
    
    # Calculate the land values for the coalition
    coalition_land = [land1[x] for x in index]
    
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
for i in clusters:
    list_of_all_vcs.append(value_calc(i))
    final_cs_value=discount(list_of_all_vcs)
print("value of each coalitions:",final_cs_value)
print("value of the entire coalition structure:",sum(final_cs_value))