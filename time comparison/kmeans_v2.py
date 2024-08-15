import numpy as np
import pandas as pd
import statsmodels.api as sm
#import seaborn as sns
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import timeit


data = pd.read_csv('test_case.csv')
print("The shape of data is",data.shape)
print("data",data)
df = pd.DataFrame(data)
data.head()

plt.scatter(data['res1'],data['res2'])
# plt.show()

wcss = []
for i in range(1,11):
   km = KMeans(n_clusters=i)
   km.fit_predict(df)
   wcss.append(km.inertia_)
print (wcss)
# plt.plot(range(1,11),wcss)


# the timer starts from here
t1 = timeit.default_timer()

X = data.iloc[:,:].values
km = KMeans(n_clusters=4, random_state=0)
y_means = km.fit_predict(X)
print(y_means)
print(X[y_means == 3,1])

# plt.show()

# Add cluster labels to the dataframe
df['Cluster'] = km.labels_

clusters = [list(df[df['Cluster'] == cluster].index) for cluster in range(km.n_clusters)]
#final_cluster = [list(item) for item in clusters]
print(clusters)

for cluster in range(km.n_clusters):
    print(f"Cluster {cluster}:")
    print(df[df['Cluster'] == cluster])

#Calculation of Coalition Structure value V(CS)
################################################################
P = 14.3
Q = 5400
res1 = 7844
res2 = 8180
res3 = 10871
# W=(7844+8180+10871)
W = 83683
agent=500

def value_calc(a_list):
    # all_agent = [i for i in range(agent)]
    ag = list(range(0, agent))
    land1 = []
    index = []
    with open('land_value_shuffeled_500.txt') as f:
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
for i in clusters:
    list_of_all_vcs.append(value_calc(i))
    final_cs_value=discount(list_of_all_vcs)
print("value of each coalitions:",final_cs_value)
print("value of the entire coalition structure:",sum(final_cs_value))

t2 = timeit.default_timer()
print("time required is:",t2 - t1)