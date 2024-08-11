import numpy as np
import pandas as pd
import array
import heapq
import math
import itertools
from itertools import chain
from scipy.spatial import distance
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import cityblock
from sklearn.preprocessing import StandardScaler
# import networkx as nx
import matplotlib.pyplot as plt
import timeit


######################## Data Set Generation Part ############################
# data=np.array(np.random.randint(500,size=(5,4)))
# data=np.array(pd.read_csv("modifiedland_vs_resource.csv",index_col=None))

data = np.loadtxt("modifiedland_vs_resource_500.csv", dtype=float, delimiter=',')
scale = StandardScaler()
scaled_data = scale.fit_transform(data)

data1 = pd.DataFrame(data, columns=None)
# print("Our Dataset is=\n",data1)
############################################################################
plt.scatter(data[:, 0], data[:, 1], s=200)
plt.title("Graphical View of Original Data Set")
plt.show()

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], s=200)
plt.title("Graphical View of Scaled Data Set")
plt.show()



#################### Assigning Data Set into Model ############################
input_data = scaled_data
agent = input_data.shape[0]
features = input_data.shape[1]
all_agent = [i for i in range(agent)]
print("all_agent", all_agent)
##### Similarities Matching using Cosine Matching(Distance Based) Approach ####
sim_idx = []
sim_elements = []
# sim=distance.cdist(input_data,input_data,metric='euclidean')
sim = manhattan_distances(input_data, input_data)



#########################Dynamic Run Coalition of Various Sized################
min_size = int(input("Enter the Size of Smallest Coalition:"))
max_size = int(input("Enter the Size of Largest Coalition:"))
t1 = timeit.default_timer()
for k in range(min_size, max_size + 1):
#x=int(math.sqrt(agent))
#print("what is x?",x)
#for k in range(x,int(agent/2)):
    for i in range(len(sim)):
        temp1 = (np.argsort(sim[i])[:k]).tolist()
        # temp1=(heapq.nsmallest(k, range(len(sim[i])), sim[i].take))
        sim_idx.append(temp1)
        # sim_idx += [temp1]
print("All coalitions of given size range are=\n", sim_idx)
for n in range(len(sim_idx)):
    # print(sim_idx[n])
    break
final_col = list(set(tuple(sorted(sub)) for sub in sim_idx))
print("\n\nFinal coalitions after removing duplicates are=\n", final_col)

###################### Value Maximized coalition structutre ###################
###############################################################################
P = 14.3
Q = 5400
res1 = 7844
res2 = 8180
res3 = 10871
# W=(7844+8180+10871)
W = 83683



#########this function calculates the total value of land of any coalition ################
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


############calculate the final COALITION VALUE ################################
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


################ We take each coalition from "final_col" then calculates its land by using "value_calc()" and then
# calculate the final coalition value using discount() function###################


coalition_value_intermediate = []
for i in final_col:
    coalition_value_intermediate.append(value_calc(i))
    # map(float,coalition_value_intermediate)
# random_data = np.random.randint(1000, size=(len(final_col)))
all_col_val = np.array(discount(coalition_value_intermediate))
#print("random_data", discount(coalition_value_intermediate))
# var1 = int(input("Enter the number of Coalitions to be used for max sum:"))


#### Removal of Negative Valued Coalitions ###########################
# negative_idx = []
# for t in range(len(all_col_val)):
#     if all_col_val[t] < 0:
#         negative_idx.append(t)
# final_col = [i for j, i in enumerate(final_col) if j not in negative_idx]

##################### Set Cover using final_col ########################
temp_col = []
for g in range(len(final_col)):
    temp_col.append(set(final_col[g]))

all_agent = set(all_agent)


def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset

    return cover


cover_col = set_cover(all_agent, temp_col)
print("coalitions after cover",cover_col)
################### Common agent detection from final coalitions ################
###############################################################################
temp2 = []
for k in range(len(cover_col)):
    temp2.append(list(cover_col[k]))
cover_col = temp2
all_col = np.concatenate(cover_col)
new_all_col = np.array(all_col)
intersect_agent = []
for x in range(len(all_col)):
    count = np.count_nonzero(all_col == all_col[x])
    # print(count)
    if count > 1:
        intersect_agent.append(all_col[x])
        new_all_col[x] = -1
intersect_agent = list(set(intersect_agent))

print("\n\n Intersect agents are=", intersect_agent)

# ########### Removal of intersect agent from particular coalition ##############
cover_col = np.array(temp2,dtype=object)
temp_list = []
for kk in range(len(intersect_agent)):
    temp_idx = []
    for jj in range(len(cover_col)):
        if (intersect_agent[kk] in cover_col[jj]) == True:
            temp_idx.append(jj)
        else:
            nothing = 0
    temp_val = list(cover_col[temp_idx])
    temp_list.append(temp_idx[temp_val.index(max(temp_val))])

########### Assignment of intersect agent into particular max valued coalition ##
#print("Each Intersect agents will be assigned into these coalitions=", temp_list)
list1 = []
for g in range(len(cover_col)):
    list1.append([ele for ele in cover_col[g] if ele not in intersect_agent])
for gm in range(len(temp_list)):
    list1[temp_list[gm]].append(intersect_agent[gm])
print("\n\nCoalitions After assigning the intersect agents=\n", list1)

########### Finding the value of each coalitions in list1 #############
abc=[]
for i in list1:
    abc.append(value_calc(i))
    final_cs_value=discount(abc)
print("value of each coalitions:",final_cs_value)

############# Function to get all coalitions values and Col Struct Value #####
def cs_val(any_list,n):
    temp5=[]
    for t in range(len(any_list)):
        temp5.append(discount ([value_calc(any_list[t])]))
    vv=list(chain(*temp5))
    if n==0:
        return list(vv)
    elif n==1:
        return sum(vv)

############### Removal of -ve Valued coalitions ############################
temp0=[]
temp1=[]
counter=0
for i in range(len(final_cs_value)):
    if final_cs_value[i] < 0:
        final_cs_value[i]=0
        temp0.append(i)
        temp1.append(list1[i])
        counter+=1
    else:
        nothing=0

if counter>0:
    final_cs_value.remove(0)
    list1=[list1[i] for i in range(len(list1)) if i not in temp0]
else:
    nothing=0


############ Final Assignment of -ve Valued Coalitions ######################
for r in range(len(temp1)):
    (list1[np.argmin(final_cs_value)]).extend(temp1[r])
    final_cs_value=cs_val(list1,0)
print("Final CS=",list1,"\nFinal CS Value=",cs_val(list1,1))

individual_coal_land_val=[]
for i in list1:
    individual_coal_land_val.append(value_calc(i))
    individual_coal_val=discount(individual_coal_land_val)
print("individual_coal_val",individual_coal_val)
################## for DP=grand coalition###################
t2 = timeit.default_timer()
print("time required is:",t2 - t1)

abc=[]

#grand = [[i for i in range(15)]]
grand = [[i for i in range(agent)]]
for i in grand:
    abc.append(value_calc(i))
    final_cs_value=discount(abc)
dp=sum(final_cs_value)
#print("grand value of each coalitions:",final_cs_value)
print("grand value of the entire CS:",dp)

optimality=(100*cs_val(list1,1))/dp
print("how close to optimality:",optimality)