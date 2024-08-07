DP.pyfrom itertools import combinations
import random, pandas as pd, timeit


#first line of the input file
ip = open("value_for_DP_file_16.txt", "r")
output = open("output_16.txt", "w")

n = int(ip.readline())
#print("jj",n)
sums = 0
temp = []
dictionary = {}
dp = {}
agents = []
coalition = []
backtrack = {}
dp[()] = 0
backtrack[()] = []

dictionary[(0,)] = ip.readline()
dictionary[(0,)] = dictionary[(0,)].strip()
print(dictionary[(0,)])
#dictionary[(0,)] = float(dictionary[(0,)])

for i in range(1, n + 1):
    agents.append(i)
#print(agents)

for i in range(n):
    all_combinations = combinations(agents, i + 1)
    for element in list(all_combinations):
        coalition.append(element)
#print("coalition", coalition)
#print("len of coalition",len(coalition))

def make_dic():
    for i in coalition:
        dictionary[i] = ip.readline()
        dictionary[i] = dictionary[i].strip()
       # dictionary[i] = float(dictionary[i])
    #print(dictionary)

make_dic()


def complement(t, superset):
    c = []
    for ele in superset:
        #print(ele)
        if ele in t:
            continue
        else:
            c.append(ele)
    tup = tuple(c)
    #print("tup",tup)
    return tup



t1 = timeit.default_timer()
for i in range(len(coalition)):
    if len(coalition[i]) == 1:
        #print(coalition[i])
        dp[coalition[i]] = dictionary[coalition[i]]
        backtrack[coalition[i]] = coalition[i]
        continue
    else:
        varMax = -1
        #print(type(varMax))
        tempCoal = 0
        tempCoalCompliment = 0
        for j in range(int((len(coalition[i]) / 2))):
            coal1 = list(combinations(coalition[i], j + 1))
            for k in range(len(coal1) - 1):

                coalComplement = complement(coal1[k], coalition[i])
                tempVar = int(dp[coal1[k]]) + int(dp[coalComplement])
                #print(type(tempVar))

                    #tempVar=int(i) int(float('55063.000000'))
                if float(tempVar) > varMax:
                    tempCoal = backtrack[coal1[k]]
                    tempCoalCompliment = backtrack[coalComplement]
                    varMax = max(varMax, tempVar)
        if varMax > int(dictionary[coalition[i]]):

            dp[coalition[i]] = varMax
            backtrack[coalition[i]] = [tempCoal, tempCoalCompliment]
        else:
            dp[coalition[i]] = dictionary[coalition[i]]
            backtrack[coalition[i]] = coalition[i]

t2 = timeit.default_timer()
print(t2 - t1)

output.write("\n" + str(backtrack[coalition[len(coalition) - 1]]))
output.write(" : " + str(dp[coalition[len(coalition) - 1]]) + "\n")
ip.close()
output.close()