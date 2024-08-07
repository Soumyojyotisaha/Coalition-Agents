import numpy as np
import random
import csv
import pandas as pd
import math

#a function for calculating the resource value with the help of land
def resource(land, value):
    r=(land * value)

    return (r)

# fist, second and third determine the amount of land in ratio 2:2:1 using the uniform distribution
agent=int(input("enter the range of the agents:"))
f=math.ceil(agent*0.3656)
s=math.floor(agent*0.4231)
#t=math.floor(agent*0.2113)
t=agent-(f+s)

first = np.random.uniform(0, 0.5, f)
second = np.random.uniform(0.5, 1, s)
third = np.random.uniform(1, 2, t)
#print("First 80%: ",first)
#print("Second 10%:",second)
#print("Third 10%:", third)
print("\n")


# now the values of land are written in the following text file 'land_value.txt'.
with open('land_value.txt', 'w') as filehandle:
    for x in first:
        filehandle.write('%s\n' % x)
    for y in second:
        filehandle.write('%s\n' % y)
    for z in third:
        filehandle.write('%s\n' % z)

#the file 'land_value.txt' is being read
my_file = open("land_value.txt", "r")
content = my_file.read()

#the read values are stored in a list "content_str" to get rid of the blank line \n
#because the blank line was getting included
content_list=[]
content_str = content.split("\n")
for i in content_str:
    content_list.append(i)
print("the list of the land value in a list form:",content_list)


#random shuffle of the values of the land, [:-1] removes the last line
content_list_to_be_shuffled=content_list[:-1]
random.shuffle(content_list_to_be_shuffled)
print("after shuffle:",content_list_to_be_shuffled)

#the new shuffled values are written in another txt file
with open('land_value_shuffeled_500.txt', 'w') as filehandle:
    for x in content_list_to_be_shuffled:
        #while x != " ":
            filehandle.write('%s\n'% x)
print("test",content_list_to_be_shuffled)


again_my_file=open('land_value_shuffeled.txt', 'r')
again_content=again_my_file.read()
#res1=[]

again_content_list=[]

again_content_str=again_content.split("\n")

res1=[]
res2=[]
res3=[]
for i in content_list_to_be_shuffled:
    res1.append(resource(float(i), 7844))
    res2.append(resource(float(i), 8180))
    res3.append(resource(float(i), 10871))

file = open("land_vs_resource.csv", "w")
writer = csv.writer(file)
for w in range(len(content_list_to_be_shuffled)):
    writer.writerow([res1[w],res2[w],res3[w]])
file.close()

df = pd.read_csv("land_vs_resource.csv")
################checking the number of empty rows in th csv file##################
print (df.isnull().sum())
############Droping the empty rows###################
modifiedDF = df.dropna()
##################Saving it to the csv file############################
modifiedDF.to_csv('modifiedland_vs_resource_500.csv',index=False)