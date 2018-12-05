from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pow

#euclian distance between two datasets
def eDistance(a, b):
    distance = 0
    for x in range(256):
        distance = distance + pow((a[x + 2] - b[x + 2]),2)
    return distance

#read data
df = pd.read_csv('/Users/paritoshgoel/Desktop/ruiz/Knowledge_Discovery_Data_Mining/AnomalyDetection/dataset/data/chicago.txt',delimiter="\t",sep="\n", header=None)

#extract out the features into a new dataframe
features = df.iloc[:,2]

#drop first and third column
df = df.drop(df.columns[[0,2]], axis=1)

#insert 257 new rows for festures
for x in range(256):
  df[x + 2] = 0

#fill the features in the dataframe
for x in range(675):
    list = features.iloc[[x]].str.split()
    for f in list:
        for fe in f:
            df.loc[x, int(fe)+2] = 1

# Plot the proximity matrix
w, h = 676, 676;
C = [[0 for x in range(w)] for y in range(h)]

for x in range(676):
    for j in range(676):
        C[x][j] = eDistance(df.ix[x],df.ix[j])
plt.matshow(C,cmap="Reds")


## To be done next - clustering and k nearest neighbor, plot the k-nearest distance with datasets to plot the elbow
