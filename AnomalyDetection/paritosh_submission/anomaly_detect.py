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



#####################
#
# The Below plott takes 60-70 minutes to work. Total computauions were 675 * 675 * 256
######################

# Plot the proximity matrix
w, h = 676, 676;
C = [[0 for x in range(w)] for y in range(h)]

for x in range(676):
    for j in range(676):
        C[x][j] = eDistance(df.ix[x],df.ix[j])
plt.matshow(C,cmap="Reds")



def plot_anomaly_score(arr):
    x = np.arange(0,len(arr),1)
    sorted_arra = np.sort(arr)
    plt.xlabel('Data Instances Sorted in Increasing order of F(x)')
    plt.ylabel('F(x) or Anomaly_Score')
    plt.plot(x,sorted_arra)
    plt.show()





# Proximity Based approaches

#K nearest , where f(x) or Anomaly score = Distance between x and its K-nearest neighbor
k_nearest_distance_array = [0 for x in range(676)]
k = 15
k_nearest_distance_array[0] = 0
for x in range(676):
    row = C[x]
    row.sort()
    k_nearest_distance_array[x] = row[7]

plot_anomaly_score(k_nearest_distance_array)

#f(x) or Anomaly score = Average(Distance between x and its K-nearest neighbor)
k_nearest_distance_array[0] = 0
for x in range(676):
    row = C[x]
    row.sort()
    k_nearest_distance_array[x] = (row[1] + row [2] + row[3] + row[4] + row[5] + row[6] + row[7])/7
score_array = k_nearest_distance_array
plot_anomaly_score(k_nearest_distance_array)



#### Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor
X = df
X = X.drop(df.columns[[0,0]], axis=1)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(X)

X_scores = clf.negative_outlier_factor_
#print(X_scores)
plot_anomaly_score(X_scores)


## remove cuisines
arr = [1,2,3,4,6,7,8,9,10,13, 14, 15, 16, 19, 20, 21,31, 32, 33, 34, 40, 68,69,70,89,90,91,92,93,94,95,96,97,99,118, 119, 121,122,123,124,125,126,127,128,129,130,131,132,125,136]
X = X.drop(df.columns[arr])
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(X)

X_scores = clf.negative_outlier_factor_
#print(X_scores)
plot_anomaly_score(X_scores)



from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
plot_anomaly_score(X_scores)
