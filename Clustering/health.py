import numpy as np
import pandas as pd
from sklearn import cluster
from scipy.cluster import hierarchy
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from skimage.measure import regionprops
from scipy.ndimage.measurements import label
from sklearn.metrics import calinski_harabaz_score as chindex

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

dataLoc="Absenteeism_at_work.csv"
data=pd.read_csv(dataLoc)

health=[8,14,15,17,18,19]
healthData=data.values[:,health]

healthPerID=np.zeros((36,6))
l=0
for k in range(740):
    if not (np.sum(np.power((healthPerID-healthData[k]),2),axis=1)==0).any():
        healthPerID[l]=healthData[k]
        l+=1
        


healthData=(healthPerID-np.mean(healthPerID,axis=0))/np.std(healthPerID,axis=0)



# k-means
losses=[]
chis=[]
for n_c in np.arange(2,36):
    clustering=cluster.KMeans(n_clusters=n_c).fit(healthData)
    losses.append(clustering.score(healthData))
    chis.append(chindex(healthData,clustering.labels_))
losses=-np.array(losses)
plt.plot(np.arange(2,36),losses)
plt.hold(True)
# plt.scatter(10,losses[8],c='r',marker='x')
plt.figure()
plt.title('CH Index over k')
plt.plot(np.arange(2,36),chis)
# There is no intrinsic clustering in the data based on the CH index and the elbow index

# Agglomerative Clustering
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

linked = linkage(healthData, 'average')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='top',
#            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()  

# ward 4
# single 11
# complete 4
# average 5

model = cluster.AgglomerativeClustering(n_clusters=3,linkage='complete')
model = model.fit(healthData)
plt.figure()
plt.title('Hierarchical Clustering Dendrogram (Complete)')
plot_dendrogram(model, labels=model.labels_)
plt.show()

model = cluster.AgglomerativeClustering(n_clusters=3,linkage='ward')
model = model.fit(healthData)
plt.figure()
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plot_dendrogram(model, labels=model.labels_)
plt.show()

model = cluster.AgglomerativeClustering(n_clusters=3,linkage='single')
model = model.fit(healthData)
plt.figure()
plt.title('Hierarchical Clustering Dendrogram (Complete)')
plot_dendrogram(model, labels=model.labels_)
plt.show()


#Z=hierarchy.linkage(healthData,'single')
#plt.figure()
#dn = hierarchy.dendrogram(Z)
#hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
#fig, axes = plt.subplots(1, 2, figsize=(8, 3))
#dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
#                           orientation='top')
#dn2 = hierarchy.dendrogram(Z, ax=axes[1],
#                           above_threshold_color='#bcbddc',
#                           orientation='right')
#hierarchy.set_link_color_palette(None)  # reset to default after use
#plt.show()
#
#
#
#clustering=cluster.AgglomerativeClustering(n_clusters=2,linkage='complete').fit(healthData)
#plt.title('Hierarchical Clustering Dendrogram')
#plot_dendrogram(clustering, labels=clustering.labels_)
#plt.show()

# DBSCAN
allEps=np.linspace(0.01,7.5,100)
allMinSamps=np.arange(1,18)
outliers=np.zeros((np.size(allEps),np.size(allMinSamps)))
n_clusters=np.zeros((np.size(allEps),np.size(allMinSamps)))
for k in range(np.size(allEps)):
    eps=allEps[k]
    for l in range(np.size(allMinSamps)):
        min_samples=allMinSamps[l]
        clustering=cluster.DBSCAN(eps=eps,min_samples=min_samples).fit(healthData)
        outliers[k,l]=np.sum(clustering.labels_==-1)
        n_clusters[k,l]=np.amax(clustering.labels_)+1

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(allEps, allMinSamps)
surf=ax.plot_surface(X,Y,np.transpose(n_clusters),cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(allEps, allMinSamps)
surf=ax.plot_surface(X,Y,np.transpose(outliers),cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

bestArea=0
bestK=-1
for n_cluster in np.unique(n_clusters):
    temp=n_clusters==n_cluster
    if np.sum(temp)>bestArea:
        bestArea=np.sum(temp)
        bestK=n_cluster
# There is no natural clusters in the health data according to DBSCAN



