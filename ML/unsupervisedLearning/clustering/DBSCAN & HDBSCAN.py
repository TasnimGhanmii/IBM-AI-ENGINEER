import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler


###DBSCAN###

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")
df = df[df.ODCAF_Facility_Type == 'museum']
df = df[['Latitude', 'Longitude']]
df = df[df.Latitude!='..']
df[['Latitude','Longitude']] = df[['Latitude','Longitude']].astype('float')

#building DBSCAN

coords_scaled = df.copy()
coords_scaled["Latitude"] = 2*coords_scaled["Latitude"]

min_samples=3 # minimum number of samples needed to form a neighbourhood
eps=1.0 # neighbourhood search radius
metric='euclidean' # distance measure 

dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords_scaled)

#adding cluster labels
df['Cluster'] = dbscan.fit_predict(coords_scaled)  # Assign the cluster labels
# Display the size of each cluster
df['Cluster'].value_counts()

#One key thing to notice here is that the clusters are not uniformly dense.
#For example, the points are quite densely packed in a few regions but are relatively sparse in between.
#DBSCAN agglomerates neighboring clusters together when they are close enough.


###HDBSCAN###

min_samples=None
min_cluster_size=3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')  

# Assign labels
df['Cluster'] = hdb.fit_predict(coords_scaled)

# unlike the case for DBSCAN, clusters quite uniformly sized, although there is a quite lot of noise identified.


