# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:17:08 2023

@author: Dell
"""

'''
Problem Statement: -

A pharmaceuticals manufacturing company is conducting a 
study on a new medicine to treat heart diseases. The company
has gathered data from its secondary sources and would like
you to provide high level analytical insights on the data.
Its aim is to segregate patients depending on their age 
group and other factors given in the data. Perform PCA and 
clustering algorithms on the dataset and check if the 
clusters formed before and after PCA are the same and
provide a brief report on your model. You can also explore 
more ways to improve your model. 

Business Objectives:
    Determine the effectiveness of the new medicine in 
    treating heart diseases across different patient 
    groups.and also Segregate patients into different 
    age groups based on available data.
    
 Constraints:
     Data Quality and Availability,Limited financial, human,
     and time resources.
          
 '''

import pandas as pd
df= pd.read_csv("C:/2-dataset/heart disease.csv")
df
df.dtypes
'''
column_name   types       Discription
age           int64     The age of the patient
sex           int64     gender of the patient
cp            int64       Chest pain type 
trestbps      int64     Resting blood pressure
chol          int64     Serum cholesterol level
fbs           int64     Fasting blood sugar > 120 mg/dl (binary: 0 for false, 1 for true).
restecg       int64     Resting electrocardiographic results 
thalach       int64     Maximum heart rate achieved 
exang         int64      Exercise-induced angina (binary: 0 for no, 1 for yes).
oldpeak     float64     ST depression induced by exercise relative to rest 
slope         int64     The slope of the peak exercise ST segment
ca            int64     Number of major vessels colored by fluoroscopy 
thal          int64     Thalassemia type 
target        int64     Presence of heart disease
'''
###########################################################
############EDA###########################################
df.shape
#(303, 14)
df.columns
'''
Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
      dtype='object')
'''
df.size
#4242
df.info

#Calculate the null values
#0

#calculate the dublicate values
duplicate=df.duplicated()

duplicate
sum(duplicate)
#------------------------------------------------------------
#############Clusterning and PCA##############################
# Apply Normalization function 

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# whenever there is mixed data apply normalization
# Now apply this normalization function to df for all the rows

df_norm=norm_fun(df.iloc[:,:])

# all data from is up to 1
b=df_norm.describe()
print(b)

# Defined the number of clusters 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans

a=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    a.append(kmeans.inertia_)

# total within sum of square

print(a)
# As k value increases the a the a value decreases
plt.plot(k,a,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_ss")

# To select value of k from elbow curve -
# k changes from 2 to 3 , then decrease
# in a is higher than 
# when k changes from 3 to 4.
# When k value changes from 5 to 6 decreases
# in a is higher than when k chages 3 to 4 .
# When k values changes from 5 to 6 decrease
# in a is considerably less , hence considered k=3

# KMeans model
model=KMeans(n_clusters=3)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

df['clust']=mb
df.head()
df=df.iloc[:,[7,0,1,2,3,4,5,6]]
df
df.iloc[:,2:8].groupby(df.clust).mean()

# Convert DataFrame into csv file
df.to_csv("heart disease.csv",encoding="utf-8")
import os
os.getcwd()

#----------------------------------------------------------
#PCA:-
'''
Principal Component Analysis (PCA) is a dimensionality
reduction technique commonly used in machine learning and 
data analysis. It helps to transform high-dimensional data 
into a lower-dimensional 
'''
#Step 1: Import Libraries
import numpy as np
from sklearn.decomposition import PCA


# Principal Component Analysis (PCA)
# Standardize the data before applying PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)


# Apply PCA and extract the first 3 principal components
pca = PCA(n_components=3)
principal_components = pca.fit_transform(df_standardized)


# Create a new dataframe with the 3 principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])


# Display the new dataframe with principal components
print("\nDataset with Principal Components:")
print(df_pca.head())


# Visualize the explained variance ratio
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

#----------------------------------------------------------------------------


#Perform Hierarchical Clustering on the Principal Components
# Dataset

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage  

# Create a linkage matrix using Ward's method
linkage_matrix_pca = linkage(df_pca, method='ward')


# Plot the dendrogram for the principal components dataset
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix_pca)
plt.title('Hierarchical Clustering Dendrogram (Principal Components)')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()


# Perform hierarchical clustering on the principal components
# dataset

num_clusters_pca = 3  # You can choose a different number of clusters
hierarchical_cluster_pca = AgglomerativeClustering(n_clusters=num_clusters_pca, linkage='ward')
df_pca['Hierarchical_Cluster'] = hierarchical_cluster_pca.fit_predict(df_pca)


# K-means Clustering on Principal Components
# Choose the number of clusters for K-means
# on principal components
num_kmeans_clusters_pca = 3  # You can choose a different number of clusters
kmeans_cluster_pca = KMeans(n_clusters=num_kmeans_clusters_pca, random_state=42)
df_pca['KMeans_Cluster'] = kmeans_cluster_pca.fit_predict(df_pca)


# Compare clustering results between the original dataset 
#and principal components dataset
print("\nHierarchical Clustering Results:")
print("Original Dataset:\n", df['Hierarchical_Cluster'].value_counts())
print("Principal Components Dataset:\n", df_pca['Hierarchical_Cluster'].value_counts())

print("\nK-means Clustering Results:")
print("Original Dataset:\n", df['KMeans_Cluster'].value_counts())
print("Principal Components Dataset:\n", df_pca['KMeans_Cluster'].value_counts())

#################################################################






