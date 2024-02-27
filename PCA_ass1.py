# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:23:00 2023

@author: Dell
"""
'''
Problem Statement: -
Perform hierarchical and K-means clustering on the dataset. 
After that, perform PCA on the dataset and extract the first
3 principal components and make a new dataset with these 3 
principal components as the columns. Now, on this new dataset,
perform hierarchical and K-means clustering. Compare the results
of clustering on the original dataset and clustering on the 
principal components dataset (use the scree plot technique to 
obtain the optimum number of clusters in K-means clustering and 
check if you’re getting similar results with and without PCA).
'''
import pandas as pd
df=pd.read_csv("C:/2-dataset/wine.csv")
df
df.dtypes
#Data Dictionary
'''
column_name         Data_types   Discription
Type                 int64
Alcohol            float64      This column seems to contain 
                                numerical values of type 
                                float64, which could represent
                                the alcohol content of some 
                                substances.
                                
Malic              float64      This column, also of type 
                               float64, likely contains 
                               numerical values representing 
                               some attribute named "Malic."
                               
Ash                float64      This column, of type float64,
                               probably contains numerical 
                               values related to the ash 
                               content of the substances.
                               
Alcalinity         float64     This column, of type float64,
                               likely contains numerical values
                               representing the alcalinity of
                               the substances.
                               
Magnesium            int64     This column, of type int64,
                               contains integer values,
                               possibly representing the 
                               magnesium content of the 
                               substances.
                               
Phenols            float64     This column, of type float64,
                               likely contains numerical values
                               representing some attribute 
                               named "Phenols."
                               
Flavanoids         float64     This column, of type float64, 
                               likely contains numerical values
                               representing flavonoid content.
                               
Nonflavanoids      float64      This column, of type float64,
                                likely contains numerical 
                                values representing the 
                                non-flavanoid content. 
                                
Proanthocyanins    float64      This column, of type float64,
                                probably contains numerical 
                                values representing some 
                                attribute named 
                                "Proanthocyanins."
                                
Color              float64      This column, of type float64, 
                                likely contains numerical 
                                values representing the color 
                                attribute
                                
Hue                float64      This column, of type float64,
                                probably contains numerical 
                                values representing the hue 
                                attribute.
                                
Dilution           float64      This column, of type float64, 
                               likely contains numerical values
                               representing the dilution of 
                               some substances.
                               
Proline              int64     This column, of type int64, 
                               contains integer values, 
                               possibly representing the 
                               proline content of the 
                               substances.
'''
#EDA
df.shape
#(178, 14)
df.info
df.columns
'''
Index(['Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline'],
      dtype='object')
'''
df.size
# 66
# Data cleaning:-
#1)Data Cleaning:
'''
Handling Missing Values: Decide how to handle missing data,
either by removing rows with missing values, imputing 
missing values, or using more advanced techniques.
Outlier Detection and Treatment: Identify and deal with 
outliers that can skew the analysis or modeling results.
'''

# Check for missing values in the entire dataset
print(df.isnull().sum())
# there is no missing value in data set

# Check for missing values in a specific column
print(df['Type'].isnull().sum())
print(df['Alcohol'].isnull().sum())
print(df['Malic'].isnull().sum())
print(df['Ash'].isnull().sum())
print(df['Alcalinity'].isnull().sum())
print(df['Magnesium'].isnull().sum())
print(df['Phenols'].isnull().sum())
print(df['Flavanoids'].isnull().sum())
print(df['Nonflavanoids'].isnull().sum())
print(df['Proanthocyanins'].isnull().sum())
print(df['Proline'].isnull().sum())

# there is not any missing values in any columns.

#----------------------------------------------------
# how to find the outlier in dataset
#Box Plots: 
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['Type'])
plt.show()

sns.boxplot(x=df['Alcohol'])
plt.show()

sns.boxplot(x=df['Malic'])
plt.show()

#Scatter Plots: 
    
plt.scatter(x=df['X_Type'], y=df['Y_Alcohol'])
plt.show()    # Error 

#Z-Score: 
import numpy as np
from scipy.stats import zscore

z_scores = np.abs(zscore(df['Type']))
outliers = (z_scores > 3)
z_scores # there is no outlier in the dataset
#Points with a high absolute Z-score (e.g., greater than 3) 
#may be considered outliers.
#---------------------------------------------------------

#Feature Engineering
#1) Handling Missing Values:
#2) Encoding Categorical Variables:
#3) Creating Interaction Terms:  
df['Interaction_Type_Alcohol'] = df['Type'] * df['Alcohol'] 
df  
#############################################################
# There is scale diffrence between among the columns hence normalize it
# whenever there is mixed data apply normalization
# drop ID#
# We Know that there is scale diff. among the columns, which romove by using 
# Normalization or standasdization

df=df.drop(['Alcohol#'],axis=1) # Error

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

# Ideal cluster 
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
df.to_csv("wine.csv",encoding="utf-8")
import os
os.getcwd()
####################################################
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
####################Using Screen Plot############################
'''
The scree plot is a graphical tool used to determine 
the optimal number of clusters in K-means clustering.
'''





