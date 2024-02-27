# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:57:28 2023

@author: urmii
"""
#Problem Statement: - 
'''A film distribution company wants to target audience based on their likes and
 dislikes, you as a Chief Data Scientist Analyze the data and come up with
 different rules of movie list so that the business objective is achieved.
'''
'''maximize=increase quality and performance of film so the automatically 
            likes are getting increases of the particular film
minimize=decrease the dislikes'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

movies = pd.read_csv('C:/2-Datasets/my_movies.csv')
movies
movies.columns

movies.shape
#(2000, 11)

movies.dtypes
'''
Sixth Sense      int64
Gladiator        int64
LOTR1            int64
Harry Potter1    int64
Patriot          int64
LOTR2            int64
Harry Potter2    int64
LOTR             int64
Braveheart       int64
Green Mile       int64
dtype: object
'''
#all the columns are of int type 
b= movies.describe()
print(b)
'''
         Sixth Sense  Gladiator      LOTR1  ...       LOTR  Braveheart  Green Mile
  count    10.000000  10.000000  10.000000  ...  10.000000   10.000000   10.000000
  mean      0.600000   0.700000   0.200000  ...   0.100000    0.100000    0.200000
  std       0.516398   0.483046   0.421637  ...   0.316228    0.316228    0.421637
  min       0.000000   0.000000   0.000000  ...   0.000000    0.000000    0.000000
  25%       0.000000   0.250000   0.000000  ...   0.000000    0.000000    0.000000
  50%       1.000000   1.000000   0.000000  ...   0.000000    0.000000    0.000000
  75%       1.000000   1.000000   0.000000  ...   0.000000    0.000000    0.000000
  max       1.000000   1.000000   1.000000  ...   1.000000    1.000000    1.000000

  [8 rows x 10 columns]
'''
movies.columns()
'''
Index(['Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
       'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile'],
      dtype='object')
'''

movies.dtypes
'''
Sixth Sense      int64
Gladiator        int64
LOTR1            int64
Harry Potter1    int64
Patriot          int64
LOTR2            int64
Harry Potter2    int64
LOTR             int64
Braveheart       int64
Green Mile       int64
dtype: object
'''
b=movies.describe()
b
#we can see that all the data is already normalised i.e
#in the range of 0 to 1 so no need to normalize the data

#Data prreprocessing --> feature engineering and data cleaning
#pairplot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(movies, height=3);
plt.show()

##pdf and cdf
##pdf and cdf
counts, bin_edges = np.histogram(movies['Sixth Sense'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

##pdf and cdf
counts, bin_edges = np.histogram(movies['Gladiator'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

##pdf and cdf
counts, bin_edges = np.histogram(movies['LOTR1'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

##pdf and cdf
counts, bin_edges = np.histogram(movies['Harry Potter1'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

##pdf and cdf
counts, bin_edges = np.histogram(movies['Patriot'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

## outliers treatment
movies.columns
sns.boxplot(movies['Sixth Sense'])
sns.boxplot(movies['Gladiator'])
sns.boxplot(movies['LOTR1'])
sns.boxplot(movies['Harry Potter1'])
sns.boxplot(movies['Patriot'])
sns.boxplot(movies['Harry Potter2'])
sns.boxplot(movies['Braveheart'])

#only child books, cooking books, do it books, geography books do not have outliers
#we need to remove all other outliers
#1  Sixth Sense
iqr = movies['Sixth Sense'].quantile(0.75)-movies['Sixth Sense'].quantile(0.25)
iqr
q1=movies['Sixth Sense'].quantile(0.25)
q3=movies['Sixth Sense'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['Sixth Sense'] =  np.where(movies['Sixth Sense']>u_limit,u_limit,np.where(movies['Sixth Sense']<l_limit,l_limit,movies['Sixth Sense']))
sns.boxplot(movies['Sixth Sense'])

#only child books, cooking books, do it books, geography books do not have outliers
#we need to remove all other outliers
#2  Gladiator
iqr = movies['Gladiator'].quantile(0.75)-movies['Gladiator'].quantile(0.25)
iqr
q1=movies['Gladiator'].quantile(0.25)
q3=movies['Gladiator'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['Gladiator'] =  np.where(movies['Gladiator']>u_limit,u_limit,np.where(movies['Gladiator']<l_limit,l_limit,movies['Gladiator']))
sns.boxplot(movies['Gladiator'])

#only child books, cooking books, do it books, geography books do not have outliers
#we need to remove all other outliers
#3  Harry potter1
iqr = movies['Harry Potter1'].quantile(0.75)-movies['Harry Potter1'].quantile(0.25)
iqr
q1=movies['Harry Potter1'].quantile(0.25)
q3=movies['Harry Potter1'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['Harry Potter1'] =  np.where(movies['Harry Potter1']>u_limit,u_limit,np.where(movies['Harry Potter1']<l_limit,l_limit,movies['Harry Potter1']))
sns.boxplot(movies['Harry Potter1'])

movies.to_csv('C:/2-Datasets/movies_processed.csv')

#data is already normalized and there are no null values

#apriori algo


#pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
#here we are going to use transactional data where in size of each row is 
#not fixed 
#we can not use pandas to load this unstructured data
#here function called open() is used 
#create an empty list

movies=[]
with open('C:/2-Datasets/movies_processed.csv') as f:book=f.read()

#splitting the data into separate transactions usinusing separator it is comma separated
#we can use new line charater
movies = movies.split('\n')
#earlier groceries ds was in string format, now it will chang to
#9836, each item is comma separated 
#our main goal is to calculate #A #c
#now let us separate out each item from the groceries list

movies_list =[]
for i in movies:
    movies_list.append(i.split(','))
    
#split function will separate each item from each list, whenever it will find 
#in order to generate association rules you can directly use groceries_list
#now let separate out each item from the groceries_list
all_movies_list = [i for item in movies_list for i in item ]
#you will get all the items occured in all transactions
#we will get 43368 items in various transactions



#now let us count frequency of each item
#we will import collections package whih has counter functionm which 
#will 

from collections import Counter 
item_frequencies = Counter(all_movies_list)

#item_frequencies is basically dictionary having x[0] as a key x[1]=values
#we want to access values and sort based on the count that occures in it

#it will show the count of each item purchsed in every transactions

item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])

#when execute this, item frequencies will be in sorted form, in the form of 
#iem name with count
#let us separate out items and their count

items = list(reversed([i[0] for i in item_frequencies]))

#when we execute this, ietm frequencies will be in sorted form 
# in the form of tuple
#item name with count 
#let us separate out items and their count
items = list(reversed([i[1] for i in item_frequencies]))
frequencies = list(reversed([i[1] for i in item_frequencies]))

#here you will get count of purchased of each item 
#now let us plot bar graph  of item frequencies
import matplotlib.pyplot as plt
#here we are taking frequencies from zero to 11, you can try any other

plt.bar(height = frequencies[0:11],x=list(range(0,11)))
plt.xticks(list(range(0,11)),items[0:11])

#plt.xtricks, you can specify the rotation for the trick
#label in degrees or with keywords

plt.xlabel("items")
plt.ylabel("count")
plt.show()

import pandas as pd

#now let us try to establish association rule mining
#we  have groceries list in the format, we need to convert it in dataframe
movies_series = pd.DataFrame(pd.Series(movies_list))
#now we will get dataframe of size 9836X1 size, column
#comprises of multiple items
#we had extra row created, check the groceries_series
#last row is empty let us first delete it

movies_series = movies_series.iloc[:9835,:]
#we have taken rows from 0 to 9834 and columns 0 to all
#groceries series has column having name 0  to all
#groceries series has column having name 0 let us rename as transactions
movies_series.columns = ['Sixth Sense']
#now we will have to apply 1 hot encoding, before that in
#one column there are various items separated by ',
#let us separate it with '*'
x = movies_series['Sixth Sense'].str.join(sep = '*')
#check the x in variable explorer which has * separator rather the ','
x = x.str.get_dummies(sep = '*')
#you will get one hot encoded dataframe of size 9835x169
#This is our input data to apply apriori algorithm,
#it will generate !169 rules, min_support_value
#it is 0.0075 (it must be between 0 to 1)
#you can give any number must be between 0 to 1

frequent_itemsets = apriori(x, min_support = 0.0075,max_len = 4,use_colnames = True)
#you will get support values for 1,2,3 and 4 max items
#let us sort these support values
frequent_itemsets.sort_values('support',ascending = False, inplace = True)
#support values will be sorted in descending order
#Even EDA was also have the same trend , in EDA there was count
#and here it support value 
#we will generate association rules This association rules this association
#rule will calculate all this matrix
#of each and every combination

rules = association_rules(frequent_itemsets,metric='lift',min_threshold=1)
#this generate association rules of size 1198X9 columns
#comprizesof antecedents and consequents
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

 


















