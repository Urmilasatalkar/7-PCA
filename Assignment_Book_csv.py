# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:07:03 2023

@author: arudr
"""

'''
Problem Statement: -
Kitabi Duniya, a famous book store in India, which was established 
before Independence, the growth of the company was incremental year 
by year, but due to online selling of books and wide spread Internet
access its annual growth started to collapse, seeing sharp downfalls,
you as a Data Scientist help this heritage book store gain its
popularity back and increase footfall of customers and provide ways 
the business can improve exponentially, apply Association 
RuleAlgorithm, explain the rules, and visualize the graphs for clear
understanding of solution.

1.) Books.csv

'''
#business objective -
#max - sales of the books,increase footfall of customers, annual growth
#min - 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

books = pd.read_csv('C:/2-Datasets/book.csv')
books
books.columns
'''
Index(['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
       'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence'],
      dtype='object')
'''
books.shape
#(2000, 11)

books.dtypes
'''
ChildBks     int64      Quantitative,Nominal,Discrete  relavent
YouthBks     int64      Quantitative,Nominal,Discrete  relavent
CookBks      int64      Quantitative,Nominal,Discrete  relavent
DoItYBks     int64      Quantitative,Nominal,Discrete  relavent
RefBks       int64      Quantitative,Nominal,Discrete  relavent
ArtBks       int64      Quantitative,Nominal,Discrete  relavent
GeogBks      int64      Quantitative,Nominal,Discrete  relavent
ItalCook     int64      Quantitative,Nominal,Discrete  relavent
ItalAtlas    int64      Quantitative,Nominal,Discrete  relavent
ItalArt      int64      Quantitative,Nominal,Discrete  relavent
Florence     int64      Quantitative,Nominal,Discrete  relavent
dtype: object
'''
#all the columns are of int type 
b= books.describe()
print(b)

'''
         ChildBks     YouthBks  ...      ItalArt     Florence
count  2000.000000  2000.000000  ...  2000.000000  2000.000000
mean      0.423000     0.247500  ...     0.048500     0.108500
std       0.494159     0.431668  ...     0.214874     0.311089
min       0.000000     0.000000  ...     0.000000     0.000000
25%       0.000000     0.000000  ...     0.000000     0.000000
50%       0.000000     0.000000  ...     0.000000     0.000000
75%       1.000000     0.000000  ...     0.000000     0.000000
max       1.000000     1.000000  ...     1.000000     1.000000

[8 rows x 11 columns]'''

#we can see that all the data is already normalised i.e
#in the range of 0 to 1 so no need to normalize the data

#Data prreprocessing --> feature engineering and data cleaning
#pairplot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(books, height=3);
plt.show()

##pdf and cdf

counts, bin_edges = np.histogram(books['ChildBks'], bins=10, 
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
books.columns
sns.boxplot(books['ChildBks'])
sns.boxplot(books['YouthBks'])
sns.boxplot(books['CookBks'])
sns.boxplot(books['DoItYBks'])
sns.boxplot(books['ArtBks'])
sns.boxplot(books['GeogBks'])
sns.boxplot(books['ItalCook'])
sns.boxplot(books['ItalAtlas'])
sns.boxplot(books['ItalArt'])
sns.boxplot(books['Florence'])

#only child books, cooking books, do it books, geography books do not have outliers
#we need to remove all other outliers
#1  YouthBks
iqr = books['YouthBks'].quantile(0.75)-books['YouthBks'].quantile(0.25)
iqr
q1=books['YouthBks'].quantile(0.25)
q3=books['YouthBks'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
books['YouthBks'] =  np.where(books['YouthBks']>u_limit,u_limit,np.where(books['YouthBks']<l_limit,l_limit,books['YouthBks']))
sns.boxplot(books['YouthBks'])

#2  ItalCook
iqr = books['ItalCook'].quantile(0.75)-books['ItalCook'].quantile(0.25)
iqr
q1=books['ItalCook'].quantile(0.25)
q3=books['ItalCook'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
books['ItalCook'] =  np.where(books['ItalCook']>u_limit,u_limit,np.where(books['ItalCook']<l_limit,l_limit,books['ItalCook']))
sns.boxplot(books['ItalCook'])

#3  ArtBks
iqr = books['ArtBks'].quantile(0.75)-books['ArtBks'].quantile(0.25)
iqr
q1=books['ArtBks'].quantile(0.25)
q3=books['ArtBks'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
books['ArtBks'] =  np.where(books['ArtBks']>u_limit,u_limit,np.where(books['ArtBks']<l_limit,l_limit,books['ArtBks']))
sns.boxplot(books['ArtBks'])

##4  ItalAtlas
iqr = books['ItalAtlas'].quantile(0.75)-books['ItalAtlas'].quantile(0.25)
iqr
q1=books['ItalAtlas'].quantile(0.25)
q3=books['ItalAtlas'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
books['ItalAtlas'] =  np.where(books['ItalAtlas']>u_limit,u_limit,np.where(books['ItalAtlas']<l_limit,l_limit,books['ItalAtlas']))
sns.boxplot(books['ItalAtlas'])

#5  ItalArt
iqr = books['ItalArt'].quantile(0.75)-books['ItalArt'].quantile(0.25)
iqr
q1=books['ItalArt'].quantile(0.25)
q3=books['ItalArt'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
books['ItalArt'] =  np.where(books['ItalArt']>u_limit,u_limit,np.where(books['ItalArt']<l_limit,l_limit,books['ItalArt']))
sns.boxplot(books['ItalArt'])

#6  Florence
iqr = books['Florence'].quantile(0.75)-books['Florence'].quantile(0.25)
iqr
q1=books['Florence'].quantile(0.25)
q3=books['Florence'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
books['Florence'] =  np.where(books['Florence']>u_limit,u_limit,np.where(books['Florence']<l_limit,l_limit,books['Florence']))
sns.boxplot(books['Florence'])


books.describe()

books.to_csv('C:/5-Recommendation/Books_processed.csv')

#data is already normalized and there are no null values

#apriori algo

#pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
#here we are going to use transactional data where in size of each row is 
#not fixed 
#we can not use pandas to load this unstructured data
#here function called open() is used 
#create an empty list

book=[]
with open('C:/5-Recommendation/Books_processed.csv') as f:book=f.read()

#splitting the data into separate transactions usinusing separator it is comma separated
#we can use new line charater
book = book.split('\n')
#earlier groceries ds was in string format, now it will chang to
#9836, each item is comma separated 
#our main goal is to calculate #A #c
#now let us separate out each item from the groceries list

book_list =[]
for i in book:
    book_list.append(i.split(','))
    
#split function will separate each item from each list, whenever it will find 
#in order to generate association rules you can directly use groceries_list
#now let separate out each item from the groceries_list
all_book_list = [i for item in book_list for i in item ]
#you will get all the items occured in all transactions
#we will get 43368 items in various transactions



#now let us count frequency of each item
#we will import collections package whih has counter functionm which 
#will 

from collections import Counter 
item_frequencies = Counter(all_book_list)

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
books_series = pd.DataFrame(pd.Series(book_list))
#now we will get dataframe of size 9836X1 size, column
#comprises of multiple items
#we had extra row created, check the groceries_series
#last row is empty let us first delete it

books_series = books_series.iloc[:9835,:]
#we have taken rows from 0 to 9834 and columns 0 to all
#groceries series has column having name 0  to all
#groceries series has column having name 0 let us rename as transactions
books_series.columns = ['ChildBks']
#now we will have to apply 1 hot encoding, before that in
#one column there are various items separated by ',
#let us separate it with '*'
x = books_series['ChildBks'].str.join(sep = '*')
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

 