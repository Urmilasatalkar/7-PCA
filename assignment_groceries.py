# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:38:22 2023

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

groceries = pd.read_csv('C:/2-Datasets/groceries.csv')
groceries
groceries.columns
'''
Index(['citrus fruit', 'semi-finished bread', 'margarine', 'ready soups',
       'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8',
       'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12',
       'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16',
       'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20',
       'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24',
       'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28',
       'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31'],
      dtype='object')
'''
groceries.shape
#(9834, 32)

groceries.dtypes
'''citrus fruit           object
semi-finished bread    object
margarine              object
ready soups            object
Unnamed: 4             object
Unnamed: 5             object
Unnamed: 6             object
Unnamed: 7             object
Unnamed: 8             object
Unnamed: 9             object
Unnamed: 10            object
Unnamed: 11            object
Unnamed: 12            object
Unnamed: 13            object
Unnamed: 14            object
Unnamed: 15            object
Unnamed: 16            object
Unnamed: 17            object
Unnamed: 18            object
Unnamed: 19            object
Unnamed: 20            object
Unnamed: 21            object
Unnamed: 22            object
Unnamed: 23            object
Unnamed: 24            object
Unnamed: 25            object
Unnamed: 26            object
Unnamed: 27            object
Unnamed: 28            object
Unnamed: 29            object
Unnamed: 30            object
Unnamed: 31            object
dtype: object
'''

#all the columns are of int type 
b= groceries.describe()
print(b)
'''citrus fruit semi-finished bread  ...       Unnamed: 30 Unnamed: 31
count          9834                7675  ...                 1           1
unique          158                 151  ...                 1           1
top         sausage          whole milk  ...  hygiene articles     candles
freq            825                 654  ...                 1           1

[4 rows x 32 columns]'''







