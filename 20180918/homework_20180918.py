# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:40:08 2018

@author: Yi Tai
"""

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df= pd.read_csv('mlb_2017_regular_season_top_hitting.csv')

X = pd.DataFrame(df, columns=['RK','Player','Team','H','HR'])

print(X.head())

sns.set(style='whitegrid', context='notebook')

cols = ['HR', '2B', '3B', 'H']

sns.pairplot(df[cols], size=2.5);

plt.tight_layout()

plt.show()