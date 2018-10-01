# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:40:08 2018

@author: Yi Tai
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import seaborn as sns


df= pd.read_csv('mlb_2017_regular_season_top_hitting.csv')

#選手，隊伍，比賽場數，打擊數，安打，全壘打，三振
X = pd.DataFrame(df, columns=['Player','Team','G','AB','H','SO','HR','BB','AVG?'])
df['BBP']=df['BB']/df['AB']
print(X.head())

sns.set(style='whitegrid', context='notebook')

cols = ['HR', '2B', '3B', 'H']

cols2=['AB','H','OBP','AVG?']

print(df[cols2].describe()) #資料集描述性統計

sns.pairplot(df[cols2], size=2.5) #kind="reg",

plt.tight_layout()

plt.show()

#相關係數-----------------
x = np.array(df[['AB']])
y = np.array(df[['H']])
n = len(x)
x_mean = x.mean()
y_mean = y.mean()

diff = (x-x_mean)*(y-y_mean)
covar = diff.sum()/n
print("共變異數:", covar)
corr = covar/(x.std()*y.std())
print("相關係數:", corr)

df1 = pd.DataFrame({"AB":df['AB'],
                   "H":df['H']})
print(df1.corr())
#------------------------

#資料正規化---------------

print(df1.head())
scaler = preprocessing.StandardScaler()
np_std = scaler.fit_transform(df1)
df_std = pd.DataFrame(np_std, columns=["AB", "H"])
print(df_std.head())
df_std.plot(kind="scatter", x="AB", y="H")
plt.show()

#------------------------

#刪除and補齊遺漏值---------------
# 刪除所有 NaN 的記錄
df2 = df.dropna()
print(df2.head())

df3 = df.dropna(how="any")
print(df3.head())

df4 = df.dropna(how="all")
print(df4.head())

df5 = df.dropna(subset=["AVG?", "H"])
print(df5.head())

# 填補遺失資料
df6 = df.fillna(value=1)
print(df6.head())

df["AVG?"] = df["AVG?"].fillna(df["AVG?"].mean())
print(df.head())

df["H"] = df["H"].fillna(df["H"].median())
print(df.head())

#z_scores----------------

s_df1_AB=pd.Series(df1['AB'])
m=s_df1_AB.mean()
s=s_df1_AB.std()
z_scores = []
for x in df1['AB']:
    z=(x-m)/s
    z_scores.append(z)
index=np.arange(len(df1['AB']))
plt.bar(index,z_scores)
plt.show()

#------------------------

#特徵最大最小值縮放-----------

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
np_minmax = scaler.fit_transform(df1)
df_minmax = pd.DataFrame(np_minmax, columns=["AB", "H"])
print(df_minmax.head())
df_minmax.plot(kind="scatter", x="AB", y="H")
plt.show()

#------------------------

#def dice_roll():
#    v=random.randint(1,6)
#    return v
#
#num_of_trials = range(100,10000,10)
#avgs=[]
#
#for num_of_trial in num_of_trials:
#    trials=[]
#    for trial in range(num_of_trial):
#        trials.append(dice_roll()) #
#    avgs.append(sum(trials)/float(num_of_trial))
#    
#plt.plot(num_of_trials,avgs)
#plt.xlabel("Number of Trials")
#plt.ylabel("Average")
#plt.show()    
    