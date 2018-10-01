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
from sklearn.linear_model import LinearRegression

df= pd.read_csv('mlb_2017_regular_season_top_hitting.csv')

#選手，隊伍，比賽場數，打擊數，安打，全壘打，三振
X = pd.DataFrame(df, columns=['Player','Team','G','AB','H','SO','R','BB','AVG?'])

#計算保送率---------------
df['BBP']=df['BB']/df['AB']
#------------------------
print(X.head())

sns.set(style='whitegrid', context='notebook')

cols = ['HR', '2B', '3B', 'H']

cols2=['AVG?','OBP','SLG','BBP']

print(df[cols2].describe()) #資料集描述性統計

sns.pairplot(df[cols2], size=2.5) #kind="reg",

plt.tight_layout()

plt.show()

#分類值1----------守備位置分類----
size_mapping = {"DH": 10,
                "RF": 9,
                "CF": 8,
                "LF": 7,
                "SS": 6,
                "3B": 5,
                "2B": 4,
                "1B": 3,
                "C": 2,
                "P": 1}
print(size_mapping)
df['Pos'] = df['Pos'].map(size_mapping)
print(df.head(10))
#分類值2----------隊伍名稱分類-----
label_encoder = preprocessing.LabelEncoder()
df["Team"] = label_encoder.fit_transform(df["Team"]) #將字串數值化
print(df.head(10))

#--------------------------------
teamHOU = (df['Team'] == 10)
teamHOUdf= df[teamHOU]
print(teamHOUdf)

#線性回歸------------------------
# 轉換維度
dfops = np.array(df['OPS'])#.reshape((len(df['OPS']), 1))
#dfr = np.array(df['AB'])#.reshape((len(df['R']), 1))
dfops=np.reshape(dfops,(len(dfops),1))
#dfr=np.reshape(dfr,(len(dfr),1))
dfp = [0.9]*144
dfp=np.array(dfp)
dfp=np.reshape(dfp,(len(dfp),1))

lm = LinearRegression()
lm.fit(dfops, dfp)

x1 = df[["AB"]].values
y1 = df["H"].values

slr = LinearRegression()
slr.fit(x1, y1)

print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
def lin_regplot(x1, y1, model):
    plt.scatter(x1, y1, c='lightblue')
    plt.plot(x1, model.predict(x1), color='red', linewidth=2)    
    return 

lin_regplot(x1, y1, slr)
plt.xlabel('AB')
plt.ylabel('H')
plt.tight_layout()
plt.show()
#MSE----------------------------
# 模型績效
mse = np.mean((lm.predict(dfops) - dfp) ** 2)
r_squared = lm.score(dfops, dfp)

# 印出模型績效
print("Mean squared error:",mse)
#print("R-squared:",r_squared)

#-------------------------------


             


