# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:21:38 2018

@author: Yi Tai
"""

import pandas as pd

weather = [0.63, 0.17, 0.20]
print("天氣機率(Sunny, Cloudy, Rainy):", weather)
weatherList = ['Sunny', 'Cloudy', 'Rainy']
humidityList = ['VeryDry', 'Dry', 'Wet', 'VeryWet']


#隔天天氣機率
weather_Predict = [[0.5,0.375,0.125], #sunny ->[sunny,cloudy,rain]
                   [0.25,0.125,0.625], #cloudy ->[sunny,cloudy,rain]
                   [0.25,0.375,0.375]] #rain ->[sunny,cloudy,rain]
print(pd.DataFrame(data = weather_Predict, index = weatherList, columns = weatherList))
print()
#濕度
weather_Humidity =[[0.6,0.2,0.15,0.05], #sunny [VD,D,W,VW]
                   [0.25,0.25,0.25,0.25], #cloudy [VD,D,W,VW]
                   [0.05,0.10,0.35,0.5]] #rain [VD,D,W,VW]
print(pd.DataFrame(data = weather_Humidity, index = weatherList, columns = humidityList))
print()
#已知連續三天的天氣狀況為 (VD,D,W)
humidities = {1 : 'VeryDry', 2 : 'Dry', 3 : 'Wet'}
print(humidities)
print()
a1=[]
a2=[]
a3=[]
count=0
a_Register=0
for i in humidities:
    if i==1:
        for j in weather:
            a1.append(j*weather_Humidity[count][0])
            count+=1    
    if i==2:
        for j in range(0,len(a1)):            
            for k in range(0,len(weatherList)):
                a_Register+=(a1[k]*weather_Predict[k][count]*weather_Humidity[count][1])
            a2.append(a_Register)
            a_Register=0      
            count+=1
    elif i==3:
        for j in range(0,len(a2)):            
            for k in range(0,len(weatherList)):
                a_Register+=(a2[k]*weather_Predict[k][count]*weather_Humidity[count][2])
            a3.append(a_Register)
            a_Register=0      
            count+=1
    count=0
print(a1)
print("argMax(a1,y)=a1(",weatherList[a1.index(max(a1))],")=",max(a1),"\n")
print(a2)
print("argMax(a2,y)=a2(",weatherList[a2.index(max(a2))],")=",max(a2),"\n")
print(a3)
print("argMax(a3,y)=a3(",weatherList[a3.index(max(a3))],")=",max(a3),"\n")