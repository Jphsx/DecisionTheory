#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:03:57 2020

@author: j342a201
"""

import pandas as pd

full= pd.read_csv('Downloads/PRSA_Data_Shunyi_20130301-20170228.csv')
print(full.head)

print(full[2:7])

print(full['hour'])

print(full.hour)

print(full[full.RAIN > 0])

print(full.T.shape)
print(full.isna().sum())

print(full.columns)

comb = full[ [ 'month', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2',
       'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM' ] ]

print(comb.head)

X = comb.dropna()[[ 'month', 'hour', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM' ]]
print(X.head)
Y = comb.dropna()[['PM2.5', 'PM10', 'SO2', 'NO2','CO', 'O3']]
print(Y.head)

import matplotlib.pyplot as plt

import numpy as np
fig = plt.figure()
x = np.linspace(-5,5,100)
plt.plot(x,np.exp(x)/(1+np.exp(x)))

plt.plot(x,np.tanh(x),'--')

from matplotlib.image import imread
img = imread('Downloads/1024px-Kansas_Jayhawks_logo.svg.png')
print("img")
print(img.shape)
plt.imshow(img[:,:,0])


for c in range(4):
    plt.subplot(2,2,c+1)
    plt.imshow(img[:,:,c], cmap=plt.cm.binary)
    plt.xticks([])
    plt.xticks([])
    plt.xlabel('Channel')    
    
from PIL import Image
pic = Image.open('Downloads/1024px-Kansas_Jayhawks_logo.svg.png').convert('L')
img2 = np.asarray(pic, dtype='int32')
plt.imshow( img2,cmap=plt.cm.binary)

