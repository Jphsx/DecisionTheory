#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:10:54 2020

@author: j342a201
"""

from scipy import stats
array = stats.t.rvs(df=8, size=20)

import numpy as np
X = np.ones(10,dtype='int16')
X = np.ones(10,dtype='float32')

#X.dtype
#X.itemsize()
Y = np.full( (4,5), 2.71, dtype='int16')

Z=np.linspace(-1,6,15).reshape( (3,5))


G = [np.random.normal(loc=2,scale=5) for i in range(25) ]
mean = np.mean(G)
var = np.var(G)


T = (mean - 2)/np.sqrt(var/25)
pval = stats.t.cdf(x=T,df=2)

import pandas as pd

full= pd.read_csv('Downloads/PRSA_Data_Shunyi_20130301-20170228.csv')
print(full.head)
