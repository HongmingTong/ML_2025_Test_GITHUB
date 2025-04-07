#Load libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pickle

from matplotlib.bezier import split_bezier_intersecting_with_closedpath
from matplotlib.lines import lineStyles
from nltk.app.nemo_app import colors
from nltk.sem.chat80 import label_indivs
from pandas.core.common import random_state
from scipy.cluster.vq import kmeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale, MinMaxScaler
from pandas.core.interchange.dataframe_protocol import Column
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans


#1. Data Cleaning
df = pd.read_csv('Wholesale customers data.csv').dropna().drop(columns=['Region','Channel'])
#print(df.describe())

#2 Scaling
scale=MinMaxScaler()
scaled_df=scale.fit_transform(df)

#3 Elbow plot with K Mean

dist=[]
delta=[]
delta2=[]
k=range(1,50)
for i in k:
    m=KMeans(n_clusters=i).fit(scaled_df)
    dist.append(m.inertia_)
    d=(m.inertia_-dist[i-2])
    delta.append(d)
    d2 = (d-delta[i-2])
    delta2.append(d2)
    print('The intertia is: '+ str(m.inertia_)+ 'for k='+str(k))

######### use second derivative of K Elbow plot to find the rate of change in inertia
#### when the rate of change of inertia is 0, the K-elbow is in a steady state###
#### the opimum K is defined as the last K val
upper=[]
lower=[]
mean_d2 = np.array(delta2).mean()
std_d2 = np.array(delta2).std()*0.25
for i in k:
    upper.append(mean_d2+std_d2)
    lower.append(mean_d2 - std_d2)

print('The optimum k is 12 given the range of K is 1 to 50 clusters')
plt.figure()
plt.plot(k,dist)
plt.plot(12,dist[12-1],marker="o",color='red')
#scatter=True, fit_reg=False,marker="o",color='red',label='Optimum K = 12')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Plot for Optimum K')
plt.suptitle('Optimum K = 12')
plt.show()
