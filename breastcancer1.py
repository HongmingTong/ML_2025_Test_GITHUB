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



#1. Data Cleaning
df = pd.read_csv('data.csv').iloc[:,:32].dropna().drop(columns=['id'])
df['diagnosis']=df['diagnosis'].astype(str)


#only take the mean for the analyses, ignor se and worst
df=df.iloc[:,:11]

#print(df.head())
#2. Preprocessing - Encoding
label= LabelEncoder()
onehot= OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')

df['diagnosis']=label.fit_transform(df['diagnosis'])
#print(df.info())

x=df.drop(columns=['diagnosis'])
y=df['diagnosis']

#3.1 Feature Scaling
scale=MinMaxScaler()
for i in (1,11):
    df.iloc[:,:i]=scale.fit_transform(df.iloc[:,:i])

print(df)
sns.pairplot(df)
plt.figure()
sns.heatmap(df.corr(),annot=True)


num_column=len(df.transpose())-1
for i in range(1,num_column):
    plt.figure()
    df_for_boxplot=pd.concat([df.iloc[:,i],df['diagnosis']],axis=1)
    col_name = str(df_for_boxplot.columns[0])
    sns.boxplot(data = df_for_boxplot,x='diagnosis',y=col_name)
    sns.violinplot(data = df_for_boxplot,x='diagnosis',y=col_name)

plt.show()

#Write to CSV
filename='data_refined.csv'

df.to_csv(filename,index=False)