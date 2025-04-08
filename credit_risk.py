#Load libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pickle

from sklearn.metrics import confusion_matrix
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
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#Load data
from sklearn.datasets import fetch_openml
df = fetch_openml('credit-g', version=4).frame.dropna()


#2. Preprocessing - Encoding
label= LabelEncoder()
onehot= OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')

#Numerical Feature Encoding 4 minimum
df_num = df[['credit_amount','age','duration','residence_since']]

# Catagorical Feature Encoding 3 minimum
df_cat1 = onehot.fit_transform(df['employment'].values.reshape(-1,1))
df_cat2 = onehot.fit_transform(df['credit_history'].values.reshape(-1,1))
df_cat3 = onehot.fit_transform(df['purpose'].values.reshape(-1,1))

# Independent/Dependent Variable
x = pd.concat([df_num,df_cat1,df_cat2,df_cat3],axis=1)


y= label.fit_transform(df['class'])

#Split data
X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)


#3.1 Feature Scaling
scale=MinMaxScaler()
X_train=scale.fit_transform(X_train)
X_val=scale.transform(X_val)
X_test=scale.transform(X_test)


#3. Training and testing with validation set
print('Running Scenarios Searching for the best parameters settings for KNN and Random Forest Methods: ')

#KNN
score_array = []
for i in range(1,20):
    m = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    y_pred_knn=m.predict(X_val)
    score=m.score(X_val,y_val)
    score_array.append(score)

max_score = max(score_array)
best_K=score_array.index(max_score) + 1
print('Best KNN score using validation set is: ',max_score, "with the best K value of",best_K)

#4. Testing with testing data set
print('---------------------Final Result-----------------------')
print('Testing data with testing set')
#KNN
m = KNeighborsClassifier(n_neighbors=best_K).fit(X_train, y_train)
y_pred_knn = m.predict(X_test)
score_test = m.score(X_test,y_test)
c_m=confusion_matrix(y_test,y_pred_knn)
print('The score for KNN Method on Final testing set is ',score_test, 'with K value of', best_K)
print('The confusion matric using the Final testing set is:')
print(c_m)

print('---------------------The end-----------------------')