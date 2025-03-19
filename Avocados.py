#Load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pickle
from sklearn.preprocessing import LabelEncoder
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#1. Data Cleaning
df = pd.read_csv('Avocado.csv').dropna().drop(columns=['Date','region'])

#2. Preprocessing - Encoding
encoder= LabelEncoder()
df['type']= encoder.fit_transform(df['type'])
plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),annot=True)
#plt.show()
x=df.drop(columns=['AveragePrice'])
y=df.iloc[:,1]
X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)
#Training and testing with validation set
#KNN
score_array = []
for i in range(1,10):
    m = KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)
    score = m.score(X_val, y_val)
    score_array.append(score)

max_score = max(score_array)
best_K=score_array.index(max_score) + 1
print('Best KNN score using validation set is: ',max_score, "with the best K value of",best_K)

m = KNeighborsRegressor(n_neighbors=best_K).fit(X_train, y_train)
score_test = m.score(X_test, y_test)



#RandomForest
score_array2 = []
for j in range(5,25):
    m2 = RandomForestRegressor(n_estimators=j, criterion='friedman_mse', random_state=0, ).fit(X_train, y_train)
    score2 = m2.score(X_val, y_val)
    score_array2.append(score2)

max_score2 = max(score_array2)
best_num_estimator=score_array2.index(max_score2) + 1
print('Best Random Forest score using validation set is: ',max_score2, "with the best number of estimator is",best_num_estimator)

m2 = RandomForestRegressor(n_estimators=best_num_estimator, criterion='friedman_mse', random_state=0, ).fit(X_train, y_train)
score2_test = m2.score(X_test, y_test)

print('The score for KNN Method on testing set is ',score_test, 'with K value of', best_K)
print('The score for Random Forest Method on testing set is ',score2_test, 'with ', best_num_estimator,'number of estimators')