#Load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#1. Data Cleaning
df = pd.read_csv('Avocado.csv').dropna().drop(columns=['Date','region'])

#2. Preprocessing - Encoding
label= LabelEncoder()
onehot= OneHotEncoder()
encoder = LabelEncoder()
df['type']= encoder.fit_transform(df['type'])

#df['type']= onehot.fit_transform(df['type']).toarray()
#plt.figure(figsize=(16,8))
#sns.heatmap(df.corr(),annot=True)
#plt.show()
x=df.drop(columns=['AveragePrice'])
y=df.iloc[:,1]

X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)


#3. Training and testing with validation set
print('Running Scenarios Searching for the best parameters settings for KNN and Random Forest Methods: ')

#KNN
score_array = []
for i in range(1,10):
    m = KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)
    y_pred_knn=m.predict(X_val)
    score = r2_score(y_val,y_pred_knn)
    score_array.append(score)

max_score = max(score_array)
best_K=score_array.index(max_score) + 1
print('Best KNN score using validation set is: ',max_score, "with the best K value of",best_K)



#RandomForest
score_array2 = []
for j in range(5,25):
    m2 = RandomForestRegressor(n_estimators=j, criterion='friedman_mse', random_state=0, ).fit(X_train, y_train)
    y_pred_rf = m2.predict(X_val)
    score2 = r2_score(y_val,y_pred_rf)
    score_array2.append(score2)

max_score2 = max(score_array2)
best_num_estimator=score_array2.index(max_score2) + 1
print('Best Random Forest score using validation set is: ',max_score2, "with the best number of estimator is",best_num_estimator)



#4. Testing with testing data set
print('')
print('Testing data with testing set')
#KNN
m = KNeighborsRegressor(n_neighbors=best_K).fit(X_train, y_train)
y_pred_knn = m.predict(X_test)
score_test = r2_score(y_test,y_pred_knn)


#Random Forest
m2 = RandomForestRegressor(n_estimators=best_num_estimator, criterion='friedman_mse', random_state=0, ).fit(X_train, y_train)
y_pred_rf = m2.predict(X_test)
score2_test = r2_score(y_test,y_pred_rf)


print('The score for KNN Method on testing set is ',score_test, 'with K value of', best_K)
print('The score for Random Forest Method on testing set is ',score2_test, 'with ', best_num_estimator,'number of estimators')

#5. Visualization
fig1, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
fig1.tight_layout()
ax1.set_title('KNN Prediction with R square of: ' + str (round(score_test,2)))
ax1.scatter(y_test,y_pred_knn)
ax1.plot(y_test,y_test,'r')
ax2.set_title('Random Forest Prediction with R square of: ' + str (round(score2_test,2)))
ax2.scatter(y_test,y_pred_rf)
ax2.plot(y_test,y_test,'r')

plt.show()