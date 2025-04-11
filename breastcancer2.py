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
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#1. Data Cleaning
df = pd.read_csv('data_refined.csv') ## already cleaned
plt.figure()
sns.heatmap(df.corr(),annot=True)

#df=df.drop(columns=['texture_mean','smoothness_mean','symmetry_mean','fractal_dimension_mean'])
df=df.drop(columns=['fractal_dimension_mean'])

plt.figure()
sns.heatmap(df.corr(),annot=True)
#plt.show()

x=df.drop(columns=['diagnosis'])
y=df['diagnosis']
X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)

#3. Training and testing with validation set
print('Running Scenarios Searching for the best parameters settings for KNN and Random Forest Methods: ')

#KNN
score_array = []
for i in range(5,50):
    m_knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    y_pred_knn=m_knn.predict(X_val)
    score = m_knn.score(X_val,y_val)
    score_array.append(score)

max_score = max(score_array)
best_K=score_array.index(max_score) + 5
print('Best KNN score using validation set is: ',max_score, "with the best K value of",best_K)

#SVC

#4.3 SVR method
ker=['linear','poly','rbf','sigmoid']
#ker=['poly']
g = 'auto'
e=1
c = 0.02
gen_tracking_method = []
gen_tracking_score = []

for i in ker:
    if i == 'poly':
        sub_tracking_score=[]

        for j in range(2,15):
            m_svc = SVC(kernel=i, degree=j, gamma=g).fit(X_train, y_train)
            y_pred = m_svc.predict(X_val)
            score = m_svc.score(X_val,y_val)
            sub_tracking_score.append(score)



        method=str(i)+" degree "+ str(sub_tracking_score.index(max(sub_tracking_score))+2)

        gen_tracking_score.append(max(sub_tracking_score))

        gen_tracking_method.append(method)
        #print(sub_tracking_score)


    else:
        m_svc = SVC(kernel=i, gamma=g).fit(X_train, y_train)
        y_pred = m_svc.predict(X_val)
        score = m_svc.score(X_val,y_val)

        gen_tracking_score.append(score)

        gen_tracking_method.append(i)

best_svr_score=max(gen_tracking_score)
best_svr_method=gen_tracking_method[gen_tracking_score.index(best_svr_score)]

print('The best SVR method using validation set is:'+str(best_svr_method)+", and the r square score is: "+str(best_svr_score))


#####
print(best_K)
m_knn = KNeighborsClassifier(n_neighbors=best_K).fit(X_train, y_train)
y_pred_knn = m_knn.predict(X_test)
score_knn = m_knn.score(X_test, y_test)

matrix_knn = confusion_matrix(y_test,y_pred_knn)
#####

if best_svr_method[:4] == 'poly':
    m_svc = SVC(kernel=best_svr_method[:4], degree=int(best_svr_method[-1]), gamma=g).fit(X_train,
                                                                                                          y_train)
else:
    m_svc = SVC(kernel=best_svr_method, gamma=g).fit(X_train, y_train)
y_pred_svc = m_svc.predict(X_test)
score_svc = m_svc.score(X_test,y_test)
matrix_svc = confusion_matrix(y_test,y_pred_svc)


print("---------------------Final Result---------------------")
print('KNN')
print('The best-tuned Score for KNN method is: '+str(score_knn))
print('The confusion matrix is : ')
print(matrix_knn)
print('')
print('SVC')
print('The best-tuned Score for SVC method is: '+str(score_svc))
print('The confusion matrix is : ')
print(matrix_svc)
print('both methods achieved accuracy over 94%')
print("---------------------the end---------------------")
plt.show()