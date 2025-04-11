#Load libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pickle

from joblib import PrintTime
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

#1. Data Cleaning
df = pd.read_csv('KaggleV2-May-2016.csv').dropna().drop(columns=['PatientId','AppointmentID','ScheduledDay','AppointmentDay','Neighbourhood']).rename(columns={'Hipertension':'Hypertension','Handcap':'Handicap'})
#print(df.info())

#2. Preprocessing - Encoding
label= LabelEncoder()
onehot= OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')

df['Gender']=label.fit_transform(df['Gender'])
df['No-show']=label.fit_transform(df['No-show'])
#print(df.info())

#3 Assigning Variables and Splitting Data
x=df.drop(columns=['No-show'])
y=df['No-show']
X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)

#3.1 Feature Scaling
scale=MinMaxScaler()
X_train=scale.fit_transform(X_train)
X_val=scale.transform(X_val)
X_test=scale.transform(X_test)


#4 Training

print('Running Scenarios and Searching for best methods:')

#Decision Tree Tuning
criterion_input = ['gini','entropy','log_loss']

criterion =[]
tree_score=[]

for i in criterion_input:
    m_tree = DecisionTreeClassifier(criterion=i,random_state=0).fit(X_train, y_train)
    y_pred = m_tree.predict(X_val)
    score = m_tree.score(X_val,y_val)
    tree_score.append(score)
    criterion.append(i)

best_tree_score=max(tree_score)
best_tree_criterion=criterion[tree_score.index(best_tree_score)]

print('The best Decision Tree Criterion using validation set is:'+str(best_tree_criterion)+", and the model score is: "+str(best_tree_score))

#RandomForest Tuning
estimators= range(1,100)

nestimator =[]
rf_score=[]

for j in estimators:
    m_rf = RandomForestClassifier(criterion='gini',n_estimators=j, random_state=0).fit(X_train, y_train)
    y_pred_rf = m_rf.predict(X_val)
    score = m_rf.score(X_val,y_val)
    rf_score.append(score)
    nestimator.append(j)

best_rf_score=max(rf_score)
best_rf_estimator=nestimator[rf_score.index(best_rf_score)]

print('The best number of estimator for Random Forest using validation set is: '+str(best_rf_estimator)+", and the model score is: "+str(best_rf_score))

#visual
plt.figure()
sns.lineplot(x=nestimator,y=rf_score,color='red',marker='o')



#5 Final Testing
print('')
print('Running Models based on the best tuned parameters')
#Decision Tree Testing
m_tree = DecisionTreeClassifier(criterion=best_tree_criterion,random_state=0).fit(X_train, y_train)
y_pred_tree_test= m_tree.predict(X_test)
score_tree_test = m_tree.score(X_test,y_test)
tree_matrix=confusion_matrix(y_test,y_pred_tree_test)


m_rf = RandomForestClassifier(criterion='gini',n_estimators=best_rf_estimator, random_state=0).fit(X_train, y_train)
y_pred_rf_test= m_rf.predict(X_test)
score_rf_test = m_rf.score(X_test,y_test)
rf_matrix=confusion_matrix(y_test,y_pred_rf_test)


print("---------------------Final Result---------------------")
print('Decision Tree')
print('The best-tuned Score for decision tree method is: '+str(score_tree_test))
print('The confusion matrix is : ')
print(tree_matrix)
print('')
print('Random Forest')
print('The best-tuned Score for random forest method is: '+str(score_rf_test))
print('The confusion matrix is : ')
print(rf_matrix)
print('Generally, in this case, the higher the number of estimator the better the result. The model accuracy peaks at 63 estimators and plateaus around 0.8 after that.')
print("---------------------the end---------------------")
plt.show()