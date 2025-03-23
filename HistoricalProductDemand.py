#Load libraries
from operator import index

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pickle

from fontTools.misc.bezierTools import epsilon
from nltk.sem.chat80 import label_indivs
from pandas.core.common import random_state
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
df = pd.read_csv('Historical Product Demand.csv').dropna()
df['Order_Demand'] = df['Order_Demand'].astype(str).str.replace("[(]", "-", regex=True).str.replace("[)]", "", regex=True).astype(float)
#turn demands into float and replace () with negative sign (encoding this column at the same time)
df['Product_Code'] = df['Product_Code'].astype(str).str.replace("Product_", "", regex=True).astype(float)
#remove the string "Product_" from Product Code (encoding this column with just the code at the same time)


#plt.figure(figsize = (12,6))
#sns.scatterplot(x=df['Product_Code'], y = df['Order_Demand'], s = 50)
#plt.show()

#print(df.info())
#print(df.describe())

#2. Preprocessing - Encoding
label= LabelEncoder()
onehot= OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
#to allow one hot encoder output a pandas data frame, which can be concatenated with other data frame

#2.1 label encode the date
df['Date'] = label.fit_transform(df['Date'])

#2.2 one hot encode the warehouse and product category
to_be_onehot_encoded = df.iloc[:,1:len(df.transpose())-2]
onehot_encode = onehot.fit_transform(to_be_onehot_encoded)



#3 Splitting data set
#3.1 assign dependent and independent variable
y=df.iloc[:,-1]
#x = pd.concat([pd.concat([df['Product_Code'],onehot_encode],axis=1),df['Date']],axis=1)
#x = pd.concat([onehot_encode,df['Date']],axis=1)
#x = df['Date'].shape(-1,1)
x=onehot_encode

#concat three encoded data frame, product code, combined one hot encoder of warehouse and category, and date
#print(x.describe())

#3.2 splitting training, testing, and validating data
X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)


#4 Training
#4.1 Decision Tree Methond
print('Running Scenarios and Searching for best methods:')
criterion = ['squared_error','friedman_mse','absolute_error','poisson']
#criterion = ['mse','mae']
gen_tracking_criterion=[]
gen_tracking_tree_score=[]
for i in criterion:
    m_tree = DecisionTreeRegressor(criterion=i,random_state=0).fit(X_train, y_train)
    y_pred = m_tree.predict(X_val)
    score = r2_score(y_val, y_pred)
    gen_tracking_tree_score.append(score)
    gen_tracking_criterion.append(i)


best_tree_score=max(gen_tracking_tree_score)
best_criterion=gen_tracking_criterion[gen_tracking_tree_score.index(best_tree_score)]

print('The best Decision Tree Criterion using validation set is:'+str(best_criterion)+", and the score is: "+str(best_tree_score))


#4.2 SVR method
ker=['linear','poly','rbf','sigmoid']
#ker=['linear']
g = 'auto'
e=1
c = 0.02
gen_tracking_method = []
gen_tracking_score = []
for i in ker:
    if i == 'poly':
        sub_tracking_score=[]
        for j in range(2,5,1):
            m_svr = SVR(kernel=i, degree=j, gamma=g, epsilon=e, C=c).fit(X_train, y_train)
            y_pred = m_svr.predict(X_val)
            score = r2_score(y_val, y_pred)
            sub_tracking_score.append(score)


        method=str(i)+" degree "+ str(sub_tracking_score.index(max(sub_tracking_score))+2)

        gen_tracking_score.append(max(sub_tracking_score))
        gen_tracking_method.append(method)
        #print(sub_tracking_score)


    else:
        m_svr = SVR(kernel=i, gamma=g, epsilon=e, C=c).fit(X_train, y_train)
        y_pred = m_svr.predict(X_val)
        score = r2_score(y_val, y_pred)
        gen_tracking_score.append(score)
        gen_tracking_method.append(i)

best_score=max(gen_tracking_score)
best_method=gen_tracking_method[gen_tracking_score.index(best_score)]

print('The best SVR method using validation set is:'+str(best_method)+", and the score is: "+str(best_score))



#5 Testing
print('')
print('Final Testing:')
#5.1 Decision Tree
m_tree = DecisionTreeRegressor(criterion=best_criterion, random_state=0).fit(X_train, y_train)
y_pred_tree = m_tree.predict(X_test)
score_tree = r2_score(y_test, y_pred_tree)
print('The Decision Tree using '+str(best_criterion)+" Criterion has a R square score of: "+str(score_tree))
#5.2 SVR
if best_method[:4] =='poly':

    m_svr = SVR(kernel=best_method[:4], degree=int(best_method[-1]), gamma=g, epsilon=e, C=c).fit(X_train, y_train)
else:
    m_svr = SVR(kernel=best_method, gamma=g, epsilon=e, C=c).fit(X_train, y_train)

y_pred_svr = m_svr.predict(X_test)
score_svr = r2_score(y_test, y_pred_svr)
print('The SVR using '+str(best_method)+" Kernel has a R square score of: "+str(score_svr))

#6. Visualization
fig1, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
fig1.tight_layout()

ax1.scatter(y_test,y_pred_tree)
ax1.plot(y_test,y_test,'r')
ax2.scatter(y_test,y_pred_svr)
ax2.plot(y_test,y_test,'r')

plt.show()