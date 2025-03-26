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
df = pd.read_csv('insurance.csv').dropna()
#print(df.describe())


#2. Preprocessing - Encoding
label= LabelEncoder()
onehot= OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')

df['sex']=label.fit_transform(df['sex'])
df['smoker']=label.fit_transform(df['smoker'])

onehot_encode = onehot.fit_transform(df['region'].values.reshape(-1,1))





#3. Visulization
#change the df for visualization
df_plot = df
df_plot['region'] = label.fit_transform(df['region'])

age_bin= []
bmi_bin = []
num_bin=28

for i in range(0,num_bin+1,1):
    age_width = round((max(df_plot['age'])-min(df_plot['age']))*i/num_bin+min(df_plot['age']),2)
    age_bin.append(age_width)
    bmi_width = round((max(df_plot['bmi'])-min(df_plot['bmi']))*i/num_bin+min(df_plot['bmi']),2)
    bmi_bin.append(bmi_width)

age_label = age_bin[1:]
bmi_label = bmi_bin[1:]

df_plot['age'] = pd.cut(df_plot['age'], bins=age_bin, labels=age_label, include_lowest=True).astype(float)
df_plot['bmi'] =  pd.cut(df_plot['bmi'], bins=bmi_bin, labels=bmi_label, include_lowest=True).astype(float)


num_column=len(df_plot.transpose())-1
#plt.figure()
sns.pairplot(df_plot)

plt.figure()
sns.heatmap(df_plot.corr(),annot=True)


for i in range(0,num_column):
    plt.figure()
    df_for_boxplot=pd.concat([df_plot.iloc[:,i],df_plot['charges']],axis=1)
    col_name = str(df_for_boxplot.columns[0])
    sns.boxplot(data = df_for_boxplot,x=col_name,y='charges')

#plt.show()




#3 Splitting data set
#x=pd.concat([df.drop(columns=['region','charges']),onehot_encode],axis=1)# Replace original data with one hot encoded data
x=pd.concat([df_plot.drop(columns=['region','charges']),onehot_encode],axis=1)

#x=df.drop(columns=['region','charges','children','sex'])

y=df['charges']
X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)

#3.1 Feature Scaling
scale=MinMaxScaler()
X_train=scale.fit_transform(X_train)
X_val=scale.transform(X_val)
X_test=scale.transform(X_test)


#4 Training
print('Running Scenarios and Searching for best methods:')

#4.1 Decision Tree Method
criterion_input = ['squared_error','friedman_mse','absolute_error','poisson']
#criterion = ['mse','mae']
criterion =[]
tree_r2score=[]
tree_msescore=[]
tree_maescore=[]
for i in criterion_input:
    m_tree = DecisionTreeRegressor(criterion=i,random_state=0).fit(X_train, y_train)
    y_pred = m_tree.predict(X_val)
    r2score = r2_score(y_val, y_pred)
    msescore = mean_squared_error(y_val, y_pred)
    maescore = mean_absolute_error(y_val, y_pred)
    tree_r2score.append(r2score)
    tree_msescore.append(msescore)
    tree_maescore.append(maescore)
    criterion.append(i)

best_tree_score=max(tree_r2score)
best_tree_criterion=criterion[tree_r2score.index(best_tree_score)]

print('The best Decision Tree Criterion using validation set is:'+str(best_tree_criterion)+", and the r square score is: "+str(best_tree_score))
print("- the mse score is: " +str(tree_msescore[tree_r2score.index(best_tree_score)]))
print("- the mae score is: " +str(tree_maescore[tree_r2score.index(best_tree_score)]))


#4.2 Random Forest
criterion_input = ['squared_error','friedman_mse','absolute_error','poisson']
rf_r2score = -2
rf_msescore = 0
rf_maescore = 0
rf_criterion = "null"
n_estimator = 0

for i in criterion_input:

    for j in range(15, 25):
        m_rf = RandomForestRegressor(n_estimators=j, criterion=i, random_state=0).fit(X_train, y_train)
        y_pred_rf = m_rf.predict(X_val)
        r2score = r2_score(y_val, y_pred_rf)
        msescore = mean_squared_error(y_val, y_pred_rf)
        maescore = mean_absolute_error(y_val, y_pred_rf)

        if r2score > rf_r2score:
            rf_r2score = r2score
            rf_msescore = msescore
            rf_maescore = maescore
            rf_criterion = i
            n_estimator= j



best_rf_score =rf_r2score
best_num_estimator=n_estimator
best_rf_criterion=rf_criterion
print('The best Random Forest Criterion using validation set is:',best_rf_criterion, "with the best number of estimator is",best_num_estimator, "and the r square score is: ",best_rf_score)
print("- the mse score is: " +str(rf_msescore))#[rf_r2score.index(max_score)]))
print("- the mae score is: " +str(rf_maescore))#[rf_r2score.index(max_score)]))



#4.3 SVR method
#ker=['linear','poly','rbf','sigmoid']
ker=['linear']
g = 'auto'
e=1
c = 0.02
gen_tracking_method = []
gen_tracking_r2score = []
gen_tracking_msescore = []
gen_tracking_maescore = []
for i in ker:
    if i == 'poly':
        sub_tracking_r2score=[]
        sub_tracking_msescore=[]
        sub_tracking_maescore=[]
        for j in range(2,5,1):
            m_svr = SVR(kernel=i, degree=j, gamma=g, epsilon=e, C=c).fit(X_train, y_train)
            y_pred = m_svr.predict(X_val)
            r2score = r2_score(y_val, y_pred)
            msescore = mean_squared_error(y_val, y_pred)
            maescore = mean_absolute_error(y_val, y_pred)
            sub_tracking_r2score.append(r2score)
            sub_tracking_msescore.append(msescore)
            sub_tracking_maescore.append(maescore)


        method=str(i)+" degree "+ str(sub_tracking_r2score.index(max(sub_tracking_r2score))+2)

        gen_tracking_r2score.append(max(sub_tracking_r2score))
        gen_tracking_msescore.append(sub_tracking_msescore[sub_tracking_r2score.index(max(sub_tracking_r2score))])
        gen_tracking_maescore.append(sub_tracking_maescore[sub_tracking_r2score.index(max(sub_tracking_r2score))])
        gen_tracking_method.append(method)
        #print(sub_tracking_score)


    else:
        m_svr = SVR(kernel=i, gamma=g, epsilon=e, C=c).fit(X_train, y_train)
        y_pred = m_svr.predict(X_val)
        r2score = r2_score(y_val, y_pred)
        msescore = mean_squared_error(y_val, y_pred)
        maescore = mean_absolute_error(y_val, y_pred)
        gen_tracking_r2score.append(r2score)
        gen_tracking_msescore.append(msescore)
        gen_tracking_maescore.append(maescore)
        gen_tracking_method.append(i)

best_svr_score=max(gen_tracking_r2score)
best_svr_method=gen_tracking_method[gen_tracking_r2score.index(best_svr_score)]

print('The best SVR method using validation set is:'+str(best_svr_method)+", and the r square score is: "+str(best_svr_score))
print("- the mse score is: " +str(gen_tracking_msescore[gen_tracking_r2score.index(best_svr_score)]))
print("- the mse score is: " +str(gen_tracking_maescore[gen_tracking_r2score.index(best_svr_score)]))


#5. Testing using the best method based on R2 score:

print("")
print("Now, Running the model on the best method based on R2 score:")
score_array = [best_tree_score,best_rf_score,best_svr_score]
method_array = ['Decision Tree', 'Random Forest', 'Support Vector']

best_method_index = score_array.index(max(score_array))+1 #just of set the array indexing
best_method= method_array[best_method_index-1]
if best_method_index == 1:
    m_tree = DecisionTreeRegressor(criterion=best_tree_criterion, random_state=0).fit(X_train, y_train)
    y_pred = m_tree.predict(X_test)

elif best_method_index ==2:
    m_rf = RandomForestRegressor(n_estimators=best_num_estimator, criterion=best_rf_criterion, random_state=0, ).fit(
        X_train, y_train)
    y_pred = m_rf.predict(X_test)

elif best_method_index == 3:
    if best_svr_method[:4] =='poly':
        m_svr = SVR(kernel=best_svr_method[:4], degree=int(best_svr_method[-1]), gamma=g, epsilon=e, C=c).fit(X_train, y_train)
    else:
        m_svr = SVR(kernel=i, gamma=g, epsilon=e, C=c).fit(X_train, y_train)
    y_pred = m_svr.predict(X_test)

final_model_score = r2_score(y_test, y_pred)
print('-----------------------------------------------------------------------')
print('The Final Score:')
print ("The",best_method,"returns a r square score of:", final_model_score, 'with the testing data set')
print('-------------------------------the end---------------------------------')



#6. supporting visual - y pred vs y test
plt.figure()

sns.scatterplot(x=y_test,y=y_pred,label='predicted value',color='blue')
sns.lineplot(x=y_test,y=y_test,label='actual value',color='red')
plt.title('Testing Set: Actual Y vs Predicted Y')



#6.1 supporting visual = bin width optimization
n_bin=[]
score_val=[]
score_test=[]
target=[]
for n in range (2,31):
    # 1. Data Cleaning
    df = pd.read_csv('insurance.csv').dropna()
    # print(df.describe())

    # 2. Preprocessing - Encoding
    label = LabelEncoder()
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

    df['sex'] = label.fit_transform(df['sex'])
    df['smoker'] = label.fit_transform(df['smoker'])

    onehot_encode = onehot.fit_transform(df['region'].values.reshape(-1, 1))

    df_plot = df
    df_plot['region'] = label.fit_transform(df['region'])

    age_bin = []
    bmi_bin = []
    num_bin = n

    for k in range(0, num_bin + 1, 1):
        age_width = round((max(df_plot['age']) - min(df_plot['age'])) * k / num_bin + min(df_plot['age']), 2)
        age_bin.append(age_width)
        bmi_width = round((max(df_plot['bmi']) - min(df_plot['bmi'])) * k / num_bin + min(df_plot['bmi']), 2)
        bmi_bin.append(bmi_width)

    age_label = age_bin[1:]
    bmi_label = bmi_bin[1:]

    df_plot['age'] = pd.cut(df_plot['age'], bins=age_bin, labels=age_label, include_lowest=True).astype(float)
    df_plot['bmi'] = pd.cut(df_plot['bmi'], bins=bmi_bin, labels=bmi_label, include_lowest=True).astype(float)

    # 3 Splitting data set
    # x=pd.concat([df.drop(columns=['region','charges']),onehot_encode],axis=1)# Replace original data with one hot encoded data
    x = pd.concat([df_plot.drop(columns=['region', 'charges']), onehot_encode], axis=1)
    y = df['charges']
    X_train, X_combo, y_train, y_combo = train_test_split(x, y, train_size=0.8, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)

    # 3.1 Feature Scaling
    scale = MinMaxScaler()
    X_train = scale.fit_transform(X_train)
    X_val = scale.transform(X_val)
    X_test = scale.transform(X_test)

    # 4.2 Random Forest
    #criterion_input = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    criterion_input = ['absolute_error','friedman_mse']
    rf_r2score_val = -2
    rf_r2score_test=-2
    rf_criterion = "null"
    n_estimator = 0

    for i in criterion_input:

        for j in range(5, 25):
            m_rf = RandomForestRegressor(n_estimators=j, criterion=i, random_state=0 ).fit(X_train, y_train)
            y_pred_rf_val = m_rf.predict(X_val)
            y_pred_rf_test = m_rf.predict(X_test)
            r2score_val = r2_score(y_val, y_pred_rf_val)
            r2score_test = r2_score(y_test, y_pred_rf_test)


            if r2score_val > rf_r2score_val:
                rf_r2score_val   = r2score_val
                rf_r2score_test= r2score_test
                rf_criterion = i
                n_estimator = j

    best_rf_score = rf_r2score_val
    best_num_estimator = n_estimator
    best_rf_criterion = rf_criterion
    #print( best_rf_criterion, n, n_estimator, "and the r square score is: ", best_rf_score, rf_r2score_test)

    n_bin.append(n)
    score_val.append(rf_r2score_val)
    score_test.append(rf_r2score_test)
    target.append(0.9)

plt.figure()
sns.lineplot(x=n_bin,y=score_val,label = 'validating score',color='blue',marker='o')
sns.lineplot(x=n_bin,y=score_test,label = 'testing score',color='red',marker='s')
sns.lineplot(x=n_bin,y=target,label = 'testing score target',color='pink',linestyle='--')
plt.title('Bin Width Selection')
#plt.suptitle('Train-Valid-Test split. (Note: the Criterion and number of estimators may various between iterations)')
plt.xlabel('Number of Bins for Age and BMI')
plt.ylabel('Max Random Forest R2 Scores for each bin size selection')
plt.show()