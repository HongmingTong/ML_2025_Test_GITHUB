import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale, MinMaxScaler
from tensorflow.python.keras.legacy_tf_layers.core import dropout

# load and encode data
df = pd.read_csv('data_refined.csv')  ## already cleaned
xbreast = df.drop(columns=['diagnosis'])
ybreast = df['diagnosis']

dfins = pd.read_csv('insurance.csv').dropna()
label = LabelEncoder()
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
dfins['sex'] = label.fit_transform(dfins['sex'])
dfins['smoker'] = label.fit_transform(dfins['smoker'])
onehot_encode = onehot.fit_transform(dfins['region'].values.reshape(-1, 1))
xins = pd.concat([dfins.drop(columns=['region', 'charges']), onehot_encode], axis=1)
yins = dfins['charges'].values


#Split and scale data
from sklearn.model_selection import train_test_split
X_train, X_combo, y_train, y_combo = train_test_split(xbreast, ybreast, train_size=0.8,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=0.5, random_state=0)


X_train2, X_combo2, y_train2, y_combo2 = train_test_split(xins, yins, train_size=0.8,random_state=0)
X_test2, X_val2, y_test2, y_val2 = train_test_split(X_combo2, y_combo2, test_size=0.5, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.transform(X_test2)



##############################################################################

#MLP Classification Training on brest cancer data
from sklearn.neural_network import MLPClassifier
m=MLPClassifier(hidden_layer_sizes=(1000,10),activation='relu',solver='adam',batch_size=20,max_iter=1000,random_state=0, early_stopping=True)
m.fit(X_train,y_train)
scoreMLP= m.score(X_val,y_val)
scoreMLPs= m.score(X_test,y_test)
#print("The accuracy score using MLP Classifier is:", scoreMLP,scoreMLPs)

# Create a chart to show how the loss improves up to a certain # of epoch's
plt.figure(figsize=(8, 2))
plt.title("ANN Model Loss Evolution")
plt.xlabel("Epochs"), plt.ylabel("Loss")
plt.plot(m.loss_curve_)
#plt.show()



#Kera Classification Training on brest cancer data
from keras.models import Sequential

m11 = Sequential()
from keras.layers import Dense
#m.add(Input(shape=(11,)))
m11.add(Dense(units=100, activation='relu', kernel_initializer='uniform'))
m11.add(Dense(units=1, activation='relu', kernel_initializer='uniform'))
m11.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m11.fit(X_train, y_train, batch_size=10, epochs=100)
y_pred = m11.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
if accuracy_score(y_test, y_pred) >= scoreMLPs:
    model = 'keras'
else:
    model = 'MLP'
print('Breast Cancer ML Training')
print('---')
print("The accuracy score using MLP Classifier is:", scoreMLP)
print('The accuracy score using Keras is:', accuracy_score(y_test, y_pred))
print('The Confusion Matrix using Kera is:')
print(confusion_matrix)
print('Both method has simular accuracy but the accuracy score is slightly higher using ', model)

###########################################################################


#MLP Classification Training on Insurance data
from sklearn.neural_network import MLPRegressor
m2=MLPRegressor(hidden_layer_sizes=(1070,10),activation='relu',solver='adam',batch_size=20,max_iter=1000,random_state=0, early_stopping=True)
m2.fit(X_train2,y_train2)
scoreMLPins= m2.score(X_val2,y_val2)
scoreMLPins2= m2.score(X_test2,y_test2)
#print("The accuracy score using MLP Classifier is:", scoreMLPins2)



#Kera Classification Training on Insurance data
from keras.models import Sequential

m2 = Sequential()
from keras.layers import Dense, Dropout

m2.add(Dense(units=1000, activation='relu'))
m2.add(Dropout(0.05))
m2.add(Dense(units=100,activation='relu'))
m2.add(Dense(units=1, activation='relu'))
m2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
m2.fit(X_train2, y_train2, batch_size=3 ,epochs=100)
y_predins =m2.predict(X_test2)

import numpy as np
SSE = 0
SSR = 0

length = len(y_test2)

for i in range(0, length):
    SSE = SSE + (y_predins[i] - y_test2[i])*(y_predins[i] - y_test2[i]) # Compares prediction to actual
    SSR = SSR + np.square(y_predins[i] - y_test2.mean())  # Compares prediction to simplest model
SST = SSR + SSE  # Total sum of squares represents total variability of y_test
r2 = np.float64(SSR / SST)  # Calculate the coefficient of determination (R-Squared value)

print(r2)
print('Insurance ANN Training')
print('---')
print("The accuracy score using MLP Classifier is:", scoreMLPins2)
print('The accuracy score using Keras is:',r2)
if r2>=scoreMLPins2:
    model = 'keras'
else:
    model = 'MLP'
print('Both method has simular accuracy but the accuracy score is slightly higher using ',model)