import pandas as pd

from MNIST import accuracy

df = pd.read_csv('IMDB Dataset.csv')  ## already cleaned
df = df.dropna()
x=df.iloc[:,0].str.lower().to_list()
y=df.iloc[:,1].str.lower().to_list()

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
vocabulary_size = len(tokenizer.word_index) + 1
sequence_length = len(x)


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
label= LabelEncoder()
y=label.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from keras.models import Sequential
model = Sequential()

from keras.layers import Embedding
model.add(Embedding(vocabulary_size, 100, input_length=sequence_length - 1))

from keras.layers import Dropout
model.add(Dropout(0.1))

from keras.layers import Dense
model.add(Dense(units=vocabulary_size, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

loss_train, accuracy_train = model.evaluate(x_train,y_train)
print('The accuracy using the training set is:', accuracy_train)

loss_test, accuracy_test = model.evaluate(x_test,y_test)
print('The accuracy using the testing set is:', accuracy_test)