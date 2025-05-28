import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('text_emotion.csv')  ## already cleaned
df = df.dropna()
x = df.iloc[:, 3].str.lower().to_list()
y = df.iloc[:, 1].str.lower().to_list()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
y = label.fit_transform(y).tolist()

texts = x
labels = y

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
vocabulary_size = len(tokenizer.word_index) + 1

sequence_length = max([len(sequence) for sequence in sequences])
print(sequence_length)
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=sequence_length,padding='pre')


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(padded_sequences,np.array(labels) , test_size=0.3, random_state=0)


########################################################

#########################################################

from keras.models import Sequential
model = Sequential()

from keras.layers import Embedding
model.add(Embedding(input_dim=vocabulary_size, output_dim=10, input_length=sequence_length))

from keras.layers import LSTM
model.add(LSTM(128))
#model.add(LSTM(64))

from keras.layers import Dense
model.add(Dense(units=100, activation='relu'))

from keras.layers import Dropout
model.add(Dropout(0.5))


from keras.layers import Dense
model.add(Dense(units=5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=256)


print('The Vocabulary size is: ',vocabulary_size)
print('The Maximum sequence length is: ',sequence_length)

loss_train, accuracy_train = model.evaluate(x_train,y_train)
print('The accuracy using the training set is:', accuracy_train)

loss_test, accuracy_test = model.evaluate(x_test,y_test)
print('The accuracy using the testing set is:', accuracy_test)