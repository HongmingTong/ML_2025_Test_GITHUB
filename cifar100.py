from keras.datasets import cifar100
from tensorflow.python.keras.legacy_tf_layers.core import dropout

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

from keras.utils import to_categorical
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)


from keras.models import Sequential
model = Sequential()

from keras.layers import Conv2D
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), input_shape=(30, 30, 3), activation='relu'))

from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=(2, 2)))


from keras.layers import Dropout
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), input_shape=(15, 15, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

from keras.layers import Flatten
model.add(Flatten())

from keras.layers import Dense
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=100, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100,batch_size=32)

loss, accuracy = model.evaluate(x_test, y_test)
print("The model loss is: ", loss)
print("The model accuracy is: ", accuracy)