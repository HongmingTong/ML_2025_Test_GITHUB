from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)/255
x_test = x_test.reshape(10000, 28, 28, 1)/255


from keras.models import Sequential
model = Sequential()

from keras.layers import Conv2D
model.add(Conv2D(28, (3, 3), input_shape=(28, 28, 1), activation='relu'))

from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=(2, 2)))

from keras.layers import Flatten
model.add(Flatten())

from keras.layers import Dense
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

loss, accuracy = model.evaluate(x_test, y_test)
print("The model accuracy is: ", accuracy)