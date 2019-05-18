import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

np.random.seed(9)

dataset = np.loadtxt('MelbourneHousing.csv', delimiter=',', skiprows=1, )
X = dataset[:, 0:9]
Y = dataset[:, 9]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=9)

model = Sequential()
model.add(Dense(12, input_dim=9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(3, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=500, batch_size=10)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
