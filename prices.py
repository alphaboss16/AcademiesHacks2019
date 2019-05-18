import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


np.random.seed(9)

dataset = np.loadtxt('MelbourneHousing.csv', delimiter=',', skiprows=1, )
X = dataset[:, 0:9]
Y = dataset[:, 9]


(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=9)

from keras import regularizers

model = Sequential()
model.add(Dense(10, input_dim=9, activation='tanh', init='uniform'))
model.add(Dense(6, activation='tanh', init='uniform', activity_regularizer=regularizers.l2(0.001)))
model.add(Dense(3, activation='relu', init='uniform', activity_regularizer=regularizers.l2(0.001)))
model.add(Dense(1, init='uniform'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=50, batch_size=10)

filepath = "Model/{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# evaluate the model

scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
