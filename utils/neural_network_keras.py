import os
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.optimizers import SGD, Adam
#from functions import read_csv
import numpy as np

def create_model_api():
    # input layer
    inputs = Input(shape=(10,), dtype='float32')

    # layer 1
    X = Dense(20)(inputs) #problem
    X = Activation('relu')(X)

    # layer 2
    X = Dense(15)(X)
    X = Activation('relu')(X)

    # layer 3
    X = Dense(6)(X)
    X = Activation('relu')(X)

    # layer 4
    X = Dense(3)(X)
    X = Activation('relu')(X)

    # output layer
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs, X)
    model.summary()

    return model

def create_model_sequencial():
    model = Sequential()

    model.add(Dense(20, input_dim=10, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    return model

def train_model(X_train, Y_train, X_test, Y_test, model):
    """ Trains the model """

    print(X_train, Y_train)
    sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10) # shows loss and accuracy in addition

    print("model trained!")
    loss, acc = model.evaluate(X_test, Y_test)
    print("Test accuracy = ", acc)
    print("Test loss = ", loss)

    # save weights into a new filename
    model.save_weights('model_weights.h5py')
    print("model weights saved to /model_weights.h5py")

    return model

def predict(point, model):
    """ Predicts the points given as an argument. The argument is actually a list of points """

    model.load_weights('model_weights.h5py')


    return model.predict(point)
