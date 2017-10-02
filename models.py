from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
import numpy as np


def build_intro_model(model_params, shape):
    model = Sequential()

    # number of convolutional filters to use
    nb_filters = 1
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 2

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(nb_filters, (nb_pool, nb_conv), padding='same'))
    model.add(MaxPool2D(padding='same'))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(28, activation='softmax'))
 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def fit_intro_model(model, x_train, y_train, x_test, y_test):
    # Fits Model
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)

    # Predicts Model
    y_test_predict = model.predict(x_test)
    y_test_predict_rounded = y_test_predict.round()

    # Scores Model
    metrics = model.evaluate(x_test, y_test, batch_size=128)
    metrics_rounded = np.round(metrics, 3)
    print("\n{}: {} and {}: {}".format(model.metrics_names[0], metrics[0], 
                                       model.metrics_names[1], metrics[1]))
    return y_test_predict_rounded, metrics_rounded