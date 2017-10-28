def build_theano_model(model_params, shape):
    """ Builds basic neural network model using Theano """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Sequential

    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(model_params['conv_filters'],
                     (model_params['nb_pool'], model_params['nb_conv']),
                     padding='same'))
    #model.add(MaxPool2D(padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.1))
    #model.add(AveragePooling2D(padding='same'))
    model.add(Flatten())

    model.add(Dense(128, activation=model_params['activate_1']))
    model.add(Dense(shape[2], activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def build_double_model(model_params, shape):
    """ Builds doubled layer neural network model """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Sequential

    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(model_params['conv_filters'],
                     (model_params['nb_pool']*2, model_params['nb_conv']*2),
                     padding='same'))
    model.add(Conv2D(16, (model_params['nb_pool'], model_params['nb_conv']),
                     padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(128, activation=model_params['activate_1']))
    model.add(Dense(64, activation=model_params['activate_1']))
    model.add(Dense(shape[2], activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def fit_precomputed_models(model_params, x_test, y_test):
    """ Uses precomputed Keras models """
    from keras.applications.resnet50 import ResNet50
    #from keras.applications.resnet50 import preprocess_input, decode_predictions

    # Resnet 50
    rn50 = ResNet50(weights='imagenet')
    y_test_predict_rn50 = rn50.predict(x_test)
    y_test_predict_rn50_rounded = y_test_predict_rn50.round()

    # Calculates Metrics
    metrics = model.evaluate(x_test, y_test, batch_size=128)
    metrics_rounded = np.round(metrics, 3)
    print("\n{}: {} and {}: {}".format(model.metrics_names[0], metrics[0], 
                                       model.metrics_names[1], metrics[1]))
    return y_test_predict_rn50_rounded, metrics_rounded


def fit_model(model, model_params, x_train, y_train, x_test, y_test):
    """ Fits neural network """
    # Fits Model
    model.fit(x_train, y_train, epochs=model_params['epoch'],
              batch_size=model_params['batch_size'], verbose=1)

    # Predicts Model
    y_pred = model.predict(x_test)
    y_pred_rounded = y_pred.round()

    # Scores Model on Test Data
    metrics = {'acc': 0.0, 'loss': 0.0}
    metrics['loss'], metrics['acc'] = model.evaluate(x_test, y_test, batch_size=128)
    print('\nAccuracy {} & Loss {}\n'.format(metrics['acc'], metrics['loss']))

    return y_pred_rounded, metrics