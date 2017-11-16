<<<<<<< HEAD
from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
import numpy as np
from sklearn import discriminant_analysis, linear_model, model_selection, naive_bayes, neighbors, svm, tree
import matplotlib.pyplot as plt


def scikit_validator(in_train, out_train, out=None):
    """ Tests Scikit Learn models against training data

        Conclusions was that Logistic Regression (Mean 0.949 & SD 0.033) and
        LinearDiscriminantAnalysis (Mean 0.956 & SD 0.033) were the best. LDA
        is consistently high while LR is most likely to perform dead on.

    """
    # Spot Check Algorithms
    models = []
    models.append(('LR', linear_model.LogisticRegression()))
    models.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
    models.append(('QDA', discriminant_analysis.QuadraticDiscriminantAnalysis()))
    models.append(('KNN', neighbors.KNeighborsClassifier()))
    models.append(('TREE', tree.DecisionTreeClassifier()))
    models.append(('NB', naive_bayes.GaussianNB()))
    models.append(('SVM', svm.SVC()))

    # Evaluate Model
    sk_summary = {}
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        results = model_selection.cross_val_score(model, in_train, out_train,
                                                  cv=kfold, scoring='accuracy')
        sk_summary[name] = results
        print('{}: {} ({})'.format(name, results.mean(), results.std()))

    # Plots Results
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Algorithm Comparison')
    plt.boxplot([sk_summary[x] for x in sk_summary])
    ax.set_xticklabels([x for x in sk_summary])
    if out:
        plt.savefig('{}/machinelearning.png'.format(out))
    else:
        plt.show()

=======
def basic_neural(model_params, shape):
    """ Builds basic neural network model """
    from keras.layers import Dense, Flatten, InputLayer
    from keras.layers.normalization import BatchNormalization
    from keras.models import Sequential
>>>>>>> 41cbd59cc760fc5840079f5c1ff40089f94d50e6

    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def basic_cnn(model_params, shape):
    """ Builds basic Convolutional neural network model """
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
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def basic_rnn(model_params, shape):
    """ Builds basic Recurrent neural network model """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling1D, SimpleRNN
    from keras.layers.normalization import BatchNormalization
    from keras.models import Sequential

<<<<<<< HEAD
def build_intro_model(model_params, shape):
    """ Builds basic neural network model """
    model = Sequential()

=======
    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2])))
    model.add(BatchNormalization())

    model.add(SimpleRNN(model_params['conv_filters']))
    #model.add(MaxPool2D(padding='same'))
    #model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(0.1))
    #model.add(AveragePooling2D(padding='same'))
    #model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def double_cnn(model_params, shape):
    """ Builds basic Convolutional neural network model """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Sequential

    model = Sequential()

>>>>>>> 41cbd59cc760fc5840079f5c1ff40089f94d50e6
    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(model_params['conv_filters'],
                     (model_params['nb_pool'], model_params['nb_conv']),
                     padding='same'))
<<<<<<< HEAD
    model.add(MaxPool2D(padding='same'))
=======
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(model_params['conv_filters'],
                     (model_params['nb_pool'], model_params['nb_conv']),
                     padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(model_params['dropout']))
>>>>>>> 41cbd59cc760fc5840079f5c1ff40089f94d50e6
    model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    model.add(Dense(28, activation='softmax'))
<<<<<<< HEAD
 
=======

>>>>>>> 41cbd59cc760fc5840079f5c1ff40089f94d50e6
    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


<<<<<<< HEAD
def fit_intro_model(model, model_params, x_train, y_train, x_test, y_test):
    """ Fits neural network """
    # Fits Model
    model.fit(x_train, y_train, epochs=model_params['epoch'],
              batch_size=model_params['batch_size'], verbose=1)
=======
def fit_precomputed_models(model_params, x_test, y_test):
    """ Uses precomputed Keras models """
    from keras.applications.resnet50 import ResNet50
    #from keras.applications.resnet50 import preprocess_input, decode_predictions
>>>>>>> 41cbd59cc760fc5840079f5c1ff40089f94d50e6

    # Resnet 50
    rn50 = ResNet50(weights='imagenet')
    y_test_predict_rn50 = rn50.predict(x_test)
    y_test_predict_rn50_rounded = y_test_predict_rn50.round()

    # Calculates Metrics
    metrics = model.evaluate(x_test, y_test, batch_size=model_params['batch_size'])
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
    metrics['loss'], metrics['acc'] = model.evaluate(x_test, y_test,
    												 batch_size=model_params['batch_size'])
    print('\nAccuracy {} & Loss {}\n'.format(metrics['acc'], metrics['loss']))

    return y_pred_rounded, metrics