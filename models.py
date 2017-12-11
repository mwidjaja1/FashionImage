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

def basic_neural(model_params, shape):
    """ Builds basic neural network model """
    from keras.layers import Dense, Flatten, InputLayer
    from keras.layers.normalization import BatchNormalization
    from keras.models import Sequential

    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Dropout(model_params['dropout']))
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
                     model_params['kernel_size'],
                     strides=model_params['kernel_stride'],
                     padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(model_params['dropout']))
    model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    #model.add(Dropout(model_params['dropout']))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def triple_cnn(model_params, shape):
    """ Builds a triple Convolutional neural network model """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling2D, ZeroPadding2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Sequential

    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(model_params['conv_filters'],
                     model_params['kernel_size'],
                     strides=model_params['kernel_stride'],
                     activation=model_params['cnn_activation'],
                     padding='same'))

    model.add(Conv2D(model_params['conv_filters'],
                     model_params['kernel_size'],
                     strides=model_params['kernel_stride'],
                     activation=model_params['cnn_activation'],
                     padding='same'))

    model.add(Conv2D(model_params['conv_filters'],
                     model_params['kernel_size'],
                     strides=model_params['kernel_stride'],
                     activation=model_params['cnn_activation'],
                     padding='same'))

    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def vgg_seq(model_params, shape):
    """ Builds a VGG-19 like model """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling2D, ZeroPadding2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Sequential

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(model_params['vgg_filters_1'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_1'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(model_params['vgg_filters_2'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_2'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(MaxPooling2D(padding='same'))
   
    model.add(Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                     activation=model_params['vgg_activation'], padding='same'))
    model.add(ZeroPadding2D())
    model.add(MaxPooling2D(padding='same'))

    model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['vgg_activation']))
    model.add(Dense(model_params['dense_1'], activation=model_params['vgg_activation']))
    #model.add(Dropout(0.5))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def vgg19(model_params, shape):
    """ Builds a VGG-19 like model """
    from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Model

    print(shape)
    input = Input(shape=(shape[1], shape[2], shape[3]))

    tensor = Conv2D(model_params['vgg_filters_1'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(input)
    tensor = Conv2D(model_params['vgg_filters_1'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)

    tensor = Conv2D(model_params['vgg_filters_2'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_2'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)

    tensor = Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)
   
    tensor = Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)

    tensor = Flatten()(tensor)

    tensor = Dense(model_params['dense_1'], activation=model_params['vgg_activation'])(tensor)
    tensor = Dense(model_params['dense_1'], activation=model_params['vgg_activation'])(tensor)
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(28, activation='softmax')(tensor)

    model_compile = Model(input, tensor)
    model_compile.compile(loss=model_params['loss'],
                          optimizer=model_params['optimizer'],
                          metrics=['accuracy'])

    print(model_compile.summary())
    return model_compile


def vgg16(model_params, shape):
    """ Builds a VGG-16 like model """
    from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Model

    print(shape)
    input = Input(shape=(shape[1], shape[2], shape[3]))

    tensor = Conv2D(model_params['vgg_filters_1'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(input)
    tensor = Conv2D(model_params['vgg_filters_1'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)

    tensor = Conv2D(model_params['vgg_filters_2'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_2'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)

    tensor = Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_3'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)
   
    tensor = Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = Conv2D(model_params['vgg_filters_4'], model_params['vgg_kernel'],
                    activation=model_params['vgg_activation'], padding='same')(tensor)
    tensor = MaxPooling2D(padding='same')(tensor)

    tensor = Flatten()(tensor)

    tensor = Dense(model_params['dense_1'], activation=model_params['vgg_activation'])(tensor)
    tensor = Dense(model_params['dense_1'], activation=model_params['vgg_activation'])(tensor)
    tensor = Dense(28, activation='softmax')(tensor)

    model_compile = Model(input, tensor)
    model_compile.compile(loss=model_params['loss'],
                          optimizer=model_params['optimizer'],
                          metrics=['accuracy'])

    print(model_compile.summary())
    return model_compile


def fit_model(model, model_params, x_train, y_train, x_test, y_test, plot=False):
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

    # Plots Weight if Requested
    if plot:
        from keras.utils import plot_model
        plot_model(model, to_file='weights.png')

    return y_pred_rounded, metrics
