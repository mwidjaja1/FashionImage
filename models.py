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

    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2])))
    model.add(BatchNormalization())

    model.add(SimpleRNN(model_params['conv_filters']))
    #model.add(MaxPool2D(padding='same'))
    #model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(model_params['dropout']))
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

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(model_params['conv_filters'],
                     model_params['kernel_size'],
                     strides=model_params['kernel_stride'],
                     padding='same'))
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(10,
                     2,
                     padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(model_params['dropout']))

    model.add(Flatten())

    model.add(Dense(model_params['dense_1'], activation=model_params['activate_1']))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def pre_compiled_model(model_params):
	from keras.applications.resnet50 import ResNet50
	from keras.preprocessing import image
	from keras.applications.resnet50 import preprocess_input, decode_predictions
	import numpy as np

	model = ResNet50(weights='imagenet')

	img_path = 'elephant.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)


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