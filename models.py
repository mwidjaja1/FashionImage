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



def build_intro_model(model_params, shape):
    """ Builds basic neural network model """
    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    model.add(Conv2D(model_params['conv_filters'],
                     (model_params['nb_pool'], model_params['nb_conv']),
                     padding='same'))
    model.add(MaxPool2D(padding='same'))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(28, activation='softmax'))
 
    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    return model


def fit_intro_model(model, model_params, x_train, y_train, x_test, y_test):
    """ Fits neural network """
    # Fits Model
    model.fit(x_train, y_train, epochs=model_params['epoch'],
              batch_size=model_params['batch_size'], verbose=1)

    # Predicts Model
    y_test_predict = model.predict(x_test)
    y_test_predict_rounded = y_test_predict.round()

    # Scores Model
    metrics = model.evaluate(x_test, y_test, batch_size=128)
    metrics_rounded = np.round(metrics, 3)
    print("\n{}: {} and {}: {}".format(model.metrics_names[0], metrics[0], 
                                       model.metrics_names[1], metrics[1]))
    return y_test_predict_rounded, metrics_rounded