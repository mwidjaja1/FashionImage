# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from argparse import ArgumentParser
from keras.datasets import fashion_mnist
from keras.utils import np_utils
import numpy as np
import os
import plot

#import featureselect
import models
import params


def parse_args(inargs=None):
    """ Parses input arguments """
    parser = ArgumentParser("./loader.py")
    standard_path = os.path.dirname(os.path.realpath(__file__))

    iargs = parser.add_argument_group('Input Files/Data')
    iargs.add_argument('--csv_file',
                       default=os.path.join(standard_path, 'data.csv'),
                       help='Path to CSV File')

    oargs = parser.add_argument_group('Output Files/Data')
    oargs.add_argument('--out',
                       default=os.path.join(standard_path, 'Run'),
                       help='Path to save output files')

    if not inargs:
        args = parser.parse_args()
    else:
        args = parser.parse_args(inargs)
    return args


def flatten_data(x_train, x_test, y_train, y_test):
    """ Flattens data into a one dimension Numpy Array
    """
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    y_train = np_utils.to_categorical(y_train, 28)
    y_test = np_utils.to_categorical(y_test, 28)
    return x_train, y_train, x_test, y_test


def np_to_csv(out_dir, out_file, data):
    """ Saves Numpy array as CSV File """
    try:
        out_path = '{}/{}'.format(out_dir, out_file)
        with open(out_path, 'wb') as out_handle:
            np.savetxt(out_handle, data, delimiter=',')
            print("INFO: Saved {}".format(out_path))
    except Exception:
        try:
            with open(out_path, 'w') as out_handle:
                np.savetxt(out_handle, data, delimiter=',')
                print("WARNING: {} needed to workaround save".format(out_file))
        except Exception as err:
            print("ERROR: Could not save {}".format(out_file))
            print(err)


def main(args):
    # Loads CSV File
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, y_train, x_test, y_test = flatten_data(x_train, x_test, y_train, y_test)

    # Visualizes Data
    #featureselect.plot_features(data_df)

    # Creates output directory
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # Runs Model
    model_params = params.standard()
    mach_learn = models.scikit_validator(x_train, y_train, args.out)
    model = models.build_intro_model(model_params, x_train.shape)
    y_test_predict, metrics = models.fit_intro_model(model, model_params,
                                                     x_train, y_train,
                                                     x_test, y_test)

    # Plots Confusion Matrix
    classes = {0: 'T-Shirt/top',
               1: 'Trouser',
               2: 'Pullover',
               3: 'Dress',
               4: 'Coat',
               5: 'Sandal',
               6: 'Shirt',
               7: 'Sneaker',
               8: 'Bag',
               9: 'Ankle boot'}
    title = 'Confusion Matrix | Loss {} & Acc {}'.format(*metrics)
    class_values = list(classes.values())
    plot.conf_matrix(y_test, y_test_predict, class_values, title=title,
                     out=args.out)

    # Saves Numpy Array as CSV File
    if args.out:
        np_to_csv(args.out, 'x_train.csv', x_train)
        np_to_csv(args.out, 'y_train.csv', y_train)
        np_to_csv(args.out, 'x_test.csv', x_test)
        np_to_csv(args.out, 'y_test.csv', y_test)
        np_to_csv(args.out, 'y_test_predict.csv', y_test_predict)
        np_to_csv(args.out, 'metrics.csv', metrics)

    return x_train, y_train, x_test, y_test, y_test_predict, metrics






if __name__ == "__main__":
    ARGS = parse_args()
    x_train, y_train, x_test, y_test, y_test_predict, metrics = main(ARGS)
    #models.random_forest(data_df)