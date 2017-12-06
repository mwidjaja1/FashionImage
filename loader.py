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
import models_res
import params

np.set_printoptions(precision=2)

def parse_args(inargs=None):
    """ Parses input arguments """
    parser = ArgumentParser("./loader.py")
    standard_path = os.path.dirname(os.path.realpath(__file__))

    iargs = parser.add_argument_group('Input Files/Data')
    iargs.add_argument('--csv_file',
                       default=os.path.join(standard_path, 'data.csv'),
                       help='Path to CSV File')
    iargs.add_argument('--model', default='cnn',
                       help='Select: cnn (default), rnn, neural, vgg')

    oargs = parser.add_argument_group('Output Files/Data')
    oargs.add_argument('--out',
                       default=os.path.join(standard_path, 'Run'),
                       help='Path to save output files')
    oargs.add_argument('--plot',
                       default=False, action='store_true',
                       help='Enable this to plot the weights from Keras')

    if not inargs:
        args = parser.parse_args()
    else:
        args = parser.parse_args(inargs)
    return args


def flatten_data(args, x_train, x_test, y_train, y_test):
    """ Flattens data into a one dimension Numpy Array
    """
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if args.model != 'rnn':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
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
    # x_train = (60000, 28, 28) ==> (28, 28)
    # y_train = (60000,)
    # x_test = 
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, y_train, x_test, y_test = flatten_data(args, x_train, x_test, y_train, y_test)

    # Visualizes Data
    #featureselect.plot_features(data_df)

    # Creates output directory
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # Creates range to loop filter between
    change = 'gpu_res_epoch'
    range = [30, 40, 50, 60, 70]
    history_dict = {x: {'loss': 0.0, 'acc': 0.0} for x in range}

    # Runs Model
    for new in range:
        history_dict[new] = {'loss': [], 'acc': []}
        for loop in [1, 2]:
            print('Creating Model with the {} {}'.format(new, change))
            model_params = params.standard()
            model_params['epoch'] = new

            if args.model == 'rnn':
                model = models.basic_rnn(model_params, x_train.shape)
            elif args.model == 'neural':
                model = models.basic_neural(model_params, x_train.shape)
            elif args.model == 'double':
                model = models.double_cnn(model_params, x_train.shape)
            elif args.model == 'vgg':
                model = models.vgg(model_params, x_train.shape)
            elif args.model == 'res':
                model = models_res.main(model_params, x_train.shape)
            else:
                model = models.basic_cnn(model_params, x_train.shape)

            y_pred, metrics = models.fit_model(model, model_params, x_train, y_train,
                                               x_test, y_test, plot=args.plot)

            # Adds Data to Trends
            history_dict[new]['loss'].append(metrics['loss'])
            history_dict[new]['acc'].append(metrics['acc'])

        # Calculates Average
        history_dict[new]['loss'] = np.mean(history_dict[new]['loss'])
        history_dict[new]['acc'] = np.mean(history_dict[new]['acc'])

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
        class_values = list(classes.values())
        title = "{} (Loss {} & Acc {})".format(new, metrics['loss'], metrics['acc'])
        conf_png = '{}/{}_{}.png'.format(args.out, new, change)
        plot.conf_matrix(y_test, y_pred, class_values, out=conf_png, title=title)

    # Plots Accuracy & Loss Trends
    trends_png = '{}/{}.png'.format(args.out, change)
    plot.dict_trends(history_dict, xlabel=change, out=trends_png)

    return x_train, y_train, x_test, y_test, y_pred


if __name__ == "__main__":
    ARGS = parse_args()
    x_train, y_train, x_test, y_test, y_pred = main(ARGS)
    #models.random_forest(data_df)