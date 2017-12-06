from keras.layers import Activation, AveragePooling2D, Dense, Flatten, InputLayer, MaxPool2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Sequential


def standard_conv(model, filter, kernel_size, activation, shape=None):
    """ Creates a stack of three standard Convolutional Layers 
    
        Inputs:
        model: The Keras Model to add onto
        filter: The number of filters to use in the Convolutional Layers
        kernel_size: The kernel size to use in the Convolutional Layers
        activation: The Activation Function to use
        shape: The Keras Model from 'model'. Set this to use the Shortcut Layer.
                        [Default = None => Don't use the shortcut layer]
    """
    eps = 1.1e-5

    model.add(Conv2D(filter, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Conv2D(filter, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Conv2D(filter*4, kernel_size, padding='same'))
    model.add(BatchNormalization())

    if shape:
        model.add(Conv2D(filter*4, kernel_size, padding='same'))
        model.add(BatchNormalization())

    model.add(Activation(activation))
    return model


def main(model_params, shape):
    """ Builds a ResNet 34 Convolutional Neural Network """
    eps = 1.1e-5

    model = Sequential()

    model.add(InputLayer(input_shape=(shape[1], shape[2], shape[3])))
    model.add(BatchNormalization())

    # First Series (Orange)
    model.add(ZeroPadding2D((3, 3)))
    model.add(Conv2D(model_params['res_filter'],
                     model_params['res_kernel_1'],
                     padding='same'))
    model.add(BatchNormalization(epsilon=eps))
    model.add(Activation(model_params['cnn_activation']))
    model.add(MaxPool2D((3, 3)))
    
    # Second Series (Purple)
    model = standard_conv(model,
                          model_params['res_filter'],
                          model_params['res_kernel_1'],
                          model_params['cnn_activation'],
                          shape=shape)
    model = standard_conv(model,
                          model_params['res_filter'],
                          model_params['res_kernel_1'],
                          model_params['cnn_activation'])
    model = standard_conv(model,
                          model_params['res_filter'],
                          model_params['res_kernel_1'],
                          model_params['cnn_activation'])

    
    # Third Series (Green)
    model = standard_conv(model,
                          model_params['res_filter'],
                          model_params['res_kernel_3'],
                          model_params['cnn_activation'],
                          shape=shape)
    #for _ in range(1, 2):
    model = standard_conv(model,
                          model_params['res_filter'],
                          model_params['res_kernel_3'],
                          model_params['cnn_activation'])


    """
    # Four Series (Red)
    model = standard_conv(model, model_params['res_filter'],
                          model_params['res_kernel_4'], model_params['cnn_activation'],
                          shape=model)
    #for _ in range(1, 36):
    #for _ in range(1, 9):
    model = standard_conv(model, model_params['res_filter'],
                          model_params['res_kernel_4'],
                          model_params['cnn_activation'])
    """

    # Five Series (Purple)
    model = standard_conv(model,
                          model_params['res_filter'],
                          model_params['res_kernel_4'],  #model_params['res_kernel_5'],
                          model_params['cnn_activation'],
                          shape=model)
    #model = standard_conv(model, model_params['res_filter'],
    #                      model_params['res_kernel_5'], model_params['cnn_activation'])
    model = standard_conv(model,
                          model_params['res_filter'],
                          model_params['res_kernel_4'],  #model_params['res_kernel_5'],
                          model_params['cnn_activation'])    

    # Final Neural Layer
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(10000, activation=model_params['res_activate']))
    model.add(Dense(model_params['res_dense'], activation=model_params['res_activate']))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=model_params['loss'],
                  optimizer=model_params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model