from keras.layers import Activation, AveragePooling2D, Add, Dense, Flatten, Input, MaxPool2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Model


def standard_conv(model, filter, kernel_size, activation, shortcut=None):
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

    model = Conv2D(filter, kernel_size, padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation(activation)(model)

    model = Conv2D(filter, kernel_size, padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation(activation)(model)

    model = Conv2D(filter*4, kernel_size, padding='same')(model)
    model = BatchNormalization()(model)

    if shortcut != None:
        shortcut_model = Conv2D(filter*4, kernel_size, padding='same')(shortcut)
        shortcut_model = BatchNormalization()(shortcut_model)
        model = Add()([model, shortcut_model])

    model = Activation(activation)(model)
    return model


def main(model_params, shape):
    """ Builds a ResNet 18 Convolutional Neural Network
        If I uncomment out the block comments, I'd get ResNet 34.
    """
    eps = 1.1e-5

    input = Input(shape=(shape[1], shape[2], shape[3]))

    # First Series (Orange)
    model = ZeroPadding2D((3, 3))(input)
    model = Conv2D(model_params['res_filters_1'],
                   model_params['res_kernel_size'],
                   padding='same')(model)
    model = BatchNormalization(epsilon=eps)(model)
    model = Activation(model_params['cnn_activation'])(model)
    model = MaxPool2D((3, 3), strides=2)(model)
    
    # Second Series (Purple)
    model = standard_conv(model,
                          model_params['res_filters_1'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'],
                          shortcut=model)
    model = standard_conv(model,
                         model_params['res_filters_1'],
                         model_params['res_kernel_size'],
                         model_params['cnn_activation'])
    model = standard_conv(model,
                          model_params['res_filters_1'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])

    
    # Third Series (Green)
    model = standard_conv(model,
                          model_params['res_filters_3'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'],
                          shortcut=model)
    model = standard_conv(model,
                          model_params['res_filters_3'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    model = standard_conv(model,
                          model_params['res_filters_3'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    """
    model = standard_conv(model,
                          model_params['res_filters_3'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    """

    # Four Series (Red)
    model = standard_conv(model,
                          model_params['res_filters_4'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'],
                          shortcut=model)
    model = standard_conv(model,
                          model_params['res_filters_4'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    model = standard_conv(model,
                         model_params['res_filters_4'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    """
    model = standard_conv(model,
                          model_params['res_filters_4'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    model = standard_conv(model,
                          model_params['res_filters_4'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    model = standard_conv(model,
                          model_params['res_filters_4'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])
    """

    # Five Series (Purple)
    model = standard_conv(model,
                          model_params['res_filters_5'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'],
                          shortcut=model)
    model = standard_conv(model,
                          model_params['res_filters_5'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])    
    model = standard_conv(model,
                          model_params['res_filters_5'],
                          model_params['res_kernel_size'],
                          model_params['cnn_activation'])

    # Final Neural Layer
    model_neural = AveragePooling2D()(model)
    model_neural = Flatten()(model_neural)
    #model_neural = Dense(model_params['res_dense'],
    #                    activation=model_params['res_activate'])(model_neural)
    model_neural = Dense(28, activation='softmax')(model_neural)

    # Compiles Model
    model_compile = Model(input, model_neural)
    model_compile.compile(loss=model_params['loss'],
                          optimizer=model_params['optimizer'],
                          metrics=['accuracy'])

    print(model_compile.summary())
    return model_compile