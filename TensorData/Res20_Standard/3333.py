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