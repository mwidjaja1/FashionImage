#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:21:50 2017

@author: matthew
"""

def standard():
    params = {}

    # Build Parameters
    params['conv_filters'] = 24
    params['kernel_size'] = 2
    # params['optimizer'] = 'nadam'  # Used for CNN
    params['optimizer'] = 'adam'  # Used for Neural
    params['loss'] = 'categorical_crossentropy'

    # Fit Parameters
    params['epoch'] = 12
    params['dropout'] = 0.1
    params['batch_size'] = 128

    # Dense Activation
    params['dense_1'] = 120
    # params['activate_1'] = 'relu'  # Used for CNN
    params['activate_1'] = 'sigmoid'  # Used for Neural
    return params
    