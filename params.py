#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:21:50 2017

@author: matthew
"""

def standard():
    params = {}

    # Build Parameters
    params['conv_filters'] = 32
    params['nb_pool'] = 2
    params['nb_conv'] = 2
    params['optimizer'] = 'nadam'
    params['loss'] = 'categorical_crossentropy'

    # Fit Parameters
    params['epoch'] = 4
    params['batch_size'] = 128

    # Dense Activation
    params['activate_1'] = 'relu'
    return params
    