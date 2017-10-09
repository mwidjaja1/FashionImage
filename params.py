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
    params['nb_pool'] = 2
    params['nb_conv'] = 2
    params['optimizer'] = 'adam'
    params['loss'] = 'categorical_crossentropy'

    # Fit Parameters
    params['epoch'] = 1
    params['batch_size'] = 128
    return params
    