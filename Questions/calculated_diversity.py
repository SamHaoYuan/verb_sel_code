# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:43:00 2017

@author: Sam
"""


import pandas as pd
# import numpy as np


rising_Chinese = pd.read_csv('WSJ_rising_verbs')


def inverse_simpson_index(X):
    r = X.value_counts()
    lambda_ = 0
    N = len(X)
    for verb in r.index:
        lambda_ += (float(r[verb]) / N)**2
    return float(1)/lambda_


def richness(X):
    return len(X.value_counts())


def diversity(dataset, direction, X):
    print('{0}, the richiness and inverse simpson inde of {1} verbs'.format(dataset, direction))
    print('baseline_verb: ')
    print(richness(X['baseline_verb']))
    print(inverse_simpson_index(X['baseline_verb']))
    print('proposed_verb: ')
    print(richness(X['proposed_verb']))
    print(inverse_simpson_index(X['proposed_verb']))
    print('-' * 37)


diversity('Chinese', 'rising', rising_Chinese)

