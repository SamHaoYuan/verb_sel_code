# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:43:00 2017

@author: Sam
"""


import pandas as pd
# import numpy as np

rising_WSJ = pd.read_csv('../UserEvaluation2/verbs_list/WSJ_rising_2_verbs')
rising_Chinese = pd.read_csv('../UserEvaluation2/verbs_list/Chinese_rising_2_verbs')
rising_Reuters = pd.read_csv('../UserEvaluation2/verbs_list/Reuters_rising_2_verbs')
falling_WSJ = pd.read_csv('../UserEvaluation2/verbs_list/WSJ_falling_2_verbs')
falling_Reuters = pd.read_csv('../UserEvaluation2/verbs_list/Reuters_falling_2_verbs')
falling_Chinese = pd.read_csv('../UserEvaluation2/verbs_list/Chinese_falling_2_verbs')


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

diversity('WSJ', 'rising', rising_WSJ)
diversity('Reuters', 'rising', rising_Reuters)
diversity('Chinese', 'rising', rising_Chinese)
diversity('WSJ', 'falling', falling_WSJ)
diversity('Reuters', 'falling', falling_Reuters)
diversity('Chinese', 'falling', falling_Chinese)
