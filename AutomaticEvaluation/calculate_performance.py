# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:43:00 2017

@author: Sam
"""
import re
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import label_ranking_average_precision_score, make_scorer
from Model.VerbSelect import (
    VerbSelectBetaMRREstimator, VerbSelectKdeMRREstimator,
    VerbSelectBaselineMRREstimator, VerbSelectMLPMMREstimator
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import stats
import seaborn as sns

VERBS = {
    'rising': [
        'rise', 'increase', 'grow', 'climb', 'jump', 'surge', 'gain', 'soar', 'raise',
        'advance', 'boost'
    ],
    'falling': [
        'fall', 'decline', 'drop', 'slip', 'plunge', 'slide', 'lose', 'tumble', 'plummet',
        'ease', 'decrease', 'reduce', 'dip', 'shrink'
    ]
}

VERBS_Reuters = {
    'rising': [
        'rise', 'increase', 'grow', 'jump'
    ],
    'falling': [
        'fall', 'decline', 'drop', 'plunge'
    ]
}

VERBS_Chinese = {
    'rising': [
         '仅增长', '猛增', '仅上升', '只增长', '激增', '剧增', '仅增加',  '猛升', '只增加', '大幅上升', '大增',
         '飙升', '微升', '大幅增长'
    ],
    'falling': [
        '锐减', '大跌', '猛跌', '暴跌'
    ]
}
COLUMNS = ['verb', 'per']


def preprocess(path, Verbs):
    # VERBS = VERBS_Chinese
    # VERBS = VERBS_Reuters
    verb_df = pd.read_csv(path, usecols=COLUMNS)
    verb_df = verb_df[verb_df.verb.isin(Verbs['rising'] + Verbs['falling'])]

    pattern = re.compile("\d*\.\d+|\d+")
    verb_df.per = verb_df.per.apply(lambda v: float(pattern.findall(str(v))[0]))
    verb_df = verb_df[(verb_df.per > 0) & (verb_df.per < 100)]

    return verb_df[verb_df.verb.isin(VERBS['rising'])], verb_df[verb_df.verb.isin(VERBS['falling'])]


def encode(series):
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False)

    label_encoding = label_encoder.fit_transform(series)
    label_encoding = label_encoding.reshape(len(label_encoding), 1)

    one_hot_encoding = one_hot_encoder.fit_transform(label_encoding)

    return one_hot_encoding, label_encoder.classes_


def estimate(method, direction, estimator_class, df, X, scorer, smooth_lambda=0.5):
    encoding, classes_ = encode(df.verb)
    estimator = estimator_class(classes_, smooth_lambda)

    print('{0}, the MMR of {1} verbs'.format(method, direction))
    # print(cross_validation.cross_val_score(
    #     estimator=estimator, X=X, y=encoding, scoring=scorer, cv=5))
    # print('-' * 37)
    mrr_result = cross_validation.cross_val_score(estimator=estimator, X=X, y=encoding, scoring=scorer, cv=5)
    print(mrr_result)
    avg = sum(mrr_result) / len(mrr_result)
    print(avg)
    print("std: " + str(np.std(mrr_result)))
    print('-' * 37)


def run_estimators(rising_verbs, falling_verbs):
    scorer = make_scorer(label_ranking_average_precision_score)

    estimate('MLP', 'rising', VerbSelectMLPMMREstimator, rising_verbs, rising_verbs.per.values.reshape(-1, 1), scorer, 0.05)
    estimate('MLP', 'falling', VerbSelectMLPMMREstimator, falling_verbs, falling_verbs.per.values.reshape(-1, 1), scorer, 0.05)

    # Beta Estimator
    estimate('beta_estimator', 'rising', VerbSelectBetaMRREstimator, rising_verbs, rising_verbs.per, scorer, 0.05)
    estimate('beta_estimator', 'falling', VerbSelectBetaMRREstimator, falling_verbs, falling_verbs.per, scorer, 0.05)

    # Baseline Estimator
    # estimate('baseline', 'rising', VerbSelectBaselineMRREstimator, rising_verbs, rising_verbs.per, scorer)
    # estimate('baseline', 'falling', VerbSelectBaselineMRREstimator, falling_verbs, falling_verbs.per, scorer)

    # KDE Estimator
    estimate('KDE', 'rising', VerbSelectKdeMRREstimator, rising_verbs, rising_verbs.per, scorer, 0.05)
    estimate('KDE', 'falling', VerbSelectKdeMRREstimator, falling_verbs, falling_verbs.per, scorer, 0.05)


def report(rising_verbs):
    y_beta_list = []
    y_baseline_list = []
    y_kde_list = []

    encoded_verbs, classes_ = encode(rising_verbs.verb)

    kf = cross_validation.KFold(len(rising_verbs.per), n_folds=5)
    for train, test in kf:
        X_train = rising_verbs.iloc[train]
        X_test = rising_verbs.iloc[test]
        y_train = [encoded_verbs[i] for i in train]
        # y_test = [encoded_verbs[i] for i in test]
        beta_estimator = VerbSelectBetaMRREstimator(classes_)
        baseline_estimator = VerbSelectBaselineMRREstimator(classes_)
        kde_estimator = VerbSelectKdeMRREstimator(classes_)
        # mrr_scorer = make_scorer(label_ranking_average_precision_score)
        beta_estimator.fit(X_train.per, y_train)
        baseline_estimator.fit(X_train.per, y_train)
        kde_estimator.fit(X_train.per, y_train)
        for i in range(len(X_test)):
            item = X_test.iloc[i]
            y_beta_list.append(beta_estimator.p_posterior(item.verb, item.per))
            y_kde_list.append(kde_estimator.p_posterior(item.verb, item.per))
            y_baseline_list.append(baseline_estimator.p_posterior(item.verb, item.per))
    print("the sign-test of beta and baseline: ")
    print(stats.wilcoxon(y_beta_list, y_baseline_list))
    print("-" * 37)
    print("the sign-test of beta and KDE:")
    print(stats.wilcoxon(y_beta_list, y_kde_list))
    print("-" * 37)
    print("the sigh-test of KDE and baseline: ")
    print(stats.wilcoxon(y_kde_list, y_baseline_list))
    print("-" * 37)


def main():
    rising_verbs, falling_verbs = preprocess('../Data/bllip_triples', VERBS)
    # rising_verbs, falling_verbs = preprocess('../Data/Reuters_triples', VERBS_Reuters)
    # rising_verbs, falling_verbs = preprocess('../Data/Chinese_news_triples', VERBS_Chinese)
    run_estimators(rising_verbs, falling_verbs)
    report(rising_verbs)
    # report(falling_verbs)

if __name__ == "__main__":
    main()
