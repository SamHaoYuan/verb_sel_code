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
from VerbSelect_smoothing import (
    VerbSelectBetaMRREstimator, VerbSelectKdeMRREstimator,
    VerbSelectBaselineMRREstimator, VerbSelectMLPMMREstimator
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import stats
import random
from questionnaire_survey import Question, QuestionnaireSurvey
subject_words = ['Gross domestic product', 'Net profits', 'Share prices']
Chinese_subject_words = ['国民生产总值', '净利润', '股票价格']

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


def preprocess(path, verbs):
    verb_df = pd.read_csv(path, usecols=COLUMNS)
    verb_df = verb_df[verb_df.verb.isin(verbs['rising'] + verbs['falling'])]

    pattern = re.compile("\d*\.\d+|\d+")
    verb_df.per = verb_df.per.apply(lambda v: float(pattern.findall(str(v))[0]))
    verb_df = verb_df[(verb_df.per > 0) & (verb_df.per < 100)]

    return verb_df[verb_df.verb.isin(verbs['rising'])], verb_df[verb_df.verb.isin(verbs['falling'])]


def encode(series):
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False)

    label_encoding = label_encoder.fit_transform(series)
    label_encoding = label_encoding.reshape(len(label_encoding), 1)

    one_hot_encoding = one_hot_encoder.fit_transform(label_encoding)

    return one_hot_encoding, label_encoder.classes_


def generate_survey(dataset_name, direction, df, X, subjects, smooth_lambda=0.05, n=75, language='Eng'):
    encoding, classes_ = encode(df.verb)
    print("train our model")
    # estimator = estimator_class(classes_, smooth_lambda)
    # estimator.fit(X, y=encoding)
    estimator = VerbSelectBetaMRREstimator(classes_, smooth_lambda)
    estimator.fit(X, y=encoding)
    print("train baseline")
    # baseline_estimator = baseline_estimator(classes_, smooth_lambda)
    # baseline_estimator.fit(X.values.reshape(-1, 1), y=encoding)  # MLP
    baseline_estimator = VerbSelectMLPMMREstimator(classes_, smooth_lambda)
    baseline_estimator.fit(X.values.reshape(-1, 1), y=encoding)  # MLP
    print("generate survey")
    question_survey = QuestionnaireSurvey(n, language)
    result_path = '../UserEvaluation2/' + dataset_name + '_' + direction
    question_survey.generate_questions(subjects, list(df.per), baseline_estimator, estimator, result_path)
    # result_path = 'questions/' + dataset_name + '_' + direction
    question_survey.generate_survey(result_path)


def generate_all_survey(rising_verbs, falling_verbs, _name, words, language='Eng'):
    generate_survey(_name, 'rising',  rising_verbs, rising_verbs.per, words, language=language)
    generate_survey(_name, 'falling', falling_verbs, falling_verbs.per, words, language=language)


def main():
    rising_verbs_WSJ, falling_verbs_WSJ = preprocess('triples.csv', VERBS)
    rising_verbs_Returs, falling_verbs_Reuter = preprocess('../reuter/Reuters_triples', VERBS_Reuters)
    rising_verbs_Chinese, falling_verbs_Chinese = preprocess('../Chinese_news/Chinese_news_triples', VERBS_Chinese)
    # WSJ
    print("WSJ: ")
    generate_all_survey(rising_verbs_WSJ, falling_verbs_WSJ, 'WSJ', subject_words)
    # Retures
    print("Reuters: ")
    generate_all_survey(rising_verbs_Returs, falling_verbs_Reuter, 'Reuters', subject_words)
    # Chinese
    print("Chinese: ")
    generate_all_survey(rising_verbs_Chinese, falling_verbs_Chinese, 'Chinese', Chinese_subject_words, 'Chinese')
    # report(rising_verbs)

if __name__ == "__main__":
    main()
