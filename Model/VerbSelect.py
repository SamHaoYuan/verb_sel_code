# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:47:17 2017

@author: Sam
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from random import choices


class VerbSelectBase(BaseEstimator):
    def __init__(self, verbs, smooth_lambda=0.05):
        """
        Fit data with beta distribution
        performance is MRR
        :param verbs: the ordered set of verbs ordered by
        """
        self.parameters = dict()
        self.verbs_count = dict()
        self.verbs = verbs
        self.smooth_lambda = smooth_lambda


class VerbSelectBetaMRREstimator(VerbSelectBase):
    def fit(self, X, y):
        """
        :param X：X is Series, the value is between 0 and 100
        :param y: y is the labeled vector
        :return: get all beta distribution parameters for all verbs
        """
        df_verbs = pd.DataFrame(y, columns=self.verbs)
        df_verbs['per'] = X.tolist()

        for verb in self.verbs:
            data = df_verbs[df_verbs[verb] == 1].per
            a, b, loc, scale = stats.beta.fit(data, floc=0, fscale=100)
            self.parameters[verb] = [a, b]
            self.verbs_count[verb] = data.size

    def predict(self, X):
        """
        given percentages, return the probability vectors of selecting the corresponding verb
        :param X: X must be a iterator, and the value of it is percents.
        :return: probability, calculated by the parameters
        """
        return np.array([self.p_x(x) for x in X])

    def p_verb(self, x):
        """
        given the per, return the sample verbs
        :param X:
        :return: verb
        """
        pro = self.p_x(x)
        return choices(self.verbs, pro)

    def m_verb(self, x):
        """
        given the per, return the argmax verbs
        :param X:
        :return: verb
        """
        pro = self.p_x(x)
        return self.verbs[np.argmax(pro)]

    def p_x(self, x):
        """
        given the percentage, return the probability vector for verbs
        :param x:
        :return:
        """
        return np.array([self.p_posterior(v, x) for v in self.verbs])

    def p_posterior(self, w, x):
        """
        given the verb w and percent value x, return the posterior probability P(w|x)
        :param w:
        :param x:
        :return:
        """
        P_w = self.P(w) * self.p_c(x, w)
        P_sum = sum(self.P(verb) * self.p_c(x, verb) for verb in self.verbs)

        return P_w / P_sum

    def P(self, w):
        """
        given the verb w, return it's probability in this data
        :param w:
        :return:
        """
        verbs_count_sum = sum(self.verbs_count[key] for key in self.verbs_count)
        # return float(self.verbs_count[w]) / verbs_count_sum
        p_mle = float(self.verbs_count[w]) / verbs_count_sum
        p_uni = float(1) / len(self.verbs)
        return self.smooth_lambda * p_mle + (1 - self.smooth_lambda) * p_uni

    def p_c(self, x, w):
        """
        given the percentage and verb, return its pdf.
        :param x:
        :param w:
        :return:
        """
        a, b = self.parameters[w]
        return stats.beta.pdf(x, a, b, 0, 100)


class VerbSelectBaselineMRREstimator(VerbSelectBase):
    def fit(self, X, y):
        """
        :param X：X is Series, the value is between 0 and 100
        :return: the IQRs of all verbs (Q1 and Q3）
        """
        df_verbs = pd.DataFrame(y, columns=self.verbs)
        df_verbs['per'] = X.tolist()

        for verb in self.verbs:
            data = df_verbs[df_verbs[verb] == 1].per
            a, b = data.quantile([0.25, 0.75])
            self.parameters[verb] = [a, b]

    def predict(self, X):
        """
        given percentages, return the probability vectors by the IQR
        """
        return np.array([self.p_x(x) for x in X])

    def p_x(self, x):
        """
        given the percentage, return the probability vector for rising verbs
        :param x:
        :return:
        """
        score = []
        p = self.p_w(x)
        for verb in self.verbs:
            q1, q2 = self.parameters[verb]
            if q1 <= x <= q2:
                score.append(p)
            else:
                score.append(0)
        return score

    def p_w(self, x):
        """
        calculate the numbers of optional verbs and return the probability
        :param x:
        :return:
        """
        count = 0  # numbers of optional verbs
        for verb in self.verbs:
            q1, q2 = self.parameters[verb]
            if q1 <= x <= q2:
                count += 1

        return 0 if count == 0 else float(1) / count

    def p_posterior(self, w, x):
        """
        given the test percentage, return the probability of the corresponding verb
        :param w:
        :param x:
        :return:
        """
        p = self.p_w(x)
        q1, q2 = self.parameters[w]
        return p if q1 <= x <= q2 else 0

    def get_verbs(self, x):
        """
        given the percent, return all the selected verbs
        :return:
        """
        verbs = []
        for verb in self.verbs:
            q1, q2 = self.parameters[verb]
            if q1 <= x <= q2:
                verbs.append(verb)
        if not verbs:
            verbs = self.verbs
        return choices(verbs)[0]


class VerbSelectKdeMRREstimator(VerbSelectBase):

    def fit(self, X, y):
        # x = percentage, y = digits
        # order of verbs is given by internal dicts
        # this only works because y order is predictable
        # X can be a series, y can be 1 hot encoded df
        df_verbs = pd.DataFrame(y, columns=self.verbs)
        df_verbs['per'] = X.tolist()

        # for each verb
        for verb in self.verbs:

            # the distribution of the percentages for a given verb
            data = df_verbs[df_verbs[verb] == 1].per
            model = stats.gaussian_kde(data)

            self.parameters[verb] = model

            # should be the length of the data
            self.verbs_count[verb] = data.size

    def predict(self, X):
        # what is the y (verb) for a given X ie (percentage)
        predictions = []

        for x in X:

            Ps = np.array([(self.P(v), self.p_c(x, v)) for v in self.verbs])

            P = Ps[:, 0]
            P_C = Ps[:, 1]

            P_v = P * P_C
            P_sum = (P *  P_C).sum()

            predictions.append(P_v / P_sum)

        return np.array(predictions)

    def p_x(self, x):
        return [self.p_posterior(v, x) for v in self.verbs]

    def p_posterior(self, w, x):
        P_w = self.P(w) * self.p_c(x, w)
        P_sum = sum(self.P(verb) * self.p_c(x, verb) for verb in self.verbs)

        return P_w / P_sum

    def P(self, verb):
        verbs_count_sum = sum(self.verbs_count[key] for key in self.verbs_count)
        # return float(self.verbs_count[verb]) / verbs_count_sum
        p_mle = float(self.verbs_count[verb]) / verbs_count_sum
        p_uni = float(1) / len(self.verbs)
        # p_smo = self.lamda*p_mle + (1-self.lamda)*p_uni
        return self.smooth_lambda * p_mle + (1 - self.smooth_lambda) * p_uni

    def p_c(self, x, verb):
        model = self.parameters[verb]
        return float(model.pdf(x))


class VerbSelectMLPMMREstimator(VerbSelectBase):
    def __init__(self, verbs, smo):
        super(VerbSelectMLPMMREstimator, self).__init__(verbs)
        self.estimator = MLPClassifier(hidden_layer_sizes=(1, 100, len(self.verbs)), max_iter=1500)

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict_proba(X)

    def get_verbs(self, x):
        pro = self.predict(x)[0]
        return choices(self.verbs, pro)[0]

    def m_verbs(self, x):
        loc = self.predict(x).argmax()
        return self.verbs[loc]