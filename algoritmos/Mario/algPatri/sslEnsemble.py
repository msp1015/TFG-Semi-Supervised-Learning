#Este fragmento servira como base para la implementacion de algoritmos semi-supervisados

#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   SSLEnsembleClassifier.py
@Time    :   2023/05/05 12:25:32
@Author  :   Patricia Hernando Fernández 
@Version :   1.0
@Contact :   phf1001@alu.ubu.es
"""

from sklearn.metrics import recall_score, precision_score
import numpy as np
import numbers


class SSLEnsemble:
    """
    SSL Ensemble Classifier.

    Defines the parent class for all the ensemble classifiers.
    Contains the common methods and attributes.
    """

    def __init__(self, classes=None, random_state=None):
        """
        Constructor. Creates the SSL ensemble classifier.

        Parameters
        ----------
        classes: list
            List of classes
        random_state:
            Random object to create deterministic experiments
        """
        self.random_state = self.check_random_state(random_state)

        if classes is None:
            classes = []
        self.classes = classes

    def predict(self, samples):
        """
        Returns the labels predicted by the classifier for a given data.

        Parameters
        ----------
        samples: np_array
            samples to predict

        Returns
        -------
        np.array:
            labels predicted by the classifier
        """
        samples = (lambda x: np.expand_dims(x, axis=0) if x.ndim == 1 else x)(
            samples
        )
        return np.array([self.single_predict(sample) for sample in samples])

    def predict_proba(self, samples):
        """
        Returns the probabilities predicted by the classifier for a given data.

        Parameters
        ----------
        samples: np_array
            samples to predict

        Returns
        -------
        np.array:
            array containing one array for each sample with probabilities
            for each class.
        """
        samples = (lambda x: np.expand_dims(x, axis=0) if x.ndim == 1 else x)(
            samples
        )
        return np.array(
            [self.single_predict_proba(sample) for sample in samples]
        )

    def score(self, X_test, y_test):
        """
        Calculates the number of hits by the classifier for a given training set.

        Parameters
        ----------
        X_test: np_array
            Samples used during testing
        y_test: np_array
            Samples' tags

        Returns
        -------
        float:
            percentage of hits.
        """
        y_predictions = self.predict(X_test)
        return np.count_nonzero(y_predictions == y_test) / len(y_test)

    @staticmethod
    def check_random_state(seed=None):
        """
        Turn seed into a np.random.RandomState instance.
        Source: Scikit-Learn

        Parameters
        ----------
        seed : None, int or instance of RandomState
            If None, return the RandomState singleton.
            If int, return a new RandomState seeded with seed.
            If RandomState instance, return it.

        Returns
        -------
        numpy.random.RandomState
            The random state object based on seed parameter.
        """
        if seed is None or seed is np.random:
            return np.random.mtrand._rand

        if isinstance(seed, numbers.Integral):
            return np.random.RandomState(seed)

        if isinstance(seed, np.random.RandomState):
            return seed

        return None

    @staticmethod
    def recall(y_true, y_pred):
        """
        Returns recall.

        Parameters
        ----------
        y_true: np.array with true labels
        y_pred: np.array with labels predicted by the classifier

        Returns
        -------
        Recall score
        """
        return recall_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        """
        Returns precision.

        Parameters
        ----------
        y_true: np.array with true labels
        y_pred: np.array with labels predicted by the classifier

        Returns
        -------
        Precision score
        """
        return precision_score(y_true, y_pred)