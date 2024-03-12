# En este fragmento de codigo se probara el algoritmo Co-forest para la clasificacion de datos

#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   CoForestClassifier.py
@Time    :   2023/03/30 20:50:46
@Author  :   Patricia Hernando Fernández 
@Version :   3.0 Inheritance from SSLEnsemble
@Contact :   phf1001@alu.ubu.es
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from copy import deepcopy
import os
import sys

src_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(src_path)
from algPatri.sslEnsemble import SSLEnsemble


class coforest(SSLEnsemble):
    """
    SSL Co-Forest Classifier.

    M. Li and Z. -H. Zhou, "Improve Computer-Aided Diagnosis With
    Machine Learning Techniques Using Undiagnosed Samples"
    """

    def __init__(self, n, theta, max_features="log2", random_state=None):
        """
        Constructor. Creates the Co-Forest.

        Parameters
        ----------
        n: int
            Number of trees in the ensemble
        theta: float
            Tolerance
        random_state:
            Random object to create deterministic experiments
        max_features: 'sqrt', 'log2', None
            Number of features to consider when looking
            for the best split
        """
        super().__init__(classes=[], random_state=random_state)
        self.n = n
        self.theta = theta
        self.max_features = max_features
        self.ensemble = {}
        #MARIO
        self.accuracy_por_iteracion = []
    def create_trees(self, L, y, percentage=0.7):
        """
        Generates a dict containing co-forest's trees.
        Initializes the classifiers using bootstrapping.

        Please, note that original bagging sets the size of the bootstrap
        to the size of L (percentage should be =1). However, since small
        datasets are being used, the percentage is smaller to ensure that
        concomitant_oob_error is calculated.

        Parameters
        ----------
        L: np.array
            Labeled data used for training
        y: np.array
            Tags of the labeled data used for training
        percentage: float
            Size (p * labeled) to bootstrap

        Returns
        -------
        mask_L: np.array
            Mask with samples from L used for each tree

        """
        self.classes = np.unique(y)  # skipcq: PYL-W0201
        ensemble = {}
        mask_L = np.zeros(shape=((L.shape[0]), self.n), dtype=int, order="C")

        for i in range(self.n):
            # Original bagging sets the size of the bootstrap to the size of L
            rand_rows = self.random_state.choice(
                L.shape[0], replace=True, size=int(percentage * L.shape[0])
            )
            mask_L[rand_rows, i] = 1
            h = DecisionTreeClassifier(
                max_features=self.max_features,
                random_state=self.random_state
            )
        
            ensemble[i] = h.fit(L[rand_rows, :], y[rand_rows])

        self.ensemble = ensemble
        return mask_L

    def fit(self, L, y, U, X_test, y_test, w_init_criteria="percentage_L"):
        """
        Fits the ensemble using both labeled and
        pseudo-labeled data.

        Parameters
        ----------
        L: np.array
            Labeled data used for training
        y: np.array
            Tags of the labeled data used for training
        U: np.array
            Unlabeled data used for training
        w_init_criteria: str
            'percentage_L': w is initialized to 0.1 * len(L) (max 100)
            'confidence_L_all': w is initialized to the confidence of the
                            ensemble in L (includying low confidence values)
            'confidence_L_thetha': w is initialized to the confidence of the
                            ensemble in L (excluding low confidence values)
        """
        mask_L = self.create_trees(L, y)

            
        e = [0] * self.n
        W = [0] * self.n

        previous_e = [0.5] * self.n
        previous_W = self.initialize_previous_W(L, w_init_criteria)

        new_data = True

        while new_data:
            tree_changes = np.array([False] * self.n)
            tree_pseudo_updates = [() for i in range(self.n)]

            for i, hi in self.ensemble.items():
                e[i] = self.concomitant_oob_error(hi, L, y, mask_L)
                W[i] = previous_W[i]
                pseudo_labeled_data = []
                pseudo_labeled_tags = []

                if e[i] < previous_e[i]:
                    if e[i] == 0:
                        Wmax = self.theta * U.shape[0]
                    else:
                        Wmax = min(
                            self.theta * U.shape[0],
                            ((previous_e[i] * previous_W[i]) / e[i]),
                        )

                    U_subsampled = self.subsample(hi, U, Wmax)
                    #print("U_subsampled:\t", U_subsampled) #TODO REMOVE
                    W[i] = 0

                    for u in U_subsampled:
                        (
                            concomitant_confidence,
                            selected_class,
                        ) = self.concomitant_confidence(hi, U[u, :])

                        if concomitant_confidence > self.theta:
                            tree_changes[i] = True
                            pseudo_labeled_data.append(U[u, :])
                            pseudo_labeled_tags.append(selected_class)
                            W[i] += concomitant_confidence

                tree_pseudo_updates[i] = (
                    np.array(pseudo_labeled_data),
                    np.array(pseudo_labeled_tags),
                )
                #print("Errores y pesos en el arbol ", i, ":\t", e, W) #TODO REMOVE

            for i in np.fromiter(self.ensemble.keys(), dtype=int)[tree_changes]:
                if e[i] * W[i] < previous_e[i] * previous_W[i]:
                    self.retrain_tree(
                        i,
                        L,
                        y,
                        tree_pseudo_updates[i][0],
                        tree_pseudo_updates[i][1],
                        mask_L,
                    )

            previous_e = deepcopy(e)
            previous_W = deepcopy(W)

            if tree_changes.sum() == 0:
                new_data = False

            self.accuracy_por_iteracion.append(self.score(X_test, y_test))
            
    def get_accuracy_por_iteracion(self):
        return self.accuracy_por_iteracion
            
    def initialize_previous_W(self, L, w_init_criteria):
        """
        Initializes the previous_W array.

        The original paper suggests to use 0 as the initial value
        of previous_W. However, this leads to a situation where
        U can't be subsampled, since Wmax is always 0. We propose
        several alternatives to initialize previous_W.

        Parameters
        ----------
        L: np.array
            Labeled data used for training
        w_init_criteria: str
            'percentage_L': w is initialized to 0.1 * len(L) (max 100)
            'confidence_L_all': w is initialized to the confidence of the
                            ensemble in L (includying low confidence values)
            'confidence_L_thetha': w is initialized to the confidence of the
                            ensemble in L (excluding low confidence values)
        Returns
        -------
        W: np.array
            Weights of the unlabeled data
        """
        if w_init_criteria == "percentage_L":
            init_value = min(0.1 * len(L), 100)
            return [init_value] * self.n

        if "confidence_L" in w_init_criteria:
            if w_init_criteria == "confidence_L_all":
                exclude_low_confidence = False
            elif w_init_criteria == "confidence_L_thetha":
                exclude_low_confidence = True

            previous_W = [0] * self.n

            for i, hi in self.ensemble.items():
                for x in L:
                    concomitant_confidence = self.concomitant_confidence(hi, x)[
                        0
                    ]

                    if (
                        exclude_low_confidence
                        and concomitant_confidence > self.theta
                    ) or not exclude_low_confidence:
                        previous_W[i] += concomitant_confidence

            return previous_W
        raise ValueError("w_init_criteria not in the allowed values")

    def retrain_tree(
        self, i, L, y, pseudo_labeled_data, pseudo_labeled_tags, mask_L
    ):
        """
        Retrains a tree given new pseudo-labeled data.

        Parameters
        ----------
        i: int
            Index of the three
        L: np.array
            Labeled data used for training
        y: np.array
            Tags of the labeled data used for training
        pseudo_labeled_data: np.array
            Unlabeled samples
        pseudo_labeled_tags: np.array
            Pseudo-labels for the unlabeled samples
        mask_L: np.array
            Mask with samples from L used for each tree
        """
        pseudo_labeled_data = (
            lambda x: np.expand_dims(x, axis=0) if x.ndim == 1 else x
        )(pseudo_labeled_data)
        X_train = np.concatenate((L[mask_L[:, i] == 1], pseudo_labeled_data))
        y_train = np.concatenate((y[mask_L[:, i] == 1], pseudo_labeled_tags))
        self.ensemble[i] = self.ensemble[i].fit(X_train, y_train)

    def subsample(self, hi, U, Wmax):
        """
        Samples from U uniformly at random until the sum of the sample
        weights reaches Wmax. Bootstraping is applied.

        Parameters
        ----------
        hi: DecisionTreeClassifier
            Excluded tree
        U: np.array
            Unlabeled data used for training
        Wmax: float
            Max weight

        Returns
        -------
        np.array
            Array containing the index of the chosen
            samples from U
        """
        W = 0
        U_subsampled = []

        while W < Wmax:
            rand_row = self.random_state.choice(U.shape[0])
            W += self.concomitant_confidence(hi, U[rand_row, :])[0]
            U_subsampled.append(rand_row)

        return np.array(U_subsampled)

    def concomitant_oob_error(self, hi, L, y, mask_L):
        """
        Calculates the Out of Bag Error of the concomitant ensemble of hi
        for the whole labeled data.

        Parameters
        ----------
        hi: DecisionTreeClassifier
            Excluded tree
        L: np.array
            Labeled data used for training
        y: np.array
            Tags of the labeled data used for training
        mask_L: np.array
            Mask with samples from L used for each tree

        Returns
        -------
        float
            OOBE if trees voted, nan if not
        """
        errors = []

        for sample, tag in zip(L, y):
            n_votes = 0
            n_hits = 0

            for i, tree in self.ensemble.items():
                rows_training = L[mask_L[:, i] == 1]
                used_training = np.any(np.all(sample == rows_training, axis=1))

                if tree is not hi and not used_training:
                    if tree.predict([sample])[0] == tag:
                        n_hits += 1
                    n_votes += 1

            if n_votes > 0:
                errors.append(1 - (n_hits / n_votes))

        return np.mean(errors)

    def concomitant_confidence(self, hi, sample):
        """
        Calculates the number of coincidences during prediction of the hi
        concomitant ensemble for a data sample.

        Parameters
        ----------
        hi: DecisionTreeClassifier
            Excluded tree
        sample: np.array
            Sample's features array

        Returns
        -------
        tuple (float, int)
            float: confidence for the sample
            int: most agreed class
        """
        count = {i: 0 for i in self.classes}

        for tree in self.ensemble.values():
            if tree is not hi:
                count[tree.predict([sample])[0]] += 1

        max_agreement = max(count.values())
        most_agreed_class = max(count, key=count.get)

        return max_agreement / (len(self.ensemble) - 1), most_agreed_class

    def single_predict(self, sample):
        """
        Returns the class predicted by coforest for a given sample.
        Majority voting is used.

        Parameters
        ----------
        sample: np_array
            sample to predict

        Returns
        -------
        np.array:
            label predicted by coforest.
        """
        count = {i: 0 for i in self.classes}
        for i in (tree.predict([sample])[0] for tree in self.ensemble.values()):
            count[i] += 1

        return max(count, key=count.get)

    def single_predict_proba(self, sample):
        """
        Returns the probability for each class predicted by coforest
        for a given sample.

        Parameters
        ----------
        sample: np_array
            sample to predict

        Returns
        -------
        np.array:
            array containing probability for each class.
        """
        count = {i: 0 for i in self.classes}

        for i in (tree.predict([sample])[0] for tree in self.ensemble.values()):
            count[i] += 1

        votes = np.array(list(count.values()))
        return votes / self.n
