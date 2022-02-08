import numpy as np
import math

import torch

from prior_knowledge_gp_model.classifiers.outlier_scoring_method import OutlierScoringMethod


class LocalOutlierFactor(OutlierScoringMethod):

    def __init__(self, data, k):
        super().__init__(data)
        self.k = k

    def calculate_scoring(self):
        # distance matrix
        pairwise_distance = [[0] * len(self.available_data) for _ in range(len(self.available_data))]

        # calculating pairwise distance
        for i in range(len(self.available_data)):
            for j in range(i, len(self.available_data)):
                dist = math.dist(self.available_data[i], self.available_data[j])
                pairwise_distance[i][j] = dist
                pairwise_distance[j][i] = dist

        # calculating the kNN
        kNN = [[0] * self.k for _ in range(len(self.available_data))]
        for i in range(len(self.available_data)):
            nearest = np.argpartition(pairwise_distance[i], (1, self.k + 1))
            for j in range(1, self.k + 1):
                kNN[i][j - 1] = nearest[j]

        # calculating lrd
        lrd = np.array([0] * len(self.available_data), dtype=float)
        for i in range(len(self.available_data)):
            score = np.float(0)
            for j in kNN[i]:
                score += np.maximum(pairwise_distance[j][kNN[j][self.k - 1]], pairwise_distance[i][j])
            lrd[i] = np.divide(1, np.divide(score, self.k))

        # finally calculating lof
        lof = np.array([0] * len(self.available_data), dtype=float)
        for i in range(len(self.available_data)):
            score = np.float(0)
            for j in kNN[i]:
                score += lrd[j]
            lof[i] = np.divide(np.divide(score, self.k), lrd[i])

        return torch.as_tensor(self.linear_normalization(lof))

    @staticmethod
    def linear_normalization(scoring):
        max_scoring = np.amax(scoring)
        min_scoring = np.amin(scoring)
        # min_scoring = 0

        normalized_scoring = np.array([])
        for i in range(len(scoring)):
            normalized_scoring = np.append(normalized_scoring, np.subtract(np.multiply(
                np.subtract(1, np.divide(np.subtract(scoring[i], min_scoring), np.subtract(max_scoring, min_scoring))),
                2), 1))

        return np.reshape(normalized_scoring, (len(normalized_scoring), 1))