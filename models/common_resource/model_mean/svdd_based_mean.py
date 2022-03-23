import numpy
from gpytorch.means import Mean
import tensorflow as tf
import torch
from models.common_resource.BaseSVDD import BaseSVDD
from sklearn.svm import OneClassSVM
import numpy as np


class SvddBasedMean(Mean):
    def __init__(self, available_points: tf.Tensor, params):
        super(SvddBasedMean, self).__init__()

        self.model = OneClassSVM(kernel=params['kernel'], gamma=params['gamma'])
        self.model.fit(available_points.numpy())
        # self.center = self.model.center
        # self.radius = self.model.radius
        # distances = self.model.get_distance(available_points.numpy())
        # sorted_dist = np.sort(np.array(distances).flatten())
        # inliers = sorted_dist[:np.argmax(sorted_dist > self.radius)]
        # outliers = sorted_dist[np.argmax(sorted_dist > self.radius):]
        # self.avg_inlier_dist = np.mean(inliers)
        # self.avg_outlier_dist = np.mean(outliers)

    def forward(self, x):
        # distances = np.array(self.model.get_distance(x)).flatten()
        try:
            scores = self.model.predict(x.numpy())
        except ValueError as e:
            z = e
            print("Value error for: " + str(x))

        # for dist in distances:
        #     if dist < self.radius:
        #         if dist < self.avg_inlier_dist:
        #             scores.append(1)
        #         else:
        #             score = ((dist - self.avg_inlier_dist) / (self.radius - self.avg_inlier_dist))
        #             scores.append(score)
        #     else:
        #         if dist > self.avg_outlier_dist:
        #             scores.append(-1)
        #         else:
        #             score = -1 * ((dist - self.avg_outlier_dist)/(self.radius - self.avg_outlier_dist))
        #             scores.append(score)
        return torch.Tensor(scores)
