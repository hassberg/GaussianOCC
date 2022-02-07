import tensorflow as tf
from typing import Tuple

import torch
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
from prior_knowledge_gp_model.prior_mean_gaussian_process import PriorMeanGaussianProcess
from prior_knowledge_gp_model.classifiers.outlier_scoring_method import OutlierScoringMethod

from gpytorch.likelihoods import GaussianLikelihood


class GaussianPriorMeanSurrogateModel(SurrogateModel):

    def __init__(self, available_points: tf.Tensor, outlier_scoring_method: OutlierScoringMethod):
        self.training_points = None
        self.training_values = None
        # TODO calculate prior class distribution here instead?...
        initial_data_point = torch.empty(1, (len(available_points[0])), dtype=torch.double)
        initial_sample = torch.empty(1, dtype=torch.double)
        self.gaussian_process_model = PriorMeanGaussianProcess(available_points, outlier_scoring_method.calculate_scoring(),
                                                               initial_data_point, initial_sample, GaussianLikelihood())

    def learn(self, points: tf.Tensor, values: tf.Tensor):
        if self.training_points is None:
            self.training_points = points
            self.training_values = values
        else:
            self.training_points = tf.concat([self.training_points, points], 0)
            self.training_values = tf.concat([self.training_values, values], 0)

        self.gaussian_process_model.train()
        self.gaussian_process_model.fit(points, values)
        self.gaussian_process_model.eval()

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        # TODO check if could be simplified (omit GP call)
        prediction = self.gaussian_process_model.likelihood(self.gaussian_process_model(points))
        # standard deviation as measure for uncertainty..
        return prediction.stddev

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        prediction = self.gaussian_process_model.likelihood(self.gaussian_process_model(points))
        return prediction.mean
