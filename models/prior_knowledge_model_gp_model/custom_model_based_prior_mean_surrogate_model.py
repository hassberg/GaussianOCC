import tensorflow as tf
from typing import Tuple

import torch
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
from models.prior_knowledge_model_gp_model.custom_model_prior_gaussian_process import CustomModelBasedGaussianProcess

from gpytorch.likelihoods import GaussianLikelihood


class CustomModelBasedPriorMeanSurrogateModel(SurrogateModel):

    def __init__(self, **params):
        self.training_points = None
        self.training_values = None
        self.gaussian_process_model = None
        self.model_parameter = params

    def post_init(self, data_retriever):
        self.query_pool = data_retriever.get_query_pool()
        self.point_shape = data_retriever.point_shape
        self.value_shape = data_retriever.value_shape
        initial_data_point = torch.zeros(1, (len(self.model_parameter["init_points"][0])), dtype=torch.double)
        initial_sample = torch.zeros(1, dtype=torch.double)
        self.gaussian_process_model = CustomModelBasedGaussianProcess(self.model_parameter["init_points"],
                                                                      initial_data_point, initial_sample,
                                                                      GaussianLikelihood(), self.model_parameter)
        params = {
            'covariance_module.lengthscale': self.model_parameter['lengthscale'],
            'likelihood.noise': 0.0004
        }
        self.gaussian_process_model.initialize(**params)


    def learn(self, points: tf.Tensor, values: tf.Tensor):
        points = tf.cast(points, tf.float64)
        if self.training_points is None:
            self.training_points = points
            self.training_values = values
        else:
            self.training_points = tf.concat([self.training_points, points], 0)
            self.training_values = tf.concat([self.training_values, values], 0)

        self.gaussian_process_model.train()
        self.gaussian_process_model.fit(self.training_points, tf.reshape(self.training_values, [-1]))
        self.gaussian_process_model.eval()

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        if points.dtype is not tf.float64:
            points = tf.cast(points, tf.float64)

        prediction = self.gaussian_process_model.likelihood(
            self.gaussian_process_model(torch.as_tensor(points.numpy())))

        return prediction.stddev.detach()

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if points.dtype is not tf.float64:
            points = tf.cast(points, tf.float64)

        prediction = self.gaussian_process_model.likelihood(
            self.gaussian_process_model(torch.as_tensor(points.numpy())))

        return points, prediction.mean.detach()
