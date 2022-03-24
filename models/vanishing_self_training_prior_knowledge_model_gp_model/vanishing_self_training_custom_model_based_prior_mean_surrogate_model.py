import gpytorch.constraints
import tensorflow as tf
from typing import Tuple

import torch
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
from models.constant_prior_gp_model.constant_gaussian_process import ConstantGaussianProcess
from models.prior_knowledge_model_gp_model.custom_model_prior_gaussian_process import CustomModelBasedGaussianProcess

from gpytorch.likelihoods import GaussianLikelihood

from models.vanishing_self_training_prior_knowledge_model_gp_model.vanishing_self_training_custom_model_prior_gaussian_process import \
    VanishingSelfTrainingCustomModelBasedGaussianProcess


class VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel(SurrogateModel):

    def __init__(self, **params):
        self.training_points = None
        self.training_values = None
        self.query_pool = None
        self.gaussian_process_model = None
        self.model_parameter = params

    def post_init(self, data_retriever):
        self.query_pool = data_retriever.get_query_pool()
        self.point_shape = data_retriever.point_shape
        self.value_shape = data_retriever.value_shape
        initial_data_point = torch.empty(1, (len(self.query_pool.get_all_elements()[0])), dtype=torch.double)
        initial_sample = torch.empty(1, dtype=torch.double)
        self.gaussian_process_model = VanishingSelfTrainingCustomModelBasedGaussianProcess(self.query_pool.get_all_elements(),
                                                                      initial_data_point, initial_sample,
                                                                      GaussianLikelihood(), self.model_parameter)

    def learn(self, points: tf.Tensor, values: tf.Tensor):
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
        prediction = self.gaussian_process_model.likelihood(
            self.gaussian_process_model(torch.as_tensor(points.numpy())))
        # standard deviation as measure for uncertainty..
        return prediction.stddev.detach()

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        prediction = self.gaussian_process_model.likelihood(
            self.gaussian_process_model(torch.as_tensor(points.numpy())))
        return points, prediction.mean.detach()
