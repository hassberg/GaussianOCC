import torch
import tensorflow as tf
from gpytorch.means.mean import Mean
from prior_knowledge_gp_model.mean_estimator.sample_based_mean import SampleBasedMeanEstimation


# TODO usage a trained model, e.g SVDD, to derive prior mean knowledge
class PriorMeanEstimator(Mean):
    def __init__(self, available_points: tf.Tensor, assumed_prior_knowledge: tf.Tensor):
        super(PriorMeanEstimator, self).__init__()
        prior_assumptions = tf.concat((available_points, assumed_prior_knowledge), axis=1)
        self.mean_estimator = SampleBasedMeanEstimation(prior_assumptions)

    def forward(self, x):
        res = self.mean_estimator.calculate_means(x)
        return torch.FloatTensor(res)
