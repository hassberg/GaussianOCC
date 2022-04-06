import tensorflow as tf
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch.nn.parameter import Parameter
from gpytorch.priors.prior import Prior
from gpytorch.priors import GammaPrior
from torch.optim import Adam

from models.common_resource.model_mean.svdd_based_mean import SvddBasedMean


class CustomModelBasedGaussianProcess(ExactGP):
    def __init__(self, all_data: tf.Tensor, train_data: tf.Tensor, train_values, likelihood, params):
        super(CustomModelBasedGaussianProcess, self).__init__(train_data, train_values,
                                                              likelihood)
        self.mean_module = SvddBasedMean(all_data, params)
        self.covariance_module = RBFKernel()
        self.eval()

    def forward(self, x: tf.Tensor):
        mean = self.mean_module(x).double()
        covariance = self.covariance_module(x)
        return MultivariateNormal(mean.flatten(), covariance)

    def fit(self, points: tf.Tensor, values: tf.Tensor):
        self.set_train_data(inputs=torch.as_tensor(points.numpy()), targets=torch.as_tensor(values.numpy()),
                            strict=False)
