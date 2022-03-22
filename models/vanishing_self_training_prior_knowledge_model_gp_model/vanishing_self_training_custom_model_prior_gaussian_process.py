import tensorflow as tf
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import Interval
from gpytorch.models import ExactGP
from torch.optim import Adam

from models.common_resource.model_mean.svdd_based_mean import SvddBasedMean


class VanishingSelfTrainingCustomModelBasedGaussianProcess(ExactGP):
    def __init__(self, all_data: tf.Tensor, train_data: tf.Tensor, train_values, likelihood, params):
        super(VanishingSelfTrainingCustomModelBasedGaussianProcess, self).__init__(train_data, train_values,
                                                                                   likelihood)  # TODO replace with actual value
        self.mean_module = SvddBasedMean(all_data, params)
        self.covariance_module = RBFKernel()

        self.register_parameter(
            name='vanish_factor', parameter=torch.nn.Parameter(torch.as_tensor(1.0))
        )
        length_constraint = Interval(lower_bound=torch.as_tensor(0.0), upper_bound=torch.as_tensor(1.0))
        self.register_constraint("vanish_factor", length_constraint)
        self.eval()

    def forward(self, x: tf.Tensor):
        mean = self.mean_module(x).double() * self.vanish_factor
        covariance = self.covariance_module(x)
        return MultivariateNormal(mean.flatten(), covariance)

    def fit(self, points: tf.Tensor, values: tf.Tensor):
        self.set_train_data(inputs=torch.as_tensor(points.numpy()), targets=torch.as_tensor(values.numpy()),
                            strict=False)

        optimizer = Adam(self.parameters(), lr=0.001)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(50):
            optimizer.zero_grad()
            output = self(torch.as_tensor(points.numpy()))
            loss = -mll(output, torch.as_tensor(values.numpy(), dtype=torch.double))
            loss = loss.sum()
            loss.backward()

            optimizer.step()
