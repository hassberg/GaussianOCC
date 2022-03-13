import tensorflow as tf
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch.optim import Adam

from models.prior_knowledge_model_gp_model.model_mean.svdd_based_mean import SvddBasedMean


class CustomModelBasedGaussianProcess(ExactGP):
    def __init__(self, all_data: tf.Tensor, train_data: tf.Tensor, train_values, likelihood):
        super(CustomModelBasedGaussianProcess, self).__init__(train_data, train_values,
                                                              likelihood)  # TODO replace with actual value
        self.mean_module = SvddBasedMean(all_data)
        self.covariance_module = ScaleKernel(RBFKernel())
        self.eval()

    def forward(self, x: tf.Tensor):
        mean = self.mean_module(x).double()
        covariance = self.covariance_module(x)
        return MultivariateNormal(mean.flatten(), covariance)

    def fit(self, points: tf.Tensor, values: tf.Tensor):
        self.set_train_data(inputs=torch.as_tensor(points.numpy()), targets=torch.as_tensor(values.numpy()),
                            strict=False)

        optimizer = Adam(self.parameters())
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(20):
            optimizer.zero_grad()
            output = self(torch.as_tensor(points.numpy()))
            loss = -mll(output, torch.as_tensor(values.numpy(), dtype=torch.double))
            loss = loss.sum()
            loss.backward()

            optimizer.step()
