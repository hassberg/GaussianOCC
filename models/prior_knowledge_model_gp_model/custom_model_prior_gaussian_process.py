import tensorflow as tf
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from gpytorch.means.constant_mean import ConstantMean
from models.prior_knowledge_model_gp_model.model_mean.svdd_based_mean import SvddBasedMean


class CustomModelBasedGaussianProcess(ExactGP):
    def __init__(self, all_data: tf.Tensor, train_data: tf.Tensor, train_values, likelihood):
        super(CustomModelBasedGaussianProcess, self).__init__(train_data, train_values,
                                                              likelihood)  # TODO replace with actual value
        self.mean_module = SvddBasedMean(all_data)
        self.covariance_module = ScaleKernel(RBFKernel())
        self.likelihood.initialize(noise=1)
        self.eval()

        # random numbers, used to stop learning process
        self.stable_iterations = 3
        self.stable_eps = 0.001

    def forward(self, x: tf.Tensor):
        mean = self.mean_module(x).double()
        covariance = self.covariance_module(x)
        return MultivariateNormal(mean.flatten(), covariance)

    def fit(self, points: tf.Tensor, values: tf.Tensor):
        self.set_train_data(inputs=torch.as_tensor(points.numpy()), targets=torch.as_tensor(values.numpy()),
                            strict=False)

        optimizer = Adam(self.parameters())
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        # scheduler = ExponentialLR(optimizer, gamma=0.9)

        stable = False
        previous_loss = torch.tensor(0.0)
        stable_since = 0
        while not stable:
            optimizer.zero_grad()
            output = self(torch.as_tensor(points.numpy()))
            loss = -mll(output, torch.as_tensor(values.numpy(), dtype=torch.double))
            loss = loss.sum()
            loss.backward()

            optimizer.step()
            # scheduler.step()

            stable, stable_since = self.eval_stability(previous_loss, loss, stable_since)
            previous_loss = loss

    def eval_stability(self, previous_loss, current_loss, stable_since):
        if np.absolute(np.subtract(previous_loss.detach().numpy(), current_loss.detach().numpy())) < self.stable_eps:
            stable_since += 1
        else:
            stable_since = 0

        return stable_since == self.stable_iterations, stable_since