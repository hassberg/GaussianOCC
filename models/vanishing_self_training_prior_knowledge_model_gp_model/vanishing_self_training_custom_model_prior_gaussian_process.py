import tensorflow as tf
import torch
from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import Interval
from gpytorch.models import ExactGP
from torch.optim import Adam, SGD
import numpy as np
from random import sample

from models.common_resource.model_mean.svdd_based_mean import SvddBasedMean


class VanishingSelfTrainingCustomModelBasedGaussianProcess(ExactGP):
    def __init__(self, all_data: tf.Tensor, train_data: tf.Tensor, train_values, likelihood, params):
        super(VanishingSelfTrainingCustomModelBasedGaussianProcess, self).__init__(train_data, train_values,
                                                                                   likelihood)  # TODO replace with actual value
        self.mean_module = SvddBasedMean(all_data, params)
        self.covariance_module = RBFKernel()
        self.vanishing_factors = []

        self.dims = []
        self.picks = 5
        self.vanishes_per_dim = 2  # means .. *2 + 1
        self.vr = {}
        self.eval()

    def init_vanishing_factor(self, data_retriever: DataRetriever):
        dims = len(data_retriever.data_source.data_points[0])
        self.dims = sample(range(dims), np.minimum(dims, self.picks))

        neg_splits = list(map(lambda x: -2 ** x, np.linspace(start=-1, stop=np.log(2), num=self.vanishes_per_dim)))  # todo
        pos_splits = list(map(lambda x: 2 ** x, np.linspace(start=-1, stop=np.log(2), num=self.vanishes_per_dim)))

        vanishing_ranges = [(float('-inf'), neg_splits[len(neg_splits) - 1])]
        for j in range(len(neg_splits) - 1, 0, -1):
            vanishing_ranges.append((neg_splits[j], neg_splits[j - 1]))
        vanishing_ranges.append((neg_splits[0], pos_splits[0]))
        for j in range(len(pos_splits) - 1):
            vanishing_ranges.append((pos_splits[j], pos_splits[j + 1]))
        vanishing_ranges.append((pos_splits[len(pos_splits) - 1], float('inf')))
        self.vr = vanishing_ranges

        self.register_parameter(
            name="vf1", parameter=torch.nn.Parameter(torch.ones((len(self.vr) ** (len(self.dims)-1))))
        )
        length_constraint = Interval(lower_bound=torch.zeros((len(self.vr) ** (len(self.dims)-1))), upper_bound=torch.ones((len(self.vr) ** (len(self.dims)-1))))
        self.register_constraint("vf1", length_constraint)

    def forward(self, x: tf.Tensor):
        mean_unvanished = self.mean_module(x).double()
        vanishes = self.get_vanishing_factor(x)
        mean = mean_unvanished.flatten() * torch.matmul(vanishes, self.vf1)
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

    def get_vf_range_window(self, dim, x):
        for i in range(len(self.vr)):
            if self.vr[i][0] <= x < self.vr[i][1]:
                return i

    def get_vanishing_factor(self, x):
        vanishing_factors = torch.zeros((len(x), len(self.vf1)))
        for i in range(len(x)):
            y = 0
            for j in range(len(self.dims)):
                k = self.get_vf_range_window(j, x[i][self.dims[j]])
                y = y + len(self.vr) ** j * k
            vanishing_factors[i][j * len(self.vr) + k] = 1
        return vanishing_factors
