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

from models.common_resource.model_mean.svdd_based_mean import SvddBasedMean


class VanishingSelfTrainingCustomModelBasedGaussianProcess(ExactGP):
    def __init__(self, all_data: tf.Tensor, train_data: tf.Tensor, train_values, likelihood, params):
        super(VanishingSelfTrainingCustomModelBasedGaussianProcess, self).__init__(train_data, train_values,
                                                                                   likelihood)  # TODO replace with actual value
        self.mean_module = SvddBasedMean(all_data, params)
        self.covariance_module = RBFKernel()
        self.vanishing_factors = []

        self.dims = 2
        self.vanishes_per_dim = 3
        self.vr = {}
        self.eval()

    def init_vanishing_factor(self, data_retriever: DataRetriever):
        # ranges = [(min(data_retriever.data_source.data_points[:, dim]), max(data_retriever.data_source.data_points[:, dim])) for dim in
        #           range(len(data_retriever.data_source.data_points[0]))]
        vr = {}

        for i in range(self.dims):
            neg_splits = list(map(lambda x: -10 ** x, np.linspace(start=-1, stop=np.log(2), num=self.vanishes_per_dim)))  # todo
            pos_splits = list(map(lambda x: 10 ** x, np.linspace(start=-1, stop=np.log(2), num=self.vanishes_per_dim)))

            vanishing_ranges = [(float('-inf'), neg_splits[len(neg_splits) - 1])]
            for j in range(len(neg_splits) - 1, 0, -1):
                vanishing_ranges.append((neg_splits[j], neg_splits[j - 1]))
            vanishing_ranges.append((neg_splits[0], pos_splits[0]))
            for j in range(len(pos_splits) - 1):
                vanishing_ranges.append((pos_splits[j], pos_splits[j + 1]))
            vanishing_ranges.append((pos_splits[len(pos_splits) - 1], float('inf')))
            vr[i] = vanishing_ranges

            for k in range(len(vanishing_ranges)):
                param_name = "vanishing_factor-" + str(i) + "-" + str(k)
                self.register_parameter(
                    name=param_name, parameter=torch.nn.Parameter(torch.as_tensor(1.0))
                )
                length_constraint = Interval(lower_bound=torch.as_tensor(0.0), upper_bound=torch.as_tensor(1.0))
                self.register_constraint(param_name, length_constraint)
        self.vr = vr

    def forward(self, x: tf.Tensor):
        # mean = self.mean_module(x).double() * self.vanish_factor
        mean = self.mean_module(x).double() * self.get_vanishing_factor(x)
        covariance = self.covariance_module(x)
        return MultivariateNormal(mean.flatten(), covariance)

    def fit(self, points: tf.Tensor, values: tf.Tensor):
        self.set_train_data(inputs=torch.as_tensor(points.numpy()), targets=torch.as_tensor(values.numpy()),
                            strict=False)

        optimizer = Adam(self.parameters(), lr=0.01)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(50):
            optimizer.zero_grad()
            output = self(torch.as_tensor(points.numpy()))
            loss = -mll(output, torch.as_tensor(values.numpy(), dtype=torch.double))
            loss = loss.sum()
            loss.backward()

            optimizer.step()

    def get_vf_range_window(self, dim, x):
        for i in range(len(self.vr[dim])):
            if self.vr[dim][i][0] <= x < self.vr[dim][i][1]:
                return i

    def get_vanishing_factor(self, x):
        vanishing_factors = []
        for pt in x:
            vf = 1.0
            for d in range(self.dims):
                dim_vf = self.get_parameter("vanishing_factor-" + str(d) + "-" + str(self.get_vf_range_window(d, pt[d])))
                if float(dim_vf) != 1.0:
                    print("vs != 0")
                vf = vf * float(dim_vf)
            vanishing_factors.append([vf])
        return torch.as_tensor(vanishing_factors)
