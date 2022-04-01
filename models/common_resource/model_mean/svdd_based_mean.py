import numpy as np
import tensorflow as tf
import torch
from gpytorch.means import Mean

from models.common_resource.BaseSVDD import BaseSVDD


class SvddBasedMean(Mean):
    def __init__(self, available_points: tf.Tensor, params):
        super(SvddBasedMean, self).__init__()

        self.model = BaseSVDD(C=params['C'], kernel=params['kernel'], gamma=params['gamma'], display='off')
        self.model.fit(available_points)

    def forward(self, x):
        try:
            scores = self.model.predict(x.numpy())
        except ValueError as e:
            z = e
            print("Value error for: " + str(x))
            scores = np.zeros(len(x))

        return torch.Tensor(scores)
