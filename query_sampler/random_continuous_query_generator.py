from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.query_selection.query_sampler import QuerySampler
import tensorflow as tf
import numpy as np


class RandomContinuousQueryGenerator(QuerySampler):
    def __init__(self):
        self.pool: Pool = None
        self.ranges = []

    def post_init(self, pool):
        self.pool = pool
        self.ranges = [(i[0][0], i[len(i) - 1][1]) for i in self.pool.ranges]

    def update_pool(self, pool):
        self.pool = pool
        self.ranges = [(i[0][0], i[len(i) - 1][1]) for i in self.pool.ranges]

    def sample(self, num_queries: int = 1) -> tf.Tensor:
        if self.pool.is_discrete():
            raise Exception("Should only be continuous pool")

        # rd = tf.random.uniform(shape=(num_queries, self.pool.shape[0]), minval=[i[0] for i in self.ranges], maxval=[i[1] for i in self.ranges])
        rd = self.generate(num_queries)

        return tf.convert_to_tensor(rd, dtype=tf.float64)

    def generate(self, num_queries):
        points = []

        for j in range(self.pool.shape[0]):
            if points == []:
                points.append((self.ranges[j][1] - self.ranges[j][0]) * np.random.uniform(size=num_queries) + self.ranges[j][0])
                points = np.transpose(np.asarray(points))
            else:
                points = np.c_[points, ((self.ranges[j][1] - self.ranges[j][0]) * np.random.uniform(size=num_queries) + self.ranges[j][0])]
        return points
