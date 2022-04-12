from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.query_selection.query_sampler import QuerySampler
import tensorflow as tf


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

        rd = tf.random.uniform(shape=(num_queries, self.pool.shape[0]), minval=[i[0] for i in self.ranges], maxval=[i[1] for i in self.ranges])
        return tf.convert_to_tensor(rd)


