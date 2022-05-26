from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel

import tensorflow as tf


class MaximumContinuousQueryOptimizer(QueryOptimizer):
    # Optimizer, which selects of all available data sample the best scoring one which hasn't been sampled so far

    def __init__(self,
                 num_tries: int = 1):
        self.max_samples = None
        self.function = lambda x: tf.reduce_max(x)
        self.num_tries = num_tries

    def post_init(self, surrogate_model: SurrogateModel,
                  selection_criteria: SelectionCriteria,
                  query_sampler: QuerySampler,

                  ):
        self.surrogate_model = surrogate_model
        self.selection_criteria = selection_criteria
        self.query_sampler = query_sampler
        self.max_samples = surrogate_model.value_shape[0]

    def optimize_query_candidates(self):
        queries = self.query_sampler.sample(self.num_tries)
        query_values = self.query_sampler.pool.get_elements_with_index(queries)
        b = self.selection_criteria.score_queries(query_values)
        b = tf.map_fn(self.function, b)

        selected_queries = tf.gather(queries, tf.math.top_k(b).indices)

        return selected_queries
