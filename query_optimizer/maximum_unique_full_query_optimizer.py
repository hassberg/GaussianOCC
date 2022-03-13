from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel

import tensorflow as tf


class MaximumUniqueFullQueryOptimizer(QueryOptimizer):
    # Optimizer, which selects of all available data sample the best scoring one which hasn't been sampled so far

    def __init__(self):
        self.max_samples = None
        self.sampled_queries = []
        self.function = lambda x: tf.reduce_max(x)

    def post_init(self, surrogate_model: SurrogateModel,
                  selection_criteria: SelectionCriteria,
                  query_sampler: QuerySampler
                  ):
        self.surrogate_model = surrogate_model
        self.selection_criteria = selection_criteria
        self.query_sampler = query_sampler
        self.max_samples = surrogate_model.value_shape[0]

    def optimize_query_candidates(self):
        queries = self.query_sampler.sample(self.max_samples)
        query_values = self.query_sampler.pool.get_elements_with_index(queries)
        b = self.selection_criteria.score_queries(query_values)

        # adds penalty already queries, already sampled
        scorings = b.numpy()
        new_min = scorings.min() - 1
        for i in self.sampled_queries:
            scorings[tf.where(tf.equal(i, queries)).numpy().flatten()] = new_min

        b = tf.convert_to_tensor(scorings)
        b = tf.map_fn(self.function, b)
        selected_queries = tf.gather(queries, tf.math.top_k(b).indices)

        self.sampled_queries.extend(selected_queries.numpy())
        return selected_queries
