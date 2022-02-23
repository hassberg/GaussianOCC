from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
import tensorflow as tf
import numpy as np


class RandomOutlierSamplingSelectionCriteria(SelectionCriteria):

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        query_results = self.surrogate_model.query(queries)[1]

        scoring = tf.add(tf.multiply(query_results, -1), np.random.uniform(0, 0.5, (len(query_results),1)))
        return scoring
