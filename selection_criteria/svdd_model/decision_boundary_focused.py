from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
import tensorflow as tf


class SvddDecisionBoundaryFocusedQuerySelection(SelectionCriteria):

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        result = self.surrogate_model.query(queries)
        return tf.math.multiply(tf.math.abs(result[1]), -1)
