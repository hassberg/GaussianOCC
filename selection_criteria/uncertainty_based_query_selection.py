from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
import tensorflow as tf


class UncertaintyBasedQuerySelection(SelectionCriteria):

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        uncertainty = self.surrogate_model.uncertainty(queries).detach()
        estimation = self.surrogate_model.query(queries)[1].detach()

        scoring = tf.math.multiply(tf.math.divide(tf.math.abs(estimation), tf.sqrt(uncertainty)), -1)
        return scoring
