from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
import tensorflow as tf


class VarianceBasedQuerySelection(SelectionCriteria):

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        return self.surrogate_model.uncertainty(queries).detach()
