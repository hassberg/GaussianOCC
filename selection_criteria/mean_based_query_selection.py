from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
import tensorflow as tf


class MeanBasedQuerySelection(SelectionCriteria):

    # abs min scoring as follows
    # max() - mean = scoring
    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        means = self.surrogate_model.query(queries)

        return
