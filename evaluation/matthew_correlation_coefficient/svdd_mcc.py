from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint
from sklearn.metrics import matthews_corrcoef

import tensorflow as tf

from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel

from models.common_resource.BaseSVDD import BaseSVDD


class SvddMcc(EvaluationMetric):
    """
    counts the number of rounds. Evaluation is the number of rounds
    """

    def __init__(self, eval_data=None):
        self.all_queries = None
        self.ground_truth = None
        self.svdd_model = None
        self.iteration_scoring = []
        self.end_experiment = None

    def post_init(self, blackboard: Blackboard, blueprint: Blueprint):
        self.svdd_model = blueprint.surrogate_model.gaussian_process_model.mean_module

        # TODO check case if not discrete data
        if blueprint.data_source.possible_queries().is_discrete():
            self.all_queries = blueprint.data_source.possible_queries().get_all_elements()
            self.ground_truth = blueprint.data_source.query(tf.convert_to_tensor(range(len(self.all_queries))))[1]

        self.append_round_scoring()

    def eval(self):
        self.append_round_scoring()

    def get_evaluation(self):
        return '[' + ",".join(str(x) for x in self.iteration_scoring) + ']'

    def append_round_scoring(self):
        prediction = self.svdd_model.model.predict(self.all_queries.numpy())
        self.iteration_scoring.append(matthews_corrcoef(self.ground_truth, prediction))