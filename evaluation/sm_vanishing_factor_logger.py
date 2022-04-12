from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint
from sklearn.metrics import matthews_corrcoef

import tensorflow as tf

from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel


class VanishingLogger(EvaluationMetric):
    """
    counts the number of rounds. Evaluation is the number of rounds
    """

    def __init__(self):
        self.all_queries = None
        self.ground_truth = None
        self.surrogate_model: SurrogateModel = None
        self.iteration_scoring = []
        self.end_experiment = None

    def post_init(self, blackboard: Blackboard, blueprint: Blueprint):
        self.surrogate_model = blueprint.surrogate_model
        # TODO check case if not discrete data
        self.append_round_scoring()

    def eval(self):
        self.append_round_scoring()

    def get_evaluation(self):
        fst_lst = [self.iteration_scoring[i] for i in [1, len(self.iteration_scoring)-1]]
        return '[' + ",".join(self.single_log(x) for x in fst_lst) + ']'

    def single_log(self, X):
        return '[' + ",".join(str(x) for x in X) + ']'

    def append_round_scoring(self):
        self.iteration_scoring.append(self.surrogate_model.gaussian_process_model.vf1.T.detach().numpy().flatten())
