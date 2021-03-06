from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint
from sklearn.metrics import accuracy_score

import tensorflow as tf

from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel


class MccKnownTrain(EvaluationMetric):
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
        self.iteration_scoring.append(1.0)

    def eval(self):
        self.append_round_scoring()

    def get_evaluation(self):
        return '[' + ",".join(str(x) for x in self.iteration_scoring) + ']'

    def append_round_scoring(self):
        points = self.surrogate_model.training_points
        values = self.surrogate_model.training_values

        prediction = self.surrogate_model.query(points)
        class_prediction = tf.map_fn((lambda t: 1 if (t >= 0) else -1), prediction[1])
        self.iteration_scoring.append(accuracy_score(values, class_prediction))
