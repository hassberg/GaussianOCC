from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
import tensorflow as tf

from evaluation.al_learning_curve_focused.learning_curve_metric import LearningCurveMetric
from typing import List


class ActiveLearningCurveEvaluator(EvaluationMetric):

    def __init__(self, learning_curve_evaluation_metrics: List[LearningCurveMetric]):
        self.learning_curve_evaluation_metrics: List[LearningCurveMetric] = learning_curve_evaluation_metrics
        self.query_points = None
        self.ground_truth = []
        self.surrogate_model: SurrogateModel = None
        self.curve_scoring = None
        self.round_counter = 0
        self.iteration_scoring = []

    def post_init(self, blackboard: Blackboard, blueprint):
        self.surrogate_model: SurrogateModel = blueprint.surrogate_model

        # TODO this is for a discrete data set
        if blueprint.data_source.possible_queries().is_discrete():
            self.query_points = blueprint.data_source.possible_queries().get_all_elements()
            self.ground_truth.append(
                blueprint.data_source.query(tf.convert_to_tensor(range(len(self.query_points))))[1])

        for x in self.learning_curve_evaluation_metrics:
            x.post_init(self.ground_truth[0])

        self.iteration_scoring.append(
            (tf.convert_to_tensor(blueprint.data_source.query(tf.convert_to_tensor(range(len(self.query_points))))[1]),
             tf.convert_to_tensor(blueprint.data_source.query(tf.convert_to_tensor(range(len(self.query_points))))[0])))
        round_scoring = self.get_round_scoring()
        self.curve_scoring = [(self.round_counter, round_scoring)]

        self.round_counter += 1

    def eval(self):
        round_scoring = self.get_round_scoring()
        self.curve_scoring.append((self.round_counter, round_scoring))
        self.round_counter += 1

    def get_evaluation(self):
        self.save_iteration_scoring()

        out = []
        f = lambda x: '[' + ', '.join([str(a) for a in x]) + ']' if isinstance(x, list) else str(x)
        [out.append('"' + type(self.learning_curve_evaluation_metrics[i]).__name__ + '" : ' + f(
            self.learning_curve_evaluation_metrics[i].evaluate_learning_curve(
                tf.reshape(tf.convert_to_tensor([x[1][i] for x in self.curve_scoring]),
                           shape=(len(self.curve_scoring))))))
         for i in
         range(len(self.learning_curve_evaluation_metrics))]

        mean_iteration = []
        [mean_iteration.append('"' + str(i) + '" :' + f(self.iteration_scoring[i][0].numpy().flatten().tolist())) for i
         in
         range(len(self.iteration_scoring))]

        stddev_iteration = []
        [stddev_iteration.append('"' + str(i) + '" :' + f(self.iteration_scoring[i][1].numpy().flatten().tolist())) for
         i in
         range(len(self.iteration_scoring))]

        f = lambda x: '{' + ',\n '.join([str(a) for a in x]) + '}' if isinstance(x, list) else str(x)
        out.append('"iteration_scoring" : \n {' + '\n "mean" : ' + f(mean_iteration) + ',\n "stddev" : ' + f(
            stddev_iteration) + '}')

        return '{\n' + ',\n'.join(out) + '\n}\n'

    def get_round_scoring(self):
        round_scoring = []
        scoring = self.surrogate_model.query(self.query_points)[1], self.surrogate_model.uncertainty(self.query_points)

        for i in range(len(self.learning_curve_evaluation_metrics)):
            round_scoring.append(self.learning_curve_evaluation_metrics[i].calculate_curve_step(scoring))

        return round_scoring

    def save_iteration_scoring(self):
        scoring = self.surrogate_model.query(self.query_points)[1], self.surrogate_model.uncertainty(self.query_points)
        self.iteration_scoring.append(scoring)

