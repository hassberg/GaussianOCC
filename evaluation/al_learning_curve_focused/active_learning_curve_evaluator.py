from active_learning_ts.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel

from evaluation.al_learning_curve_focused.learning_curve_metric import LearningCurveMetric
from typing import List


class ActiveLearningCurveEvaluator(EvaluationMetric):

    def __init__(self, learning_curve_evaluation_metrics: List[LearningCurveMetric]):
        self.learning_curve_evaluation_metrics: List[LearningCurveMetric] = learning_curve_evaluation_metrics
        self.query_points = None
        self.ground_truth = None
        self.surrogate_model: SurrogateModel = None
        self.curve_scoring = None
        self.round_counter = 0

    def post_init(self, blackboard: Blackboard, blueprint):
        self.surrogate_model: SurrogateModel = blueprint.surrogate_model

        # TODO this is for a discrete data set
        if blueprint.data_source.possible_queries().is_discrete():
            self.query_points = blueprint.data_source.possible_queries().get_all_elements()

        self.ground_truth = self.surrogate_model.query(self.query_points)[1], self.surrogate_model.uncertainty(self.query_points)
        round_scoring = self.get_round_scoring()
        self.curve_scoring = [(self.round_counter, round_scoring)]

        self.round_counter += 1

    def eval(self):
        round_scoring = self.get_round_scoring()
        self.curve_scoring.append((self.round_counter, round_scoring))
        self.round_counter += 1

    def get_evaluation(self):
        # 4 eval criteria for each metric
        pass

    def get_round_scoring(self):
        round_scoring = []
        scoring = self.surrogate_model.query(self.query_points)[1], self.surrogate_model.uncertainty(self.query_points)

        for i in range(len(self.learning_curve_evaluation_metrics)):
            round_scoring.append(self.learning_curve_evaluation_metrics[i].calculate_curve_step(scoring))

        return round_scoring
