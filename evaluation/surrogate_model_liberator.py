from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.logging.data_blackboard import Blackboard
from gridsearch.surrogate_receiver import SurrogateReceiver


class SurrogateModelLiberator(EvaluationMetric):

    def __init__(self, base_estimator: SurrogateReceiver):
        self.base_estimator = base_estimator
        self.end_experiment = None

    def post_init(self, blackboard: Blackboard, blueprint: Blueprint):
        self.base_estimator.save_surrogate_model(blueprint.surrogate_model)
