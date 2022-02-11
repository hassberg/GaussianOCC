import time
from typing import List, Tuple

from active_learning_ts.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
from matplotlib.backends.backend_pdf import PdfPages

from evaluation.model_evaluation_metric import ModelEvaluationMetric


# extends evaluator, saves for query steps the results of the model..
class ModelEvaluator(EvaluationMetric):
    def __init__(self, model_evaluation_metrics: List[ModelEvaluationMetric], folder_path):
        self.model_evaluation_metrics = model_evaluation_metrics
        self.query_points = None
        self.prediction_list: List[Tuple] = []
        self.surrogate_model: SurrogateModel = None
        self.folder_path = folder_path

    def post_init(self, blackboard: Blackboard, blueprint: Blueprint):
        self.surrogate_model = blueprint.surrogate_model

        # TODO this is for a discrete data set
        if blueprint.data_source.possible_queries().is_discrete():
            self.query_points = blueprint.data_source.possible_queries().get_all_elements()

    def eval(self) -> None:
        if self.query_points is not None:
            _, mean = self.surrogate_model.query(self.query_points)
            stddev = self.surrogate_model.uncertainty(self.query_points)
            self.prediction_list.append((mean, stddev))

    def get_evaluation(self):
        if self.query_points is not None:
            millis = int(time.time() * 1000)
            file_name = self.folder_path + str(millis) + ".pdf"
            with PdfPages(file_name) as pdf:
                for m in self.model_evaluation_metrics:
                    m.evaluate_model(self.query_points, self.prediction_list, pdf)
