from statistics import mean
from typing import List

import tensorflow as tf
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.experiments.experiment_runner import ExperimentRunner

from datasources.parametrized_data_source import ParametrizedDataSource
from evaluation.surrogate_model_liberator import SurrogateModelLiberator
from gridsearch_handler.grid_search_base_blueprint import GridSearchBaseBlueprint
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
from sklearn.base import BaseEstimator
from sklearn.metrics import matthews_corrcoef, f1_score

from gridsearch_handler.surrogate_receiver import SurrogateReceiver


class GridSearchBlueprintBaseEstimator(BaseEstimator, SurrogateReceiver):
    def __init__(self, blueprint_parameter: dict):
        self.exp_repeat = 1
        self.blueprint_parameter: dict = blueprint_parameter
        self.surrogate_models: List[SurrogateModel] = []
        self.surrogate_model_parameter: dict = None

    def set_params(self, **params):
        self.surrogate_model_parameter = params
        return self

    def score(self, x, y):
        scorings = []
        for sm in self.surrogate_models:
            query_results = sm.query(tf.convert_to_tensor(x))[1]
            prediction = list(map(lambda x: 1 if x > 0 else -1, query_results))
            scoring = f1_score(y, prediction)
            scorings.append(scoring)

        return mean(scorings)

    def fit(self, X, y, **params):
        current_bp = GridSearchBaseBlueprint
        current_bp.__name__ = self.blueprint_parameter["sm"].__name__ + "BP"

        current_bp.repeat = self.exp_repeat
        current_bp.learning_steps = 0# self.blueprint_parameter["learning_steps"]
        current_bp.data_source = BlueprintElement[ParametrizedDataSource]({'data_points': X, 'values': y})

        current_bp.surrogate_model = BlueprintElement[self.blueprint_parameter["sm"]](self.surrogate_model_parameter)
        current_bp.selection_criteria = BlueprintElement[self.blueprint_parameter["sc"]]()
        current_bp.evaluation_metrics = [BlueprintElement[SurrogateModelLiberator]({'base_estimator': self})]

        ExperimentRunner(experiment_blueprints=[current_bp], log=False).run()
        for sm in self.surrogate_models:
            sm.learn(tf.convert_to_tensor(X), tf.convert_to_tensor(y))

    def save_surrogate_model(self, new_sm):
        self.surrogate_models.append(new_sm)
