import os
import sys

import pandas as pd
from sklearn.model_selection import PredefinedSplit, GridSearchCV

from datasources.parametrized_data_source import ParametrizedDataSource

from evaluation.matthew_correlation_coefficient.mcc_eval import MccEval
from evaluation.matthew_correlation_coefficient.mcc_test import MccTest
from evaluation.matthew_correlation_coefficient.mcc_train import MccTrain
from evaluation.sm_parameter_logger import SmParameterLogger
from gridsearch_handler.grid_search_blueprint_base_estimator import GridSearchBlueprintBaseEstimator

tail, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tail)

import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.experiments.experiment_runner import ExperimentRunner

from _experiment_pipeline.base_blueprint import BaseBlueprint
## Data Source
from datasources.csv_file_reading_data_source import CsvFileReadingDataSource
## Models
from models.constant_prior_gp_model.constant_prior_mean_surrogate_model import ConstantPriorMeanSurrogateModel
from models.prior_knowledge_model_gp_model.custom_model_based_prior_mean_surrogate_model import \
    CustomModelBasedPriorMeanSurrogateModel
from models.svdd_neg.svdd_neg_surrogate_model import SVDDNegSurrogateModel
from selection_criteria.gp_model.decision_boundary_focused_query_selection import \
    GpDecisionBoundaryFocusedQuerySelection
## Selection criteria
from selection_criteria.gp_model.uncertainty_based_query_selection import UncertaintyBasedQuerySelection
from selection_criteria.gp_model.variance_based_query_selection import VarianceBasedQuerySelection
from selection_criteria.svdd_model.decision_boundary_focused import SvddDecisionBoundaryFocusedQuerySelection
from selection_criteria.svdd_model.random_outlier_sample import RandomOutlierSamplingSelectionCriteria

experiment_repeats: int = 3
learning_steps: int = 3
best_k_to_score: int = 3

## List of surrogate models to use for evaluation
available_surrogate_models = [  # CustomModelBasedPriorMeanSurrogateModel,
    SVDDNegSurrogateModel,
    #                              ConstantPriorMeanSurrogateModel,
]

# Dictionary containing selection criteria for each surrogate model
available_selection_criteria = {
    CustomModelBasedPriorMeanSurrogateModel: [UncertaintyBasedQuerySelection, GpDecisionBoundaryFocusedQuerySelection],
    # , VarianceBasedQuerySelection],
    ConstantPriorMeanSurrogateModel: [UncertaintyBasedQuerySelection, GpDecisionBoundaryFocusedQuerySelection],
    # , VarianceBasedQuerySelection],
    SVDDNegSurrogateModel: [RandomOutlierSamplingSelectionCriteria]  # , SvddDecisionBoundaryFocusedQuerySelection]
}


# function to run blueprint with parameters in parallel
def run_experiment(arg_map):
    train_set = pd.read_csv(os.path.join(arg_map["file"], "train.csv")).values
    test_set = pd.read_csv(os.path.join(arg_map["file"], "test.csv")).values
    eval_set = pd.read_csv(os.path.join(arg_map["file"], "eval.csv")).values

    data = np.concatenate((train_set[:, :-1], test_set[:, :-1]), axis=0)
    targets = np.concatenate((train_set[:, -1], test_set[:, -1]), axis=0)
    indices = np.append(np.full((len(train_set)), -1, dtype=int), np.full((len(test_set)), 0, dtype=int))
    ps = PredefinedSplit(indices)

    arg_map["learning_steps"] = learning_steps
    base_estimator = GridSearchBlueprintBaseEstimator(blueprint_parameter=arg_map)

    parameter = {
        "kernel": ['rbf'],
        "gamma": [0.5, 1, 2],
        "C": [0.5, 1, 2],
    }
    clf = GridSearchCV(estimator=base_estimator, param_grid=parameter, cv=ps, refit=False)
    fit = clf.fit(data, targets)

    results = list(map(list, zip(fit.cv_results_['mean_test_score'], fit.cv_results_['params'])))

    results.sort(key=lambda x: x[0], reverse=True)

    for i in range(best_k_to_score):
        sm_args = results[i][1]
        current_bp = BaseBlueprint
        current_bp.__name__ = arg_map["sm"].__name__ + "BP"

        current_bp.repeat = experiment_repeats
        current_bp.learning_steps = learning_steps
        current_bp.data_source = BlueprintElement[ParametrizedDataSource](
            {'data_points': train_set[:, :-1], 'values': train_set[:, -1]})

        current_bp.surrogate_model = BlueprintElement[arg_map["sm"]](sm_args)
        current_bp.selection_criteria = BlueprintElement[arg_map["sc"]]()

        current_bp.evaluation_metrics = [
            BlueprintElement[MccTrain](),
            BlueprintElement[MccTest]({'eval_data': test_set}),
            BlueprintElement[MccEval]({'eval_data': eval_set}),
            BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
        ]

        logfile = os.path.join(arg_map['log'][0], arg_map['log'][1] + "_" + "log-" + str(i))
        ExperimentRunner(experiment_blueprints=[current_bp], file=logfile, log=True).run()


if __name__ == '__main__':
    N = mp.cpu_count()

    # getting available data csv files
    root = os.path.join(tail, '_experiment_pipeline', 'data_sets')
    pattern = "_datasample"
    datasamples = list(filter(lambda x: x.endswith(pattern), [path for path, _, _ in os.walk(root)]))

    all_experiments = []
    for sm in available_surrogate_models:
        for sc in available_selection_criteria.get(sm):

            output_path = os.path.join("output", sm.__name__, sc.__name__)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for sample in datasamples:
                _, filename = os.path.split(sample)
                logfile_name = [output_path, filename.split('_')[0] + "_" + filename.split('_')[2]]
                all_experiments.append({"sm": sm, "sc": sc, "file": sample, "log": logfile_name})

    # with mp.Pool(processes=1) as p: # TODO scale N correctly
    #     for _ in tqdm(p.imap_unordered(run_experiment, all_experiments), total=len(all_experiments)):
    #         pass

    for arg_map in all_experiments:
        run_experiment(arg_map)
