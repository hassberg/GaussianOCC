import os

import pandas as pd
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.experiments.experiment_runner import ExperimentRunner

from _experiment_pipeline.base_blueprint import BaseBlueprint
from datasources.parametrized_data_source import ParametrizedDataSource
from evaluation.matthew_correlation_coefficient.mcc_eval import MccEval
from evaluation.matthew_correlation_coefficient.mcc_train import MccTrain
from evaluation.sm_lengthscale_logger import LengthscaleLogger
from evaluation.sm_noise_logger import NoiseLogger
from evaluation.sm_parameter_logger import SmParameterLogger
from evaluation.sm_vanishing_factor_logger import VanishingLogger
from gridsearch.paramaeter_gridsearch import get_best_parameter
from models.constant_prior_gp_model.constant_prior_mean_surrogate_model import ConstantPriorMeanSurrogateModel
from models.prior_knowledge_model_gp_model.custom_model_based_prior_mean_surrogate_model import CustomModelBasedPriorMeanSurrogateModel
from models.self_training_prior_knowledge_model_gp_model.self_training_custom_model_based_prior_mean_surrogate_model import SelfTrainingCustomModelBasedPriorMeanSurrogateModel
from models.vanishing_self_training_prior_knowledge_model_gp_model.vanishing_self_training_custom_model_based_prior_mean_surrogate_model import \
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel

gp_models = [
    CustomModelBasedPriorMeanSurrogateModel,
]

ls_learning_gp = [
    SelfTrainingCustomModelBasedPriorMeanSurrogateModel,
    ConstantPriorMeanSurrogateModel,
]

vanishing_model = [
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel,
]


def run_discrete_experiment(arg_map, best_k, repeats, learning_steps):
    fitting_results = get_best_parameter(arg_map)

    for i in range(best_k):
        for data_sample in arg_map["data_samples"]["samples"]:
            train_set = pd.read_csv(os.path.join(data_sample, "train.csv")).values
            eval_set = pd.read_csv(os.path.join(data_sample, "test.csv")).values

            # loop over samples
            sm_args = fitting_results[i][1]
            current_bp = BaseBlueprint
            current_bp.__name__ = arg_map["sm"].__name__ + "BP"

            current_bp.repeat = repeats
            current_bp.learning_steps = learning_steps
            current_bp.data_source = BlueprintElement[ParametrizedDataSource]({'data_points': train_set[:, :-1], 'values': train_set[:, -1]})

            current_bp.surrogate_model = BlueprintElement[arg_map["sm"]](sm_args)
            current_bp.selection_criteria = BlueprintElement[arg_map["sc"]]()
            if arg_map["sm"] in gp_models:
                current_bp.evaluation_metrics = [
                    BlueprintElement[MccTrain](),
                    BlueprintElement[MccEval]({'eval_data': eval_set}),
                    BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
                ]
            elif arg_map["sm"] in ls_learning_gp:
                current_bp.evaluation_metrics = [
                    BlueprintElement[MccTrain](),
                    BlueprintElement[MccEval]({'eval_data': eval_set}),
                    BlueprintElement[LengthscaleLogger](),
                    BlueprintElement[NoiseLogger](),
                    BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
                ]
            elif arg_map["sm"] in vanishing_model:
                current_bp.evaluation_metrics = [
                    BlueprintElement[MccTrain](),
                    BlueprintElement[MccEval]({'eval_data': eval_set}),
                    BlueprintElement[LengthscaleLogger](),
                    BlueprintElement[VanishingLogger](),
                    BlueprintElement[NoiseLogger](),
                    BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
                ]
            else:
                current_bp.evaluation_metrics = [
                    BlueprintElement[MccTrain](),
                    BlueprintElement[MccEval]({'eval_data': eval_set}),
                    BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
                ]

            logfile = os.path.join(arg_map['output_path'],  "log-" + os.path.split(data_sample)[1])
            ExperimentRunner(experiment_blueprints=[current_bp], file=logfile, log=True).run()
