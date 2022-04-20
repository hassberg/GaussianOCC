import os

import pandas as pd
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from active_learning_ts.query_selection.query_optimizers.maximum_query_optimizer import MaximumQueryOptimizer

from _experiment_pipeline.base_blueprint_continuous import ContinuousBaseBlueprint
from _experiment_pipeline.base_blueprint_discrete import DiscreteBaseBlueprint
from datasources.parametrized_continuous_data_source import ParametrizedContinuousDataSource
from datasources.parametrized_discrete_data_source import ParametrizedDiscreteDataSource
from evaluation.gp_uncertainty_eval import GpUncertainty
from evaluation.ground_truth_logger import GroundTruthLogger
from evaluation.matthew_correlation_coefficient.mcc_eval import MccEval
from evaluation.matthew_correlation_coefficient.mcc_know_train import MccKnownTrain
from evaluation.matthew_correlation_coefficient.mcc_train import MccTrain
from evaluation.sm_lengthscale_logger import LengthscaleLogger
from evaluation.sm_noise_logger import NoiseLogger
from evaluation.sm_parameter_logger import SmParameterLogger
from evaluation.sm_vanishing_factor_logger import VanishingLogger
from evaluation.svdd_uncertainty_eval import SvddUncertainty
from gridsearch.paramaeter_gridsearch import get_best_parameter
from models.constant_prior_gp_model.constant_prior_mean_surrogate_model import ConstantPriorMeanSurrogateModel
from models.prior_knowledge_model_gp_model.custom_model_based_prior_mean_surrogate_model import CustomModelBasedPriorMeanSurrogateModel
from models.self_training_prior_knowledge_model_gp_model.self_training_custom_model_based_prior_mean_surrogate_model import SelfTrainingCustomModelBasedPriorMeanSurrogateModel
from models.vanishing_self_training_prior_knowledge_model_gp_model.vanishing_self_training_custom_model_based_prior_mean_surrogate_model import \
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel

gp_models = [
    CustomModelBasedPriorMeanSurrogateModel,
    ConstantPriorMeanSurrogateModel,
]

ls_learning_gp = [
    SelfTrainingCustomModelBasedPriorMeanSurrogateModel,
]

vanishing_model = [
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel,
]


def get_evaluation_metrics(arg_map, eval_set, i, sm_args):
    if arg_map["sm"] in gp_models:
        eval_metrics = [
            BlueprintElement[MccTrain](),
            # BlueprintElement[MccKnownTrain](),
            BlueprintElement[MccEval]({'eval_data': eval_set}),
            BlueprintElement[GpUncertainty]({'eval_data': eval_set}),
            BlueprintElement[GroundTruthLogger]({'eval_data': eval_set}),
            BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
        ]
    elif arg_map["sm"] in ls_learning_gp:
        eval_metrics = [
            BlueprintElement[MccTrain](),
            # BlueprintElement[MccKnownTrain](),
            BlueprintElement[MccEval]({'eval_data': eval_set}),
            BlueprintElement[GpUncertainty]({'eval_data': eval_set}),
            BlueprintElement[GroundTruthLogger]({'eval_data': eval_set}),
            BlueprintElement[LengthscaleLogger](),
            BlueprintElement[NoiseLogger](),
            BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
        ]
    elif arg_map["sm"] in vanishing_model:
        eval_metrics = [
            BlueprintElement[MccTrain](),
            BlueprintElement[MccKnownTrain](),
            BlueprintElement[MccEval]({'eval_data': eval_set}),
            BlueprintElement[LengthscaleLogger](),
            BlueprintElement[VanishingLogger](),
            BlueprintElement[NoiseLogger](),
            BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
        ]
    else:
        eval_metrics = [
            BlueprintElement[MccTrain](),
            BlueprintElement[MccEval]({'eval_data': eval_set}),
            BlueprintElement[SvddUncertainty]({'eval_data': eval_set}),
            BlueprintElement[GroundTruthLogger]({'eval_data': eval_set}),
            BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
        ]
    return eval_metrics


def run_single_experiment(arg_map, best_k, repeats, learning_steps):
    if arg_map["sampling_mode"] == "continuous":
        ground_truth = pd.read_csv(os.path.join(arg_map["data_samples"]["ground_truth"], "ground_truth.csv")).values
        arg_map["gt"] = ground_truth

    fitting_results = get_best_parameter(arg_map)

    for i in range(best_k):
        for data_sample in arg_map["data_samples"]["samples"]:
            train_set = pd.read_csv(os.path.join(data_sample, "train.csv")).values
            eval_set = pd.read_csv(os.path.join(data_sample, "test.csv")).values

            sm_args = dict(fitting_results[i][1])
            sm_args["sampling_mode"] = arg_map["sampling_mode"]
            sm_args["init_points"] = train_set[:, :-1]
            sm_args["ground_truth"] = train_set[:, -1]

            if arg_map["sampling_mode"] == "discrete":
                current_bp = DiscreteBaseBlueprint
                current_bp.data_source = BlueprintElement[ParametrizedDiscreteDataSource]({'data_points': train_set[:, :-1], 'values': train_set[:, -1]})
            elif arg_map["sampling_mode"] == "continuous":
                current_bp = ContinuousBaseBlueprint
                current_bp.data_source = BlueprintElement[ParametrizedContinuousDataSource]({'data_points': ground_truth[:, :-1], 'values': ground_truth[:, -1]})
                #TODO current_bp.query_optimizer = BlueprintElement[MaximumQueryOptimizer]({"num_tries": arg_map['poolsize']})
            else:
                raise Exception("Unknown sampling mode: " + arg_map["sampling_mode"])

            current_bp.__name__ = arg_map["sm"].__name__ + "BP"
            current_bp.surrogate_model = BlueprintElement[arg_map["sm"]](sm_args)

            current_bp.repeat = repeats
            current_bp.learning_steps = learning_steps

            current_bp.selection_criteria = BlueprintElement[arg_map["sc"]]()
            current_bp.evaluation_metrics = get_evaluation_metrics(arg_map=arg_map, eval_set=eval_set, i=i, sm_args=fitting_results[i][1])

            logfile = os.path.join(arg_map['output_path'], str(i) + "-log-" + os.path.split(data_sample)[1])
            ExperimentRunner(experiment_blueprints=[current_bp], file=logfile, log=True).run()
