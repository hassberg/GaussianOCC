import os
import sys

tail, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tail)

import pandas as pd
from evaluation.sm_lengthscale_logger import LengthscaleLogger
from evaluation.sm_noise_logger import NoiseLogger
from models.vanishing_self_training_prior_knowledge_model_gp_model.vanishing_self_training_custom_model_based_prior_mean_surrogate_model import \
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel

from evaluation.matthew_correlation_coefficient.mcc_eval import MccEval
from evaluation.matthew_correlation_coefficient.mcc_test import MccTest
from evaluation.matthew_correlation_coefficient.mcc_train import MccTrain
from evaluation.sm_parameter_logger import SmParameterLogger
from evaluation.sm_vanishing_factor_logger import VanishingLogger
from gridsearch_handler.paramaeter_gridsearch import get_best_parameter
os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

import multiprocessing as mp
from tqdm import tqdm

from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.experiments.experiment_runner import ExperimentRunner

from _experiment_pipeline.base_blueprint import BaseBlueprint
## Data Source
from datasources.parametrized_data_source import ParametrizedDataSource
## Models
from models.constant_prior_gp_model.constant_prior_mean_surrogate_model import ConstantPriorMeanSurrogateModel
from models.prior_knowledge_model_gp_model.custom_model_based_prior_mean_surrogate_model import CustomModelBasedPriorMeanSurrogateModel
from models.svdd_neg.svdd_neg_surrogate_model import SVDDNegSurrogateModel
from models.self_training_prior_knowledge_model_gp_model.self_training_custom_model_based_prior_mean_surrogate_model import SelfTrainingCustomModelBasedPriorMeanSurrogateModel
## Selection criteria
from selection_criteria.gp_model.decision_boundary_focused_query_selection import GpDecisionBoundaryFocusedQuerySelection
from selection_criteria.gp_model.uncertainty_based_query_selection import UncertaintyBasedQuerySelection
from selection_criteria.svdd_model.decision_boundary_focused import SvddDecisionBoundaryFocusedQuerySelection
from selection_criteria.svdd_model.random_outlier_sample import RandomOutlierSamplingSelectionCriteria

experiment_repeats: int = 3
learning_steps: int = 50
best_k_to_score: int = 4

## List of surrogate models to use for evaluation
available_surrogate_models = [
    CustomModelBasedPriorMeanSurrogateModel,
    SelfTrainingCustomModelBasedPriorMeanSurrogateModel,
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel,
    SVDDNegSurrogateModel,
    ConstantPriorMeanSurrogateModel,
]

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

# Dictionary containing selection criteria for each surrogate model
available_selection_criteria = {
    CustomModelBasedPriorMeanSurrogateModel: [
        UncertaintyBasedQuerySelection,
        GpDecisionBoundaryFocusedQuerySelection,
        # VarianceBasedQuerySelection,
    ],
    SelfTrainingCustomModelBasedPriorMeanSurrogateModel: [
        UncertaintyBasedQuerySelection,
        GpDecisionBoundaryFocusedQuerySelection,
        # VarianceBasedQuerySelection,
    ],
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel: [
        UncertaintyBasedQuerySelection,
        GpDecisionBoundaryFocusedQuerySelection,
        # VarianceBasedQuerySelection,
    ],
    ConstantPriorMeanSurrogateModel: [
        UncertaintyBasedQuerySelection,
        GpDecisionBoundaryFocusedQuerySelection,
        # VarianceBasedQuerySelection,
    ],
    SVDDNegSurrogateModel: [
        RandomOutlierSamplingSelectionCriteria,
        SvddDecisionBoundaryFocusedQuerySelection,
    ]
}


def run_experiment(arg_map):
    train_set = pd.read_csv(os.path.join(arg_map["file"], "train.csv")).values
    test_set = pd.read_csv(os.path.join(arg_map["file"], "test.csv")).values
    eval_set = pd.read_csv(os.path.join(arg_map["file"], "eval.csv")).values

    fitting_results = get_best_parameter(arg_map)
    for i in range(best_k_to_score):
        sm_args = fitting_results[i][1]
        current_bp = BaseBlueprint
        current_bp.__name__ = arg_map["sm"].__name__ + "BP"

        current_bp.repeat = experiment_repeats
        current_bp.learning_steps = learning_steps
        current_bp.data_source = BlueprintElement[ParametrizedDataSource]({'data_points': train_set[:, :-1], 'values': train_set[:, -1]})

        current_bp.surrogate_model = BlueprintElement[arg_map["sm"]](sm_args)
        current_bp.selection_criteria = BlueprintElement[arg_map["sc"]]()
        if arg_map["sm"] in gp_models:
            current_bp.evaluation_metrics = [
                BlueprintElement[MccTrain](),
                BlueprintElement[MccTest]({'eval_data': test_set}),
                BlueprintElement[MccEval]({'eval_data': eval_set}),
                BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
            ]
        elif arg_map["sm"] in ls_learning_gp:
            current_bp.evaluation_metrics = [
                BlueprintElement[MccTrain](),
                BlueprintElement[MccTest]({'eval_data': test_set}),
                BlueprintElement[MccEval]({'eval_data': eval_set}),
                BlueprintElement[LengthscaleLogger](),
                BlueprintElement[NoiseLogger](),
                BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
            ]
        elif arg_map["sm"] in vanishing_model:
            current_bp.evaluation_metrics = [
                BlueprintElement[MccTrain](),
                BlueprintElement[MccTest]({'eval_data': test_set}),
                BlueprintElement[MccEval]({'eval_data': eval_set}),
                BlueprintElement[LengthscaleLogger](),
                BlueprintElement[VanishingLogger](),
                BlueprintElement[NoiseLogger](),
                BlueprintElement[SmParameterLogger]({'kth_best': i, 'parameter': sm_args})
            ]
        else:
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
    for sample in datasamples:
        for sm in available_surrogate_models:
            for sc in available_selection_criteria.get(sm):

                output_path = os.path.join("output", sm.__name__, sc.__name__)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)


                _, filename = os.path.split(sample)
                logfile_name = [output_path, filename.split('_')[0] + "_" + filename.split('_')[2]]
                all_experiments.append({"sm": sm, "sc": sc, "file": sample, "log": logfile_name})

    with mp.Pool(processes=N) as p:
        for _ in tqdm(p.imap_unordered(run_experiment, all_experiments), total=len(all_experiments)):
            pass
    #
    # for arg_map in all_experiments:
    #     run_experiment(arg_map)
