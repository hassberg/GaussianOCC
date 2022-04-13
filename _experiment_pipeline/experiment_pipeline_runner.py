import os
import sys

tail, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tail)

from _experiment_pipeline.dataset_reader import get_data_samples_dictionary
from _experiment_pipeline.single_experiment_run import run_single_experiment

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

import multiprocessing as mp
from tqdm import tqdm

from models.constant_prior_gp_model.constant_prior_mean_surrogate_model import ConstantPriorMeanSurrogateModel
from models.prior_knowledge_model_gp_model.custom_model_based_prior_mean_surrogate_model import CustomModelBasedPriorMeanSurrogateModel
from models.svdd_neg.svdd_neg_surrogate_model import SVDDNegSurrogateModel
from models.self_training_prior_knowledge_model_gp_model.self_training_custom_model_based_prior_mean_surrogate_model import SelfTrainingCustomModelBasedPriorMeanSurrogateModel
from models.vanishing_self_training_prior_knowledge_model_gp_model.vanishing_self_training_custom_model_based_prior_mean_surrogate_model import \
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel
## Selection criteria
from selection_criteria.gp_model.decision_boundary_focused_query_selection import GpDecisionBoundaryFocusedQuerySelection
from selection_criteria.gp_model.uncertainty_based_query_selection import UncertaintyBasedQuerySelection
from selection_criteria.svdd_model.decision_boundary_focused import SvddDecisionBoundaryFocusedQuerySelection
from selection_criteria.svdd_model.random_outlier_sample import RandomOutlierSamplingSelectionCriteria

experiment_repeats: int = 2
learning_steps: int = 4
best_k_to_score: int = 2

## List of surrogate models to use for evaluation
available_surrogate_models = [
    # CustomModelBasedPriorMeanSurrogateModel,
    SelfTrainingCustomModelBasedPriorMeanSurrogateModel,
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel,
    # SVDDNegSurrogateModel,
    # ConstantPriorMeanSurrogateModel,
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
    run_single_experiment(arg_map=arg_map, best_k=best_k_to_score, repeats=experiment_repeats, learning_steps=learning_steps)


if __name__ == '__main__':
    N = mp.cpu_count()

    # getting available data csv files
    data_samples = get_data_samples_dictionary(tail=tail)

    all_experiments = []
    for sample_mode, data_set, sampels in [(key, sample, data_samples[key][sample]) for key in data_samples.keys() for sample in data_samples[key].keys()]:
        for sm in available_surrogate_models:
            for sc in available_selection_criteria.get(sm):

                output_path = os.path.join("output", sample_mode, data_set, sm.__name__, sc.__name__)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                all_experiments.append({"sm": sm, "sc": sc, "sampling_mode": sample_mode, "data_samples": sampels, "output_path": output_path})

    if N == 32:
        with mp.Pool(processes=N) as p:
            for _ in tqdm(p.imap_unordered(run_experiment, all_experiments), total=len(all_experiments)):
                pass
    else:
        for arg_map in all_experiments:
            run_experiment(arg_map)
