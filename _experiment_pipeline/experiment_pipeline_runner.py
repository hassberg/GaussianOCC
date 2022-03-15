import os
import sys
tail, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tail)

import multiprocessing as mp
from tqdm import tqdm

from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.experiments.experiment_runner import ExperimentRunner

from _experiment_pipeline.base_blueprint import BaseBlueprint
## Data Source
from datasources.csvFileReadingDataSource import CsvFileReadingDataSource
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
learning_steps: int = 20

## List of surrogate models to use for evaluation
available_surrogate_models = [CustomModelBasedPriorMeanSurrogateModel, SVDDNegSurrogateModel,
                              ConstantPriorMeanSurrogateModel]

# Dictionary containing selection criteria for each surrogate model
available_selection_criteria = {
    CustomModelBasedPriorMeanSurrogateModel: [UncertaintyBasedQuerySelection, GpDecisionBoundaryFocusedQuerySelection],
    # , VarianceBasedQuerySelection],
    ConstantPriorMeanSurrogateModel: [UncertaintyBasedQuerySelection, GpDecisionBoundaryFocusedQuerySelection],
    # , VarianceBasedQuerySelection],
    SVDDNegSurrogateModel: [RandomOutlierSamplingSelectionCriteria, SvddDecisionBoundaryFocusedQuerySelection]
}


# function to run blueprint with parameters in parallel
def run_experiment(arg_map):
    current_bp = BaseBlueprint
    current_bp.__name__ = arg_map["sm"].__name__ + "BP"

    current_bp.repeat = experiment_repeats
    current_bp.learning_steps = learning_steps
    current_bp.data_source = BlueprintElement[CsvFileReadingDataSource]({'fileName': arg_map["file"]})

    current_bp.surrogate_model = BlueprintElement[arg_map["sm"]]()
    current_bp.selection_criteria = BlueprintElement[arg_map["sc"]]()
    ExperimentRunner(experiment_blueprints=[current_bp], file=arg_map["log"], log=True).run()


if __name__ == '__main__':
    N = mp.cpu_count()

    # getting available data csv files
    files = []
    root = os.path.join(tail, '_experiment_pipeline', 'data_sets')
    pattern = "*.csv"
    files = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]

    all_experiments = []
    for sm in available_surrogate_models:
        for sc in available_selection_criteria.get(sm):

            output_path = os.path.join("output", sm.__name__, sc.__name__)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for file in files:
                _, filename = os.path.split(file)
                logfile_name = os.path.join(output_path,
                                            ("log_" + filename.split('_')[0] + "_" + filename.split('_')[2]))
                all_experiments.append({"sm": sm, "sc": sc, "file": file, "log": logfile_name})

    with mp.Pool(processes=int(N/2)) as p: # TODO scale N correctly
        for _ in tqdm(p.imap_unordered(run_experiment, all_experiments), total=len(all_experiments)):
            pass
