import os

from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.experiments.experiment_runner import ExperimentRunner

from _experiment_pipeline.base_blueprint import BaseBlueprint
## Data Source
from datasources.csvFileReadingDataSource import CsvFileReadingDataSource
## Models
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
available_surrogate_models = [CustomModelBasedPriorMeanSurrogateModel, SVDDNegSurrogateModel]

# Dictionary containing selection criteria for each surrogate model
available_selection_criteria = {
    CustomModelBasedPriorMeanSurrogateModel: [UncertaintyBasedQuerySelection, VarianceBasedQuerySelection,
                                              GpDecisionBoundaryFocusedQuerySelection],
    SVDDNegSurrogateModel: [RandomOutlierSamplingSelectionCriteria, SvddDecisionBoundaryFocusedQuerySelection]
}

# getting available data csv files
files = []
root = 'data_sets'
pattern = "*.csv"
files = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]

for sm in available_surrogate_models:
    print("Testing surrogate model: " + sm.__name__)

    for sc in available_selection_criteria.get(sm):
        print(" - With selection criteria: " + sc.__name__)

        output_path = os.path.join("output", sm.__name__, sc.__name__)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file in files:
            current_bp = BaseBlueprint
            current_bp.__name__ = sm.__name__ + "BP"

            current_bp.repeat = experiment_repeats
            current_bp.learning_steps = learning_steps
            current_bp.data_source = BlueprintElement[CsvFileReadingDataSource]({'fileName': file})

            current_bp.surrogate_model = BlueprintElement[sm]()
            current_bp.selection_criteria = BlueprintElement[sc]()

            _, filename = os.path.split(file)
            logfile_name = os.path.join(output_path, ("log_" + filename.split('_')[0] + "-" + filename.split('_')[2]))
            print("   - With data set: " + filename.split('_')[0] + "-" + filename.split('_')[2])
            ExperimentRunner(experiment_blueprints=[current_bp], file=logfile_name, log=True).run()
