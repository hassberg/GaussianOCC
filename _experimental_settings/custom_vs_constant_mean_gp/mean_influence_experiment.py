from _experimental_settings.custom_vs_constant_mean_gp.mean_experiment.custom_prior_mean_based import \
    CustomPriorMeanBasedBP
from _experimental_settings.custom_vs_constant_mean_gp.mean_experiment.custom_prior_uncertainty_based import \
    CustomPriorUncertaintyBasedBP
from _experimental_settings.custom_vs_constant_mean_gp.mean_experiment.constant_prior_mean_based import \
    ConstantPriorMeanBasedBP
from _experimental_settings.custom_vs_constant_mean_gp.mean_experiment.constant_prior_uncertainty_based import \
    ConstantPriorUncertaintyBasedBP

from active_learning_ts.experiments.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    experiment_blueprints=[ConstantPriorUncertaintyBasedBP,
                           ConstantPriorMeanBasedBP,
                           CustomPriorUncertaintyBasedBP,
                           CustomPriorMeanBasedBP], log=True)
runner.run()
