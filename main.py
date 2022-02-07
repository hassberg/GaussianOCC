from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from blueprints import custom_mean_gaussian_process_blueprint

runner = ExperimentRunner(experiment_blueprints=[custom_mean_gaussian_process_blueprint], log=True)
runner.run()
