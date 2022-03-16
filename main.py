from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from test_blueprint import TestBlueprint

runner = ExperimentRunner(experiment_blueprints=[TestBlueprint], log=True)
runner.run()
