from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from blueprints.test_blueprint import TestBlueprint

runner = ExperimentRunner(experiment_blueprints=[TestBlueprint], log=True)
runner.run()
