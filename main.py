import os,sys

from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from test_blueprint import TestBlueprint


tail, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tail)


runner = ExperimentRunner(experiment_blueprints=[TestBlueprint], log=True)
runner.run()
