from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.max_improvement_query_optimizer import MaximumImprovementQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.knowledge_uncertainty_selection_criteria import KnowledgeUncertaintySelectionCriteria
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

from data_sources.BananaDataSource import BananaDataSource
from knowledge_discovery_task.surrogate_stddev_discovery_task import SurrogateStdDevDiscoveryTask
from prior_knowledge_gp_model.gaussian_prior_mean_surrogate_model import GaussianPriorMeanSurrogateModel
from prior_knowledge_gp_model.classifiers.local_outlier_scoring import LocalOutlierFactor
from evaluation.model_evaluator import ModelEvaluator
from evaluation.model_evaluation_metrics.stddev_development_evaluator import StddevDevelopmentEvaluator
from evaluation.model_evaluation_metrics.mean_development_evaluator import MeanDevelopmentEvaluator

repeat = 2
learning_steps = 30
num_knowledge_discovery_queries = 0

data_source = BananaDataSource()
retrievement_strategy = ExactRetrievement()
augmentation_pipeline = NoAugmentation()

#TODO was genau mach die interpolation strategy? und warum nimmt sie nen 3D array an?
interpolation_strategy = FlatMapInterpolation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = GaussianPriorMeanSurrogateModel(data_source.data_points,
                                                  LocalOutlierFactor(data_source.data_points, k=3))
training_strategy = DirectTrainingStrategy()

surrogate_sampler = RandomContinuousQuerySampler()
query_optimizer = MaximumImprovementQueryOptimizer(num_tries=10)
selection_criteria = KnowledgeUncertaintySelectionCriteria()

knowledge_discovery_sampler = RandomContinuousQuerySampler()
knowledge_discovery_task = SurrogateStdDevDiscoveryTask()

evaluation_metrics = [RoundCounterEvaluator(), ModelEvaluator([StddevDevelopmentEvaluator(), MeanDevelopmentEvaluator()])]
