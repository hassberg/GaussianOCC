from typing import Iterable

from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.interpolation_strategies.flat_map_interpolation import FlatMapInterpolation
from active_learning_ts.data_retrievement.interpolation_strategy import InterpolationStrategy
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.discover_tasks.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy
from active_learning_ts.training.training_strategy import TrainingStrategy

from datasources.csvFileReadingDataSource import CsvFileReadingDataSource
from models.svdd_neg.svdd_neg_surrogate_model import SVDDNegSurrogateModel
from selection_criteria.svdd_model.random_outlier_sample import RandomOutlierSamplingSelectionCriteria
from selection_criteria.svdd_model.decision_boundary_focused import SvddDecisionBoundaryFocusedQuerySelection

from datasources.BananaDataSource import BananaDataSource
from evaluation.mathew_correlation_coefficient import MatthewCorrelationCoefficient
from query_optimizer.maximum_unique_full_query_optimizer import MaximumUniqueFullQueryOptimizer
from query_sampler.full_discrete_query_sampler import FullDiscreteQuerySampler


class TestBlueprint(Blueprint):

    repeat = 2
    learning_steps = 20
    num_knowledge_discovery_queries = 0

    file = '_experiment_pipeline\data_sets\page-blocks\page-blocks-2_sample_1_data-points_346_inlier-fraction_0.95.csv'
    data_source: BlueprintElement[DataSource] = BlueprintElement[CsvFileReadingDataSource]({'fileName': file})

    retrievement_strategy: BlueprintElement[RetrievementStrategy] = BlueprintElement[ExactRetrievement]()
    augmentation_pipeline: BlueprintElement[DataPipeline] = BlueprintElement[NoAugmentation]()
    interpolation_strategy: BlueprintElement[InterpolationStrategy] = BlueprintElement[FlatMapInterpolation]()

    instance_level_objective: BlueprintElement[InstanceObjective] = BlueprintElement[ConstantInstanceObjective]()
    instance_cost: BlueprintElement[InstanceCost] = BlueprintElement[ConstantInstanceCost]()

    surrogate_model: BlueprintElement[SurrogateModel] = BlueprintElement[SVDDNegSurrogateModel]()

    training_strategy: BlueprintElement[TrainingStrategy] = BlueprintElement[DirectTrainingStrategy]()

    surrogate_sampler: BlueprintElement[QuerySampler] = BlueprintElement[RandomContinuousQuerySampler]()

    query_optimizer: BlueprintElement[QueryOptimizer] = BlueprintElement[MaximumUniqueFullQueryOptimizer]()

    selection_criteria: BlueprintElement[SelectionCriteria] = BlueprintElement[SvddDecisionBoundaryFocusedQuerySelection]()

    evaluation_metrics: Iterable[BlueprintElement[EvaluationMetric]] = [BlueprintElement[MatthewCorrelationCoefficient]()]
    knowledge_discovery_sampler: BlueprintElement[QuerySampler] = BlueprintElement[FullDiscreteQuerySampler]()
    knowledge_discovery_task: BlueprintElement[KnowledgeDiscoveryTask] = BlueprintElement[NoKnowledgeDiscoveryTask]()
