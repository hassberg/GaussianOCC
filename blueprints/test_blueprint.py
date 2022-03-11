from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.discover_tasks.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.maximum_query_optimizer import MaximumQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

from datasources.BananaDataSource import BananaDataSource
from evaluation.al_learning_curve_focused.active_learning_curve_evaluator import ActiveLearningCurveEvaluator
from evaluation.al_learning_curve_focused.active_learning_curve_metric.basic_active_learning_curve_metric import \
    BasicActiveLearningCurveMetric
from evaluation.al_learning_curve_focused.active_learning_curve_metric.mcc_score_learning_curve_metric import \
    MccScoreLearningCurveMetric
from evaluation.model_focused.model_evaluation_metrics.mean_development_evaluator import MeanDevelopmentEvaluator
from evaluation.model_focused.model_evaluation_metrics.stddev_development_evaluator import StddevDevelopmentEvaluator
from evaluation.model_focused.model_evaluator import ModelEvaluator
from models.constant_prior_gp_model.constant_prior_mean_surrogate_model import ConstantPriorMeanSurrogateModel
from selection_criteria.gp_model.uncertainty_based_query_selection import UncertaintyBasedQuerySelection


class TestBlueprint(Blueprint):
    repeat = 1

    def __init__(self):
        self.learning_steps = 20
        self.num_knowledge_discovery_queries = 0

        self.data_source = BananaDataSource()
        self.retrievement_strategy = ExactRetrievement()
        self.augmentation_pipeline = NoAugmentation()

        self.interpolation_strategy = FlatMapInterpolation()

        self.instance_level_objective = ConstantInstanceObjective()
        self.instance_cost = ConstantInstanceCost()

        # self.surrogate_model = CustomModelBasedPriorMeanSurrogateModel()
        self.surrogate_model = ConstantPriorMeanSurrogateModel()
        self.training_strategy = DirectTrainingStrategy()

        ## important things
        self.surrogate_sampler = RandomContinuousQuerySampler()
        self.query_optimizer = MaximumQueryOptimizer(num_tries=1000)
        # TODO here use of surrogate model to rate queries
        self.selection_criteria = UncertaintyBasedQuerySelection()
        ##

        self.knowledge_discovery_sampler = RandomContinuousQuerySampler()
        self.knowledge_discovery_task = NoKnowledgeDiscoveryTask()
        self.evaluation_metrics = [ModelEvaluator([StddevDevelopmentEvaluator(), MeanDevelopmentEvaluator()],
                                                  folder_path="main_"),
                                   ActiveLearningCurveEvaluator(
                                       [BasicActiveLearningCurveMetric(), MccScoreLearningCurveMetric()])]
