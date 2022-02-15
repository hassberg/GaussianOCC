from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.maximum_query_optimizer import MaximumQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

from data_sources.BananaDataSource import BananaDataSource
from evaluation.al_learning_curve_focused.active_learning_curve_evaluator import ActiveLearningCurveEvaluator
from evaluation.al_learning_curve_focused.active_learning_curve_metric.basic_active_learning_curve_metric import \
    BasicActiveLearningCurveMetric
from evaluation.al_learning_curve_focused.active_learning_curve_metric.certainty_reached import CertaintyReachedMetric
from evaluation.model_focused.model_evaluation_metrics.mean_development_evaluator import MeanDevelopmentEvaluator
from evaluation.model_focused.model_evaluation_metrics.stddev_development_evaluator import StddevDevelopmentEvaluator
from evaluation.model_focused.model_evaluator import ModelEvaluator
from models.prior_knowledge_gp_model.classifiers.local_outlier_scoring import LocalOutlierFactor
from models.prior_knowledge_gp_model.gaussian_prior_mean_surrogate_model import GaussianPriorMeanSurrogateModel
from selection_criteria.uncertainty_based_query_selection import UncertaintyBasedQuerySelection


class CustomPriorUncertaintyBasedBP(Blueprint):
    repeat = 20

    def __init__(self):
        self.learning_steps = 15
        self.num_knowledge_discovery_queries = 0

        self.data_source = BananaDataSource()
        self.retrievement_strategy = ExactRetrievement()
        self.augmentation_pipeline = NoAugmentation()

        self.interpolation_strategy = FlatMapInterpolation()

        self.instance_level_objective = ConstantInstanceObjective()
        self.instance_cost = ConstantInstanceCost()

        self.surrogate_model = GaussianPriorMeanSurrogateModel(LocalOutlierFactor(k=3))
        self.training_strategy = DirectTrainingStrategy()

        ## important things
        self.surrogate_sampler = RandomContinuousQuerySampler()
        self.query_optimizer = MaximumQueryOptimizer(num_tries=30)
        # TODO here use of surrogate model to rate queries
        self.selection_criteria = UncertaintyBasedQuerySelection()
        ##

        self.knowledge_discovery_sampler = RandomContinuousQuerySampler()
        self.knowledge_discovery_task = NoKnowledgeDiscoveryTask()

        self.evaluation_metrics = [RoundCounterEvaluator(),
                                   ModelEvaluator([StddevDevelopmentEvaluator(), MeanDevelopmentEvaluator()],
                                                  folder_path="plot_out/CustomMean_"),
                                   ActiveLearningCurveEvaluator([BasicActiveLearningCurveMetric(), CertaintyReachedMetric()])]
