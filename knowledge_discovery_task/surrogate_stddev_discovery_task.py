from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
import tensorflow as tf


class SurrogateStdDevDiscoveryTask(KnowledgeDiscoveryTask):

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return self.surrogate_model.uncertainty(points).detach()
