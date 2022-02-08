from evaluation.model_evaluation_metric import ModelEvaluationMetric
import matplotlib.pyplot as plt


class MeanDevelopmentEvaluator(ModelEvaluationMetric):
    def evaluate_model(self, data_points, prediction, pdf):
        fig = plt.figure()
        fig.suptitle('development of mean')

        stddev_progress = []
        for i in range(len(prediction)):
            stddev_progress.append(prediction[i][0])

        # TODO pointwise development
        for i in range(len(stddev_progress[0])):
            progress = []
            for j in range(len(stddev_progress)):
                progress.append(stddev_progress[j].detach().numpy()[i])
            plt.plot(progress)

        pdf.savefig()
        plt.close(fig)
