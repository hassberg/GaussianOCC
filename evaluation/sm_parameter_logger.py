from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric


class SmParameterLogger(EvaluationMetric):
    def __init__(self, kth_best, parameter):
        self.kth_best = kth_best
        self.parameter = parameter
        self.end_experiment = None

    def get_evaluation(self):
        log = "{\"kth_ranking\" : [" + str(self.kth_best) + "], \n"
        params = ',\n'.join("\"" + key + "\" : [" + ( "\"" + value + "\"" if type(value) == str else str(value)) + "]" for key,value in self.parameter.items())
        log += params
        log += "\n}"
        return log
