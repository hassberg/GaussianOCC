import math
import os
import sys

import numpy as np
import pandas as pd
from models.vanishing_self_training_prior_knowledge_model_gp_model.vanishing_self_training_custom_model_based_prior_mean_surrogate_model import \
    VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel
from scipy.spatial import distance
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

from models.self_training_prior_knowledge_model_gp_model.self_training_custom_model_based_prior_mean_surrogate_model import SelfTrainingCustomModelBasedPriorMeanSurrogateModel

tail, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tail)

from gridsearch_handler.grid_search_blueprint_base_estimator import GridSearchBlueprintBaseEstimator
from models.constant_prior_gp_model.constant_prior_mean_surrogate_model import ConstantPriorMeanSurrogateModel
from models.prior_knowledge_model_gp_model.custom_model_based_prior_mean_surrogate_model import CustomModelBasedPriorMeanSurrogateModel
from models.svdd_neg.svdd_neg_surrogate_model import SVDDNegSurrogateModel



def get_parameter_grid(model, data_shape, points, outlier_fraction):
    gamma_range_numbers = 5
    lengthscale_numbers = 5
    combined_numbers = 3

    if model == SVDDNegSurrogateModel:
        tax_cost_estimation = np.divide(1, np.multiply(data_shape[1], outlier_fraction))
        gamma_range = list(map(lambda x: 2 ** x, np.linspace(start=-4, stop=4, num=gamma_range_numbers)))
        parameter = {
            'kernel': ['rbf'],
            'C': [tax_cost_estimation],
            'gamma': gamma_range
        }
        return parameter
    elif model == ConstantPriorMeanSurrogateModel:
        dist = distance.pdist(points)
        lengthscale_range = np.exp(np.linspace(start=np.log(np.maximum(dist.min(), 0.0001)), stop=np.log(dist.max()), num=lengthscale_numbers))
        parameter = {
            'kernel': ['rbf'],
            'lengthscale': lengthscale_range,
        }
        return parameter
    elif model == CustomModelBasedPriorMeanSurrogateModel:
        dist = distance.pdist(points)
        lengthscale_range = np.exp(np.linspace(start=np.log(np.maximum(dist.min(), 0.0001)), stop=np.log(dist.max()), num=combined_numbers))
        tax_cost_estimation = np.divide(1, np.multiply(data_shape[1], outlier_fraction))
        gamma_range = list(map(lambda x: 2 ** x, np.linspace(start=-4, stop=4, num=combined_numbers)))
        parameter = {
            'kernel': ['rbf'],
            'C': [tax_cost_estimation],
            'gamma': gamma_range,
            'lengthscale': lengthscale_range,
        }
        return parameter
    elif model == SelfTrainingCustomModelBasedPriorMeanSurrogateModel:
        tax_cost_estimation = np.divide(1, np.multiply(data_shape[1], outlier_fraction))
        gamma_range = list(map(lambda x: 2 ** x, np.linspace(start=-4, stop=4, num=lengthscale_numbers)))
        parameter = {
            'kernel': ['rbf'],
            'C': [tax_cost_estimation],
            'gamma': gamma_range,
        }
        return parameter
    elif model == VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel:
        tax_cost_estimation = np.divide(1, np.multiply(data_shape[1], outlier_fraction))
        gamma_range = list(map(lambda x: 2 ** x, np.linspace(start=-4, stop=4, num=lengthscale_numbers)))
        parameter = {
            'kernel': ['rbf'],
            'C': [tax_cost_estimation],
            'gamma': gamma_range,
        }
        return parameter
    else:
        raise Exception('parameter for model not defined: ' + model)


def generate_pseudo_validation_data(data, k=None, threshold=0.1):
    if k is None:
        k = math.ceil(5 * np.log10(len(data)))
    tree = KDTree(data)
    edges_index = []
    norm_vec = []

    target_data = []
    outlier_data = []

    for i in range(len(data)):
        knn = tree.query([data[i]], return_distance=True, k=k + 1)
        v_ij = [normalize(data[i] - x) for x in data[knn[1][:, 1:]]][0]
        n_i = np.sum(v_ij, axis=0)
        teta_ij = np.dot(v_ij, n_i)
        l_i = 1 / k * sum(1 for i in teta_ij if i >= 0)
        if l_i >= 1 - threshold:
            edges_index.append(i)
            norm_vec.append(n_i)

        n_i = - normalize(n_i[:, np.newaxis], axis=0).ravel()
        delta_i_pos = np.sum(np.multiply(n_i, (data[(knn[1][:, 1:]).flatten()] - data[i])), axis=1)
        if len(delta_i_pos[[k for k in range(len(delta_i_pos)) if delta_i_pos[k] > 0]]) > 0:
            delta_ij_min_positive = min(delta_i_pos[[i for i in range(len(delta_i_pos)) if delta_i_pos[i] > 0]])
            target_data.append(data[i] + delta_ij_min_positive * n_i)

    l_ns = 1 / len(edges_index)
    outlier_data.extend(data[edges_index] + normalize(norm_vec, axis=1) * l_ns)

    target = np.concatenate([target_data, np.ones((len(target_data), 1))], axis=1)
    outlier = np.concatenate([outlier_data, (-1 * np.ones((len(outlier_data), 1)))], axis=1)
    return np.concatenate([target, outlier], axis=0)


def get_best_parameter(arg_map):
    train_set = pd.read_csv(os.path.join(arg_map["file"], "train.csv")).values
    test_set = pd.read_csv(os.path.join(arg_map["file"], "test.csv")).values
    eval_set = pd.read_csv(os.path.join(arg_map["file"], "eval.csv")).values

    pseudo_data = generate_pseudo_validation_data(train_set[:, :-1])

    data = np.concatenate((train_set[:, :-1], test_set[:, :-1], pseudo_data[:, :-1]), axis=0)
    targets = np.concatenate((train_set[:, -1], test_set[:, -1], pseudo_data[:, -1]), axis=0)
    indices = np.append(np.full((len(train_set)), -1, dtype=int), np.full((len(test_set) + len(pseudo_data)), 0, dtype=int))
    ps = PredefinedSplit(indices)

    arg_map["learning_steps"] = 3
    base_estimator_cycle = GridSearchBlueprintBaseEstimator(blueprint_parameter=arg_map, learning_cycle_evaluation=True)
    grid_search_cycle = GridSearchCV(estimator=base_estimator_cycle, param_grid=get_parameter_grid(arg_map['sm'], train_set.shape, train_set, 0.05), cv=ps, refit=False)
    cycle_fit = grid_search_cycle.fit(data, targets)
    cycle_results = list(map(list, zip(cycle_fit.cv_results_['mean_test_score'], cycle_fit.cv_results_['params'])))
    cycle_results.sort(key=lambda x: x[0], reverse=True)

    return cycle_results
