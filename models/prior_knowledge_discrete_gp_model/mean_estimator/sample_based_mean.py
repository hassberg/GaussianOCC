import random

import numpy as np
import math
import tensorflow as tf
from scipy.spatial import ConvexHull


def max_pairwise_dist(hull_points):
    max_dist = 0.0
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            max_dist = np.maximum(max_dist, math.dist(hull_points[i], hull_points[j]))
    return max_dist


def calc_diameter(points):
    sample_points = tf.convert_to_tensor(np.asarray(points)[random.sample(range(len(points)), 50)])
    # convex_hull_points = tf.gather(points, np.unique(ConvexHull(sample_points).simplices.flat))
    return max_pairwise_dist(sample_points)


def nearest_neighbor(x, remaining_points):
    f = lambda p: math.dist(x, p[[*range(len(x))]])
    distances = np.apply_along_axis(f, 1, remaining_points)
    index = np.argmin(distances)
    return remaining_points[index], np.delete(remaining_points, index, axis=0)


def append_to_matrix(v1):
    arr = np.zeros((len(v1), len(v1)))

    for i in range(len(v1)):
        arr[0][i] = v1[i]

    ind = np.nonzero(np.array(v1) != 0)
    line = ind[0][0]
    remaining_basis = np.c_[range(1, len(v1)), [x for x in range(len(v1)) if x != line]]
    for x in remaining_basis:
        arr[x[0]][x[1]] = 1

    return arr


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (np.abs(w) > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def change_of_basis(new_basis):
    # Todo new basis as 0:0??
    old_basis = np.zeros((len(new_basis), len(new_basis)))
    for i in range(len(new_basis)):
        old_basis[i][i] = 1

    cob = []
    for i in range(len(old_basis)):
        b = np.linalg.solve(new_basis, old_basis[i])
        cob.append(b)

    return np.asarray(cob)


def filter_covered_by_hyperplane(base_vectors, considered_point, position_vector, points):
    if points.size != 0:
        cob = change_of_basis(base_vectors)

        res = np.dot(cob, (considered_point - position_vector))[0]

        f = lambda x: np.dot(cob, x[[*range(len(position_vector))]] - position_vector)[0]
        orthogonal_vector_influence = np.apply_along_axis(f, 1, points)

        eval_influence = lambda x: x > 0 if res > 0 else x < 0
        on_correct_side = np.apply_along_axis(eval_influence, 0, orthogonal_vector_influence)

        return points[on_correct_side]
    else:
        return np.array([])


def remove_covered(point, nn, array):
    orth_direction = point - nn[[*range(len(point))]]
    base_arr = append_to_matrix(orth_direction)
    base_vectors = gram_schmidt(base_arr)
    return filter_covered_by_hyperplane(base_vectors, point, nn[[*range(len(nn) - 1)]], array)


def calc_relative_distance_weights(x, neighbor_points):
    distances = np.apply_along_axis((lambda p: math.dist(p[[*range(len(x))]], x)), axis=1, arr=neighbor_points)
    dist_ges = sum(distances)
    distance_weights = np.apply_along_axis((lambda d: np.subtract(1, np.divide(d, dist_ges))), axis=0,
                                           arr=distances)
    dist_weights_ges = sum(distance_weights)
    return np.apply_along_axis((lambda d: np.divide(d, dist_weights_ges)), axis=0, arr=distance_weights)


class SampleBasedMeanEstimation:
    def __init__(self, reference_points):
        self.ref_points = reference_points
        self.points_diameter = calc_diameter(reference_points[:, :-1])
        # probably as parameter
        self.diameter_ratio = 1.0

    def calculate_influence_sphere(self, x):
        available_points = self.ref_points.numpy()
        sphere = []
        while len(available_points) != 0:
            nn, available_points = nearest_neighbor(x, available_points)
            sphere.append(nn)
            available_points = remove_covered(x, nn, available_points)
        return sphere

    def vanishing_factor(self, x, neighbor_points):
        return np.apply_along_axis((lambda p: np.subtract(1, np.minimum(
            np.divide(math.dist(x, p[[*range(len(x))]]), np.product((self.points_diameter, self.diameter_ratio))), 1))),
                                   axis=1, arr=neighbor_points)

    def calc_mean(self, x, neighbor_points):
        vanishing_factor = self.vanishing_factor(x, neighbor_points)
        if len(neighbor_points) == 1:
            return np.asarray(neighbor_points)[:, -1] * vanishing_factor
        else:
            relative_distance_weights = calc_relative_distance_weights(x, neighbor_points)
            if np.min(relative_distance_weights) <= 0:
                print("ERR: dist weights of: ", relative_distance_weights)

            distance_weights = relative_distance_weights * vanishing_factor

            mean = sum(distance_weights * np.asarray(neighbor_points)[:, -1])
            if (mean > 1 or mean < -1):
                print("ERR: mean of: ", mean)
            return mean

    def estimate_mean(self, x):
        if x in (self.ref_points[:, :-1]).numpy():
            # TODO correctly apply where operator...
            i, _ = np.where((self.ref_points[:, :-1]).numpy() == x)
            ind = i[0]
            return self.ref_points[ind, -1]
        else:
            influence_sphere = self.calculate_influence_sphere(x)
            mean = self.calc_mean(x, influence_sphere)
            return mean

    def calculate_means(self, points):
        means = np.zeros((len(points), 1))
        for i in range(len(points)):
            means[i] = self.estimate_mean(points[i].numpy())
        return means
