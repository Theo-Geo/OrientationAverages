import numpy as np
from numba import njit
from .quaternion_utils import (quat_geodesic, symmetrize_wrt_ref,
                               quaternions_to_rotation_matrix, 
                               rotation_matrix_to_quaternion)


# objective function of a symmetrized quaternion array and their mean
@njit
def objective_fct(quat_array, q_mean, weights): 
    obj = 0
    for i in range(len(quat_array)):
        obj += weights[i]*quat_geodesic(quat_array[i], q_mean)**2
    return obj/np.sum(weights)

def objective_fct_rotation(rotation_matrix_array, mean_rotation, weights): 
    quaternion_array = np.array([rotation_matrix_to_quaternion(
                                    R, 1e-12) for R in rotation_matrix_array])
    q_mean = rotation_matrix_to_quaternion(mean_rotation, 1e-12)
    q_sym = np.array([symmetrize_wrt_ref(q, q_mean) for q in quaternion_array])
    obj = 0
    for i in range(len(quaternion_array)):
        obj += weights[i]*quat_geodesic(q_sym[i], q_mean)**2
    return obj/np.sum(weights)

