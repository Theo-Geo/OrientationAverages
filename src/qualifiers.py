import numpy as np
from .quaternion_utils import (quat_geodesic, symmetrize_wrt_ref,
                               quaternions_to_rotation_matrix, 
                               rotation_matrix_to_quaternion)


# objective function of a symmetrized quaternion array and their mean
def objective_fct(quat_array, q_mean, weights): 
    obj = 0
    for i in range(len(quat_array)):
        obj += weights[i]*quat_geodesic(quat_array[i], q_mean)**2
    return obj/np.sum(weights)


def minimum_objective_quaternion(quat_array, weights, **kwargs):
    objs = np.empty_like(quat_array)
    for i, q0 in enumerate(quat_array): 
        q_sym = np.array([symmetrize_wrt_ref(q, q0, **kwargs) 
                          for q in quat_array])
        objs[i] = objective_fct(q_sym, q0, weights)
    id_min = np.argmin(objs)
    
    return quat_array[id_min] 
