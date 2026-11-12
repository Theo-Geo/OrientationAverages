import numpy as np 
from .quaternion_utils import (rotation_matrix_to_quaternion, 
                               quaternions_to_rotation_matrix, 
                               quat_geodesic, 
                               symmetrize_wrt_ref)
from .qualifiers import objective_fct

# average as defined in Averaging Quaternions - Markley et.al. 2007 
def quaternion_average_Markley(quat_array, weights): 
    M = np.zeros((4,4), dtype=float)
    for q, w in zip(quat_array, weights): M += np.outer(q, q)*w
    evs, evecs = np.linalg.eigh(4*M/np.sum(weights)-np.eye(4)) # return sorted evs, evecs
    mean = evecs[:,-1]
    # stay in halfspace where q.w > 0
    if mean[0] < 0: mean *= -1

    return mean 
"""
def rotation_average_SO3_G(rotation_matrix_array, weights, *args, **kwargs): 
    quaternion_array = np.array([rotation_matrix_to_quaternion(
                                    R, 1e-12) for R in rotation_matrix_array])
    print(quaternion_array.shape, weights.shape)
    q_mean, obj_fct = quaternion_average_SO3_G(quaternion_array, weights, *args, **kwargs)
    print(q_mean.shape)
    return quaternions_to_rotation_matrix(q_mean), obj_fct
"""
def quaternion_average_SO3_G_approx(quat_array, weights, max_iterations=100, guess="mean"): 
    
    q_mean = quaternion_average_Markley(quat_array, weights) 
    obj_fct = objective_fct(quat_array, q_mean, weights) 
    
    # craft an initial guess initial guess    
    geodesics = np.array([quat_geodesic(q, q_mean) for q in quat_array])
    #geodesics = np.array([quat_geodesic(symmetrize_wrt_ref(q, q_mean), q_mean) 
    #                      for q in quat_array])

    i_max = np.argmax(geodesics)
    i_min = np.argmin(geodesics)
    
    if guess == "mean": 
        q0 = q_mean
    elif guess == "half outlier":
        q0 = quaternion_average_Markley(np.array([q_mean,quat_array[i_max]]), np.ones(2))
    elif guess == "outlier":
        q0 = np.copy(quat_array[i_max])
    elif guess == "closest to mean":
        q0 = np.copy(quat_array[i_min])
    else: print("not a valid guess")        
    
    for n in range(max_iterations):
        q_sym = np.array([symmetrize_wrt_ref(q, q0) for q in quat_array])
        q_mean = quaternion_average_Markley(q_sym, weights)
        new_objective = objective_fct(q_sym, q_mean, weights)
        
        # check if objective is decreasing
        if obj_fct > new_objective: 
            q0 = np.copy(q_mean)
            obj_fct = np.copy(new_objective) 
        else: break
        if n==max_iterations-1: print("ah")
    print(n)

    return q_mean, new_objective


def quaternion_average_SO3_G(quat_array:np.ndarray, weights:np.ndarray, 
                             max_iterations:int=100, **kwargs
                             ) -> (float, np.ndarray):
    
    # craft an initial guess initial guess    
    objs = np.empty_like(quat_array)
    for i, q0 in enumerate(quat_array): 
        q_sym = np.array([symmetrize_wrt_ref(q, q0, **kwargs) 
                          for q in quat_array])
        objs[i] = objective_fct(q_sym, q0, weights)
    id_min = np.argmin(objs)
    
    q0 = quat_array[id_min] 
    obj_fct = objs[id_min]
    
    #iterate for solution
    for n in range(max_iterations):
        if n>0 : q_sym = np.array([symmetrize_wrt_ref(q, q0, **kwargs) 
                                   for q in quat_array])
        q_mean = quaternion_average_Markley(q_sym, weights)
        new_objective = objective_fct(q_sym, q_mean, weights)
        
        # check if objective is decreasing
        if obj_fct > new_objective: 
            q0 = np.copy(q_mean)
            obj_fct = np.copy(new_objective) 
        else: break
        if n==max_iterations-1: print("ah")
    print(n)

    return q_mean, new_objective



