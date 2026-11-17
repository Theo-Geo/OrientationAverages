import numpy as np 
from numba import njit
from .quaternion_utils import (rotation_matrix_to_quaternion, 
                              quaternions_to_rotation_matrix, 
                              quat_geodesic, symmetrize_wrt_ref)
from .qualifiers import objective_fct

# average as defined in Averaging Quaternions - Markley et.al. 2007 
def markley_average(quat_array, weights): 
    M = np.zeros((4,4), dtype=float)
    for q, w in zip(quat_array, weights): M += np.outer(q, q)*w
    evs, evecs = np.linalg.eigh(4*M/np.sum(weights)-np.eye(4)) # return sorted evs, evecs
    mean = evecs[:,-1]
    assert np.abs(np.linalg.norm(mean)-1)<1e-10, ValueError("average is not of norm one")
    # stay in halfspace where q.w > 0
    if mean[0] < 0: mean *= -1

    return mean 

def symmetry_average_approx(quat_array, weights, 
                            max_iterations:int=100, guess:str="mean", 
                            sym_group="orthorhombic", euclid=False): 
    
    # craft an initial guess initial guess
    q_mean = markley_average(quat_array, weights)  # kind of random
    obj_fct = objective_fct(quat_array, q_mean, weights, euclid) 
      
    geodesics = np.array([quat_geodesic(symmetrize_wrt_ref(q, q_mean, sym_group, euclid),
                                        q_mean) for q in quat_array])

    i_max = np.argmax(geodesics)
    i_min = np.argmin(geodesics)
    
    if guess == "mean":                 q0 = q_mean
    elif guess == "half outlier":
        q0 = markley_average(np.array([q_mean, quat_array[i_max]]), np.ones(2))
    elif guess == "outlier":            q0 = np.copy(quat_array[i_max])
    elif guess == "closest to mean":    q0 = np.copy(quat_array[i_min])
    else: print("not a valid guess")        
    
    # iterate for solution
    for n in range(max_iterations):
        q_sym = np.array([symmetrize_wrt_ref(q, q0, sym_group, euclid) 
                          for q in quat_array])
        q_mean = markley_average(q_sym, weights)
        new_objective = objective_fct(q_sym, q_mean, weights, euclid)
        
        # check if objective is decreasing
        if obj_fct > new_objective: 
            q0 = np.copy(q_mean)
            obj_fct = np.copy(new_objective) 
        else: return q0
        if n==max_iterations-1: 
            raise Exception("Maximum number of iterations exceeded " + 
                            f"in symmetry average. n = {n}")

    print("ahhh")
   

def symmetry_average(quat_array, weights, max_iterations:int=100, 
                     sym_group="orthorhombic", euclid=False):
    # craft an initial guess
    q0, obj_fct = _minimum_objective_quaternion(quat_array, weights, 
                                                sym_group, euclid)
    
    #iterate for solution
    for n in range(max_iterations):
        q_sym = np.array([symmetrize_wrt_ref(q, q0, sym_group, euclid) 
                                   for q in quat_array])
        q_mean = markley_average(q_sym, weights)
        new_objective = objective_fct(q_sym, q_mean, weights, euclid)
        
        # check if objective is decreasing
        if obj_fct > new_objective: 
            q0 = np.copy(q_mean)
            obj_fct = np.copy(new_objective) 
        else: return q0
        if n==max_iterations-1: 
            raise Exception("Maximum number of iterations exceeded " + 
                            f"in symmetry average. n = {n}")

    print("ahhh")
    

@njit
def _minimum_objective_quaternion(quat_array, weights, sym_group="orthorhombic", 
                                  euclid:bool=False):
    objs = np.zeros(len(quat_array))
    for i, q0 in enumerate(quat_array): 
        for j,qn in enumerate(quat_array):
            q_sym = symmetrize_wrt_ref(qn, q0, sym_group, euclid) 
            objs[i] += weights[j]*quat_geodesic(q_sym, q0, euclid)**2
        objs[i] /= np.sum(weights)
    id_min = np.argmin(objs)
    
    return quat_array[id_min], objs[id_min]


# only applicable to orthorhombic materials 
def bingham_average(R_mats, weights, correct:bool=True):
    # principal eigenvectors for the [100], [010], [001] axes
    eig_vecs = np.empty((3,3), dtype=float)
    for i in range(3):        
        evs, evecs = np.linalg.eigh(_struc2(R_mats[:,i]))
        eig_vecs[i] = evecs[:,-1]
    
    # correct for handedness and non-orthogonality
    if correct:
        handedness = np.dot(np.cross(eig_vecs[0], eig_vecs[1]), eig_vecs[2])
        if handedness < 0: eig_vecs[-1] *= -1
        U,S,Vh = np.linalg.svd(eig_vecs.T)
        return U@Vh # orthogonalized 
    else: return eig_vecs
    
@njit
def _struc2(n_ens, weights=None):
    """
    computes the second order structure tensor 
    of an ensemble of unit vectors
    """
    struc_tens = np.zeros((3,3))    
    if weights is None: weights = np.ones(len(n_ens))
    for n, w in zip(n_ens,weights): struc_tens += np.outer(n,n)*w
    return struc_tens/np.sum(weights)

    



