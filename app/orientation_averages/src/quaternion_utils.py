import numpy as np 
from numba import njit

def wrapper_of(quat_fct, rotation_matrices, *args, **kwargs):
    quat_array = np.array([rotation_matrix_to_quaternion(R) 
                           for R in rotation_matrices])
    return quat_fct(quat_array,*args, **kwargs)

# equivalent to the riemannian metric
@njit
def quat_geodesic(quat, quat_ref, deg=False): 
   
    # avoid round off errors 
    product = min(np.abs(quat_ref@quat),1)
    
    return 2*np.arccos(product) if not deg else 2*np.rad2deg(np.arccos(product))

@njit
def symmetrize_wrt_ref(quat, quat_ref, sym_group="orthorhombic"): 
    
    # all equivalent representations of one crystal orientation
    possible_mutations = np.empty((4,4), dtype=float)
    if sym_group == "orthorhombic":
        # q*g_i where gi = {identity, +/-i, +/-j, +/-k }
        # skip identity element
        possible_mutations[0] = np.copy(quat)
        possible_mutations[1] = np.array([-quat[1], quat[0], quat[3],-quat[2]])
        possible_mutations[2] = np.array([-quat[2],-quat[3], quat[0], quat[1]])
        possible_mutations[3] = np.array([-quat[3], quat[2],-quat[1], quat[0]])
            
    elif sym_group =="triclinic": 
        return quat    
    else: raise NotImplementedError(sym_group," is not a valid symmetry group")

    i0:int = 0
    geodesic_0 = np.pi
    for i, qg in enumerate(possible_mutations):
        geodesic = quat_geodesic(qg, quat_ref)
        if geodesic < geodesic_0: 
            geodesic_0 = geodesic
            i0 = i
            
    return possible_mutations[i0]
    

@njit
def rotation_matrix_to_quaternion(rotation_matrix, tolerance=1e-12): 
    
    quaternion = np.empty(4, dtype=float)
    # positive sign convention
    w = 0.5*np.sqrt(np.abs(rotation_matrix[0][0] + rotation_matrix[1][1] + rotation_matrix[2][2] + 1));
    
    if (w >= tolerance):
        quaternion[0] = w;
        quaternion[1] = 1/(4*w)*(rotation_matrix[2][1] - rotation_matrix[1][2]);
        quaternion[2] = 1/(4*w)*(rotation_matrix[0][2] - rotation_matrix[2][0]);
        quaternion[3] = 1/(4*w)*(rotation_matrix[1][0] - rotation_matrix[0][1]);
    else:
        quaternion[0] = w;
        quaternion[1] = 0.5*np.sign(rotation_matrix[2][1]-rotation_matrix[1][2])*np.sqrt(rotation_matrix[0][0]-rotation_matrix[1][1]-rotation_matrix[2][2]+1);
        quaternion[2] = 0.5*np.sign(rotation_matrix[0][2]-rotation_matrix[2][0])*np.sqrt(-rotation_matrix[0][0]+rotation_matrix[1][1]-rotation_matrix[2][2]+1); 
        quaternion[3] = 0.5*np.sign(rotation_matrix[1][0]-rotation_matrix[0][1])*np.sqrt(-rotation_matrix[0][0]-rotation_matrix[1][1]+rotation_matrix[2][2]+1);
    
    return quaternion/np.linalg.norm(quaternion);

@njit
def quaternions_to_rotation_matrix(quat):
    w, x, y, z = quat
    rotation_matrix = np.empty((3,3), dtype=float)  
    rotation_matrix[0,0] = 1 - 2*(y**2 + z**2);
    rotation_matrix[1,1] = 1 - 2*(x**2 + z**2);
    rotation_matrix[2,2] = 1 - 2*(x**2 + y**2);
    
    rotation_matrix[0,1] = 2*(x*y - z*w); 
    rotation_matrix[0,2] = 2*(x*z + y*w);
    rotation_matrix[1,2] = 2*(y*z - x*w);
     
    rotation_matrix[1,0] = 2*(x*y + z*w); 
    rotation_matrix[2,0] = 2*(x*z - y*w);
    rotation_matrix[2,1] = 2*(y*z + x*w);
    
    return rotation_matrix;