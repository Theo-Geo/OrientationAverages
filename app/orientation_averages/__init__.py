# -*- coding: utf-8 -*-
from .src.quaternion_utils import (
    quaternions_to_rotation_matrix, rotation_matrix_to_quaternion, 
    symmetrize_wrt_ref, wrapper_of, quat_geodesic)

from .src.averages import (
    markley_average, 
    symmetry_average, 
    symmetry_average_approx,
    bingham_average)

from .src.qualifiers import objective_fct, objective_fct_rotation
