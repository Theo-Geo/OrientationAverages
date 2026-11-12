# -*- coding: utf-8 -*-
from .src.quaternion_utils import (
    quaternions_to_rotation_matrix, rotation_matrix_to_quaternion,
    quat_geodesic, symmetrize_wrt_ref, wrapper_of)

from .src.averages import (
    quaternion_average_Markley, 
    quaternion_average_SO3_G)

from .src.qualifiers import objective_fct
