# -*- coding: utf-8 -*-

import numpy as np
from orientation_averages import quaternions_to_rotation_matrix


q = np.array([1,0,0,0])

print(quaternions_to_rotation_matrix(q))