#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#%% average on S^1 

# the notion of an average on S^1 is well defined in directional statistics 
# it comes down to averaging all components of a vector separately and then 
# taking the arctan of the new product 
# see e.g. https://en.wikipedia.org/wiki/Circular_mean
# the triclinic average of rotations around e.g. the x-axis by an angle 

rot_axis = np.eye(3)[0]


