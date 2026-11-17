#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import sys; sys.path.append("../app/")
from orientation_averages.src.averages import (markley_average, 
                                               symmetry_average, 
                                               symmetry_average_approx)
from orientation_averages.src.quaternion_utils import (
    axis_angle_to_quaternion, quaternion_to_axis_angle)

#import unittest


#%% average on S^1 

# the notion of an average on S^1 is well defined in directional statistics 
# it comes down to averaging all components of a vector separately and then 
# taking the arctan of the new product 
# see e.g. https://en.wikipedia.org/wiki/Circular_mean
# the triclinic average should return this average 
# the easiest symmetry group in S^1 i can think of 
# is a reflection around the origin here implemented by a sign flip if the 
# angle between vec1 and vec2 is larger than 90°

def _symmetric_circular_mean(angle1, angle2, w1, w2, reflect=False): 
    vec1 = np.array([np.cos(angle1), np.sin(angle1)])
    vec2 = np.array([np.cos(angle2), np.sin(angle2)])
    if reflect: 
        product = min(1, max(-1,vec1@vec2)) # avoid floating point errors
        if np.arccos(product) > np.pi/2: vec2 *= -1
    mean_vec = 0.5*(w1*vec1 + w2*vec2)
    return np.arctan2(mean_vec[1], mean_vec[2])
    
# test data: two rotations around the x-axis by an angle phi

def test_circular_average(): 
    rot_axis = np.array([1,0,0])
    phi1 = np.deg2rad(10)
    phi2 = np.deg2rad(350) 
    #weights = np.array([1/3,2/3])
    quat1 = axis_angle_to_quaternion(rot_axis, phi1)
    quat2 = axis_angle_to_quaternion(rot_axis, phi2) 

    for weights in [np.ones(2), np.array([1/3,2/3])]:
        true_mean = _symmetric_circular_mean(phi1, phi2, weights)
        mean_quat = symmetry_average(np.array([quat1,quat2]), weights, 
                                          sym_group="triclinic")
        mean_quat_markley = markley_average(np.array([quat1,quat2]), weights)
        
        axis, mean_angle = quaternion_to_axis_angle(mean_quat)    
        axis, mean_angle_markley = quaternion_to_axis_angle(mean_quat_markley) 
        
        if all(weights==1): assert np.abs(mean_angle) < 1e-12
        assert np.abs(mean_angle - true_mean) < 1e-12
        assert np.abs(mean_angle - mean_angle_markley) < 1e-12
        
def test_symmetry_average():
    rot_axis = np.array([1,0,0])
    phi1 = np.deg2rad(10)
    phi2 = np.deg2rad(170) 
    #weights = np.array([1/3,2/3])
    quat1 = axis_angle_to_quaternion(rot_axis, phi1)
    quat2 = axis_angle_to_quaternion(rot_axis, phi2) 

    for weights in [np.ones(2), np.array([1/3,2/3])]:
        true_mean = _symmetric_circular_mean(phi1, phi2, weights, reflect=True)
        mean_quat = symmetry_average(np.array([quat1,quat2]), weights, 
                                          sym_group="orthorhombic")
        axis, mean_angle = quaternion_to_axis_angle(mean_quat)    
        
        if all(weights==1): assert np.abs(mean_angle) < 1e-12
        assert np.abs(mean_angle - true_mean) < 1e-12
        
    
    

    














