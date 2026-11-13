#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from matplotlib import pyplot as plt
import time

import sys; sys.path.append("../../")
from orientation_averages import (quaternions_to_rotation_matrix, 
                              objective_fct_rotation, 
                              markley_average, symmetrize_wrt_ref,
                              symmetry_average, symmetry_average_approx, 
                              bingham_average, objective_fct)

def measure_time(fct, *args, **kwargs): 
    run_time = np.empty(4, dtype=float)
    
    t0 = time.time()
    for i in range(4): 
        out = fct(*args, **kwargs)
        run_time[i] = time.time() - t0
        t0 = time.time()
    return out, np.min(run_time)

#%%

rng = np.random.default_rng(seed=42)

cpo_data = np.genfromtxt("CPO-DDRX-quaternion.dat", skip_header=1)
ids = np.unique(np.int64(cpo_data[:,0]))

run_times = np.empty( (len(ids), 4), dtype=float)
objectives = np.empty((len(ids), 4), dtype=float)

names = ["corrected bingham average",
         "markley average",
         "sym. ave. guess min obj.",
         "sym. ave. guess max outlier"]

colors = ["k", "tab:red", "tab:blue", "tab:green"]

for i0, id_ in enumerate(ids):
    
    id_mask = cpo_data[:,0] == id_
    quat_array = cpo_data[id_mask, 2:]
    vf = cpo_data[id_mask, 1] #np.random.rand(len(quat_array)) # np.ones(len(quat_array)) #   
    
    # reduce number of points if wanted
    
    mask = vf > np.percentile(vf,97)
    quat_array = quat_array[mask]
    vf = np.ones_like(vf[mask])
    
    R_mats = np.array([quaternions_to_rotation_matrix(q) for q in quat_array])
    
    # bingham average
    R_bing, run_times[i0,0] = measure_time(bingham_average, R_mats, vf)
    objectives[i0,0] = objective_fct_rotation(R_mats, R_bing, vf)

    # # # quaternion averages # # # 
    mean_markley, run_times[i0,1] = measure_time(markley_average, quat_array, vf) 
    _sym_quats = np.array([symmetrize_wrt_ref(q, mean_markley) for q in quat_array])
    objectives[i0,1] = objective_fct(_sym_quats, mean_markley, vf)
    
    (_, objectives[i0,2]), run_times[i0,2] = measure_time(symmetry_average, 
                                                          quat_array, vf)
    
    (_, objectives[i0,3]), run_times[i0,3] = measure_time(symmetry_average_approx, 
                                            quat_array, vf, guess = "outlier")
        

#%%

fig, (ax,ax_hist) = plt.subplots(2,1, dpi=200, sharex=True)
for i in range(len(run_times.T)):
    ax.scatter(objectives[:,i], run_times[:,i], label=names[i], c=colors[i], s=5)
    ax_hist.hist(objectives[:,i], label=names[i], bins=25, range=(0,2.5),
                 color=colors[i], histtype="step")
    ax_hist.axvline(np.mean(objectives[:,i]), ls="--", color=colors[i])
    #ax_hist.axvspan(np.std(objectives[:,i]))
    
    
ax.set(yscale="log", xlim=(0,2.5))
ax.legend(frameon=False)













