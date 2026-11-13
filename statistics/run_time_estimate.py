#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from matplotlib import pyplot as plt
import time

import sys; sys.path.append("../app/")
from orientation_averages import (quaternions_to_rotation_matrix, 
                              objective_fct_rotation, quat_geodesic,
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
    
    mask = vf > np.percentile(vf,99)
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

n_grains = np.sum(mask)

fig, (ax,ax_hist) = plt.subplots(2,1, dpi=400, sharex=True)
for i in range(len(run_times.T)):
    ax.scatter(objectives[:,i], run_times[:,i]*1000, label=names[i], c=colors[i], s=5)
    ax_hist.hist(objectives[:,i], label=names[i], bins=30, range=(0,3),
                 color=colors[i], histtype="step")
    ax_hist.axvline(np.mean(objectives[:,i]), ls="--", color=colors[i])
    
ax.set(yscale="log", xlim=(0,np.pi), title=fr"$N_{{grains}}$ = {n_grains}, $N_{{samples}}$ = {len(ids)}", 
       ylabel="run time (ms)")
ax_hist.set(xlabel="objective function (rad)", ylabel="count")
ax.legend(frameon=False)
fig.savefig(f"../plots/run_time_vs_obj_{n_grains}_{len(ids)}.png")












