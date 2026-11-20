#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from matplotlib import pyplot as plt
import time

import sys; sys.path.append("../app/")
from orientation_averages import (quaternions_to_rotation_matrix, 
                                  rotation_matrix_to_quaternion,
                                  objective_fct_rotation, quat_geodesic,
                                  markley_average, wrapper_of,
                                  symmetry_average, symmetry_average_approx, 
                                  bingham_average, average_geodesic, )

def measure_time(fct, *args, **kwargs): 
    run_time = np.empty(4, dtype=float)
    
    t0 = time.time()
    for i in range(4): 
        out = fct(*args, **kwargs)
        run_time[i] = time.time() - t0
        t0 = time.time()
    return out, np.min(run_time)

def pm_mean(arr): 
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    return mean-std, mean, mean+std

#def euclid_to_riemann(d): return 2*np.arcsin(d/(2*np.sqrt(2)))
#def riemann_to_euclid(d): return 2*np.sqrt(2)*np.sin(d/2)

#%% load data 

cpo_data = np.genfromtxt("CPO-DDRX-quaternion.dat", skip_header=1)
ids = np.unique(np.int64(cpo_data[:,0]))
N_samples = len(ids)

# plotting specifications
names = ["corrected bingham average",
         "markley average",
         "sym. ave. guess min obj.",
         "sym. ave. guess max outlier"]

colors = ["k", "tab:red", "tab:blue", "tab:green"]

use_euclid = True
symmetry_group = "orthorhombic"
plot_deg = True
use_mean_std = True
use_vf = True
use_random_masking = True
if plot_deg: xlab = "weighted average geodesic (°)"
else:        xlab = "weighted average geodesic (rad)"
max_geo = 100 

# run characteristics
fractions = np.array([0, 25, 50, 75, 85, 90, 95, 97, 98, 99, 99.5])
plot_fig  = np.array([1,  0,  1,  0,  0,  1,  0,  0,  0,  1,    1])

# 10, 50 (median), 90 percentile 
run_time_stats = np.empty((len(fractions), 4, 3))
# 10, 50 (median), 90 percentile 
ave_geo_stats  = np.empty((len(fractions), 4, 3))
 
N_grains = np.empty_like(fractions, dtype=int)

for j0, p_ in enumerate(fractions):
    
    print(f"progress: {j0/(len(fractions)-1)*100}% current p: {p_}")
    
    run_times = np.empty( (N_samples, 4), dtype=float)
    ave_geo = np.empty((N_samples, 4), dtype=float)
    
    for i0, id_ in enumerate(ids):
        
        id_mask = cpo_data[:,0] == id_
        quat_array = cpo_data[id_mask, 2:]
        vf = np.random.rand(len(quat_array)) if use_random_masking else cpo_data[id_mask, 1] #    
        
        # reduce number of points if wanted
        mask = vf > np.percentile(vf, p_)
        N_grains[j0] = np.sum(mask)
        
        quat_array = quat_array[mask]
        vf = vf[mask] if use_vf else np.ones_like(vf[mask]) # np.ones_like(vf[mask]) #  
        
        R_mats = np.array([quaternions_to_rotation_matrix(q) for q in quat_array])
        
        # bingham average
        R_bing, run_times[i0,0] = measure_time(bingham_average, R_mats, vf)
        ave_geo[i0,0] = wrapper_of(average_geodesic, R_mats, 
                                   rotation_matrix_to_quaternion(R_bing), vf, 
                                   deg=plot_deg)
    
        # # # quaternion averages # # # 
        mean, run_times[i0,1] = measure_time(markley_average, quat_array, vf) 
        ave_geo[i0,1] = average_geodesic(quat_array, mean, vf, deg=plot_deg)
        
        mean, run_times[i0,2] = measure_time(symmetry_average, quat_array, vf, 
                                                              euclid=use_euclid)
        ave_geo[i0,2] = average_geodesic(quat_array, mean, vf, deg=plot_deg)
        
        mean, run_times[i0,3] = measure_time(symmetry_average_approx, quat_array, vf, 
                                             guess = "outlier", euclid=use_euclid)
        ave_geo[i0,3] = average_geodesic(quat_array, mean, vf, deg=plot_deg)
        
    # possibly plotting
    if plot_fig[j0]:
        fig, (ax,ax_hist) = plt.subplots(2,1, dpi=400, sharex=True)
        for i in range(4):
            ax.scatter(ave_geo[:,i], run_times[:,i]*1000, label=names[i], c=colors[i], s=5)
            ax_hist.hist(ave_geo[:,i], label=names[i], bins=20, range=(0,max_geo),
                         color=colors[i], histtype="step")
            ax_hist.axvline(np.mean(ave_geo[:,i]), ls="--", color=colors[i])
            
        ax.set(yscale="log", xlim=(0,max_geo), ylabel="run time per sample (ms)",
               title=fr"$N_{{grains}}$ = {N_grains[j0]}, $N_{{samples}}$ = {len(ids)}")
        ax_hist.set(xlabel= xlab, ylabel="count")
        ax.legend(frameon=False)
        #if use_euclid:
        #    fig.savefig(f"../plots/run_time_vs_obj_{N_grains[j0]}_{len(ids)}_euclid.png", bbox_inches='tight')
        #else: 
        #    fig.savefig(f"../plots/run_time_vs_obj_{N_grains[j0]}_{len(ids)}.png", bbox_inches='tight')
        plt.close(fig)

    # meta evaluation
    for n in range(4): # averaging method
        if use_mean_std:   
            run_time_stats[j0, n, :] = pm_mean(run_times[:,n])
            ave_geo_stats[j0, n, :] = pm_mean(ave_geo[:,n])
        else:
            for i, q in enumerate([33, 50, 66]):
                run_time_stats[j0, n, i] = np.percentile(run_times[:,n], q)
                ave_geo_stats[j0, n, i] = np.percentile(ave_geo[:,n], q)
    mask_better = ave_geo[:,0] < ave_geo[:,2] 
    print(f"bingham better than symmetry average: {np.sum(mask_better)} times " 
          + f"{np.sum(mask_better)/N_samples*100:.1f} %")

        
#%% final evaluation 

fig, (axt, axgeo) = plt.subplots(2,1, dpi=400, sharex=True, figsize=(5,5))
for n in range(4): # averaging method
    axt.plot(N_grains, run_time_stats[:,n,1]*1000, ".-", label=names[n], c=colors[n])
    axt.fill_between(N_grains, run_time_stats[:,n,0]*1000, 
                     run_time_stats[:,n,2]*1000, alpha=0.2, color=colors[n])
    
    axgeo.plot(N_grains, ave_geo_stats[:,n,1], ".-", label=names[n], c=colors[n])
    axgeo.fill_between(N_grains, ave_geo_stats[:,n,0], ave_geo_stats[:,n,2], 
                       alpha=0.2, color=colors[n])
    
axt.legend(frameon=False)

axgeo.set(xlabel=r"$N_{grains}$", ylabel=xlab, ylim=(0,90),
          xlim=(4,1e3), xscale="log")
axt.set(ylabel="average run time \nper sample (ms)", yscale="log")

#if use_euclid:
#    fig.savefig(f"../plots/run_time_scaling_{len(ids)}_euclid.png", bbox_inches='tight')
#else: 
#    fig.savefig(f"../plots/run_time_scaling_{len(ids)}.png", bbox_inches='tight')









