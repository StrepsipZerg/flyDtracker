#Functions to graphically visualize data.
#Currently only polar_histogram is implemented.
#Better legend should be added.
#The mean and standard deviation look odd, and are probably not very useful considering our data.
import numpy as np
import matplotlib.pyplot as plt
from data_processing import *


def polar_histogram(flystack, distance=True, #if distance is false, will plot angle instead.
                    force_bin=72, #Number of divisions
                    dist_range=500): #For distance only, range ot consider
    
    #Visual rendering of a relative_fly set (angle and distance)
    
    rel_set = relative_set(flystack)
    
    if distance:
        num_bins_theta = 1 # Number of bin edges in angular direction (just one so we get info only about distance)
        num_bins_r = force_bin

    elif distance==False:
        num_bins_r = 1
        num_bins_theta = force_bin
        
    # Create polar edges
    r_edges = np.linspace(0, dist_range, num_bins_r + 1) 
    theta_edges = np.linspace(0, 2*np.pi, num_bins_theta + 1)

    # Transform cartesian to polar coordinates
    r = rel_set.dist
    theta = np.radians(rel_set.orientations)
    
    
    # Calculate the 2d histogram and binned statistics for focal turning and acceleration
    Pos = stats.binned_statistic_2d(r, theta, None, 'count', bins=np.asarray([r_edges,theta_edges]))
    # Calculate binsize for normalization (binsize increases with radius)
    dr = np.pi*(r_edges[1:]**2 - r_edges[0:-1]**2)
    dtheta = (theta_edges[1] - theta_edges[0])/(2*np.pi)
    area = np.repeat(dtheta*dr[:,np.newaxis],theta_edges.shape[0]-1,1)
    
    def interpolate_polarmap_angles(histogram, theta_edges, r_edges, factor = 1):
        histogram_interpolated = np.zeros((histogram.shape[0], histogram.shape[1]*factor))
        for k in range(factor):
            histogram_interpolated[:,k::factor] = histogram
        theta_edges = np.linspace(-np.pi, np.pi, (theta_edges.shape[0]-1)*factor + 1)
        Theta, R = np.meshgrid(theta_edges, r_edges)
        return histogram_interpolated, Theta, R
    
    def plot_polar_histogram(values, label, ax, cmap=None, sym=False):

        Theta, R = np.meshgrid(theta_edges, r_edges)
        mp, Theta, R = interpolate_polarmap_angles(values, theta_edges, r_edges, factor = 30)

        # Select color limits: 
        if sym:
            vmax = np.max(np.abs(values))
            vmin = -vmax
        else:
            vmax = np.max(values)
            vmin = 0

        # Plot histogram/map
        im = ax.pcolormesh(Theta, R, mp, cmap=cmap, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im, ax=ax, cmap=cmap)
        cb.ax.tick_params(labelsize=24)
        ax.set_title(label,fontsize=36)

        # Adjusting axis and sense of rotation to make it compatible with [2]:
        # Direction of movement along vertical axis
        ax.set_theta_zero_location("N")
        
    plt.figure(num=None, figsize=(40, 10), facecolor='w', edgecolor='k')
    # Plot polar histogram/maps for relative neighbor positions, turning and acceleration 
    if distance:
        label = "Distance"
    else:
        label = "Angle"
    label = label + " to other flies"
    plot_polar_histogram(Pos.statistic/area/np.sum(Pos.statistic), label, plt.subplot(131, polar=True))
    theta_mean = stats.circmean(theta[~np.isnan(theta)])
    theta_std = stats.circstd(theta[~np.isnan(theta)])
    plt.axvline(theta_mean, color="red", linewidth=3)
    plt.axvline(theta_mean-theta_std, color="red", linestyle=":", linewidth=2)
    plt.axvline(theta_mean+theta_std, color="red", linestyle=":", linewidth=2)


