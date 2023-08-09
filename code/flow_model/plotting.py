import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
import torch
sys.path.append('..')
import util

def distribution(trajectories, pca, label, baseline=None):
    # Plot a scatter plot showing the overall distribution of points in the trajectories
    t_emb = pca.transform(trajectories.reshape(-1, trajectories.shape[2]))[:,:2]
    t_emb = t_emb.reshape(trajectories.shape[0], trajectories.shape[1], -1)
    # Connect the scatter plot points
    # Color the points by the distance along the trajectory
    colormap = matplotlib.colormaps.get_cmap('gnuplot')
    len_trajectory = trajectories.shape[0]
    colors = colormap(np.arange(len_trajectory)/len_trajectory)
    # plt.scatter(proj[:,0], proj[:,1], s=.1,
    #             c=cell_colors, cmap='tab20c')
    for i in range(t_emb.shape[1]):
        # plt.plot(t_emb[:,i,0], t_emb[:,i,1], color='black', linewidth=.5, alpha=.1)
        plt.scatter(t_emb[:,i,0], t_emb[:,i,1],
                    color=colors, 
                    marker='o', s=.5, alpha=0.5)
    if baseline is not None:
        baseline_emb = pca.transform(baseline)[:,0:2]
        plt.scatter(baseline_emb[:,0], baseline_emb[:,1], color='black', alpha=.2, s=.25)
    # Remove the axis labels
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{label.capitalize()} trajectory distribution', fontsize=12)
        
# Plot a sample of the individual trajectories
def sample_trajectories(trajectories, X, pca, label):
    # Get a 4x4 grid of plots
    nrow = 3
    ncol = 3
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20,20))
    # Choose a random sample of 16 trajectories
    num_trajectories = trajectories.shape[1]
    proj = np.array(pca.transform(X)[:,0:2])
    t_emb = pca.transform(trajectories.reshape(-1, trajectories.shape[2]))[:,:2]
    t_emb = t_emb.reshape(trajectories.shape[0], trajectories.shape[1], -1)
    # Color the points by the distance along the trajectory
    colormap = matplotlib.colormaps.get_cmap('plasma')
    len_trajectory = trajectories.shape[0]
    colors = colormap(np.arange(len_trajectory)/len_trajectory)
    for i,idx in enumerate(random.sample(range(num_trajectories), nrow*ncol)):
        # Plot the scatter plot
        ax = axs[i//ncol, i%ncol]
        ax.scatter(proj[:,0], proj[:,1], color='grey', alpha=1, s=.1)
        # Plot the trajectory
        ax.plot(t_emb[:,idx,0], t_emb[:,idx,1], color='black', linewidth=.5)
        ax.scatter(t_emb[:,idx,0], t_emb[:,idx,1], c=colors, 
                   marker='o', s=1, zorder=100)
        # Remove the axis labels
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'{label.capitalize()} sampled trajectories', fontsize=30)
    plt.tight_layout() 
    # Create a new axis to the right of the plots for the colorbar
    cax = fig.add_axes([1.01, .1, .02, .8])
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=colormap), cax=cax, label='Time')
    cbar.ax.tick_params(labelsize=20)
    # Increase the font size of the colorbar label
    cbar.ax.yaxis.label.set_size(26)
    plt.subplots_adjust(top=0.95)

def compute_velo(model, X, perturbation=None, numpy=False):
    # Run the model on every data point
    with torch.no_grad():
        x = X.clone()
        if perturbation is not None:
            perturb_idx, perturb_value = perturbation
            # If we specified a gene to perturb, set the expression of that gene to the perturbation value
            x[:,perturb_idx] = perturb_value
            velos, vars = model(x)
            # Set the velocity of the perturbed gene to the perturbation value
            velos[:,perturb_idx] = perturb_value
        else:
            velos, vars = model(x)
    if numpy:
        return velos.detach().cpu().numpy(), vars.detach().cpu().numpy()
    else:
        return velos, vars

import matplotlib
def arrow_grid(velos, data, pca, labels):
    # If we pass in just a single data set of velocities, convert it to a list
    # so that all the comparison based code works
    if type(velos) is not list and type(velos) is not tuple:
        velos = [velos]
        data = [data]
    num_comparisons = len(velos)
    num_cells = velos[0].shape[1]
    if num_comparisons != len(labels):
        raise ValueError(f'Number of labels ({len(labels)}) must match number of comparisons ({num_comparisons})')
    X = [d.X.toarray() for d in data]
    # Compute the projection of gene expression onto the first two principal components
    proj = [np.array(pca.transform(x))[:,0:2] for x in X]
    # Find the extents of the projection
    minX = min([np.min(p[:,0]) for p in proj])
    maxX = max([np.max(p[:,0]) for p in proj])
    minY = min([np.min(p[:,1]) for p in proj])
    maxY = max([np.max(p[:,1]) for p in proj])
    # Add some buffer to the sides of the plot
    xbuf = (maxX - minX) * 0.05
    ybuf = (maxY - minY) * 0.05
    # Set the granularity of the grid, i.e. how many grids to divide the space into
    # in both the x and y directions
    n_points = 20
    x_grid_points = np.linspace(minX, maxX, n_points)
    y_grid_points = np.linspace(minY, maxY, n_points)
    # Find the width and height of each grid cell
    x_spacing = x_grid_points[1] - x_grid_points[0]
    y_spacing = y_grid_points[1] - y_grid_points[0]
    # This creates a sequential list of points defining the upper left corner of each grid cell
    grid = np.array(np.meshgrid(x_grid_points, y_grid_points)).T.reshape(-1,2)
    # Set up a list of velocities for each grid cell
    velocity_grid = np.zeros_like(grid)
    # This is nan, rather than zero so that we can distinguish between
    # grid cells with zero velocity and grid cells with 
    # no points inside, which wil be (nan)
    velocity_grid[:] = np.nan

    # Find points inside each grid cell
    mean_velocities = np.zeros((num_comparisons, len(grid), num_cells))
    variances = np.zeros((num_comparisons, len(grid), num_cells))
    mean_X = np.zeros((num_comparisons, len(grid), num_cells))

    for i,(x,y) in enumerate(grid):
        for j in range(num_comparisons):
            # Find the points inside the grid cell
            idx = np.where((proj[j][:,0] > x) & (proj[j][:,0] < x+x_spacing) & 
                        (proj[j][:,1] > y) & (proj[j][:,1] < y+y_spacing))[0]
            # If there are any points inside the grid cell
            if len(idx) > 0:
                # Get the average velocity vector for the points 
                # inside the grid cell
                velo = velos[j][idx,:]
                # var = vars[j][idx,:]
                # Compute the mean velocity vector of the points inside the grid cell
                mean_velocities[j,i] = velo.mean(axis=0).reshape(-1)
                # variances[j,i] = var.mean(axis=0).reshape(-1)
                mean_X[j,i] = X[j][idx,:].mean(axis=0)
            
    # variances = util.tonp(variances)
    pca_embed = lambda x: np.array(pca.transform(x)[:,0:2])
    velocity_grid = [util.embed_velocity(x, v, pca_embed) for x,v in zip(mean_X, mean_velocities)]
    
    fig, ax = plt.subplots(1,1, figsize=(10,10))

    cell_types = {name:i for i,name in enumerate(sorted(data[0].obs['cell_type'].unique()))}
    cell_colors = data[0].obs['cell_type'].map(cell_types)
    ax.set_xlim(minX-xbuf, maxX+xbuf)
    ax.set_ylim(minY-ybuf, maxY+ybuf)
    ax.set_facecolor('grey')
    # Add lines to show the grid
    for x in x_grid_points:
        ax.axvline(x, color='white', alpha=0.1)
    for y in y_grid_points:
        ax.axhline(y, color='white', alpha=0.1)
    #Plot the velocity vectors
    ax.scatter(proj[0][:,0], proj[0][:,1], c=cell_colors, s=.7, alpha=.4)
    colors = matplotlib.cm.get_cmap('tab20c')(np.arange(num_comparisons)/num_comparisons)
    for j in range(num_comparisons):
        for i in range(len(grid)):
            velo = velocity_grid[j][i,:]
            # If the velocity is zero, don't plot it
            if np.abs(velo).sum() == 0:
                continue
            x,y = grid[i,:]
            # Plot an arrow in the center of the grid cell, 
            # pointing in the direction of the velocity
            # TODO change color from sequence from tab20c
            arrow = ax.arrow(x + x_spacing/2, y + y_spacing/2, velo[0], velo[1], 
                             width=0.01, head_width=0.07, head_length=0.05, 
                             color=colors[j])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # Manually specify handles and labels for the legend
    ax.legend([matplotlib.patches.Arrow(0,0,0,0, color=colors[i], width=.1) 
               for i in range(num_comparisons)],
              labels)
    ax.set_title(f'{"vs".join([label.capitalize() for label in labels])}', fontsize=14);

def cell_type_trajectories(nearest_idxs, data, genotype):
    # Plot the proportion of each cell type in the trajectories
    len_trajectory = nearest_idxs.shape[0]
    # Convert the cell type names to integers
    cell_types = {name:i for i,name in enumerate(sorted(data.obs['cell_type'].unique()))}
    num_cell_types = len(cell_types)
    cell_type_trajectories = np.zeros((num_cell_types, len_trajectory), dtype=int)
    # For each point in every trajectory
    for i in range(len_trajectory):
        # Increment the count for the cell type of the nearest cell
        cell_type = data.obs['cell_type'][nearest_idxs[i]]
        cell_type_idxs = cell_type.map(cell_types)
        # Convert the cell_type_idx to a numpy array with num_cell_types elements
        # and a count of the number of times each cell type appears in the trajectory
        cell_type_trajectories[:,i] = np.bincount(cell_type_idxs, minlength=num_cell_types)
    # Normalize the counts to get the proportion of each cell type in the trajectory
    # cell_type_trajectories = cell_type_trajectories / cell_type_trajectories.sum(axis=1)[:,None]
        
    plt.imshow(cell_type_trajectories, aspect='auto', cmap='Blues', interpolation='none')
    # Label the y-axis with the cell type names
    plt.yticks(range(num_cell_types), cell_types);
    plt.title(f'{genotype.capitalize()} cell type proportion in trajectories')
    return cell_type_trajectories

def compare_cell_type_trajectories(nearest_idxs, data, cell_type_to_idx, labels):
    # Plot the proportion of each cell type in the trajectories
    num_comparisons = len(nearest_idxs)
    len_trajectory = nearest_idxs[0].shape[0]
    # Convert the cell type names to integers
    num_cell_types = len(cell_type_to_idx)
    data_cell_types = [d.obs['cell_type'] for d in data]
    cell_type_trajectories = np.zeros((num_comparisons, num_cell_types, len_trajectory), dtype=int)
    # For each point in every trajectory
    for i in range(num_comparisons):
        for j in range(len_trajectory):
            # Increment the count for the cell type of the nearest cell
            sim_cell_types = data_cell_types[i][nearest_idxs[i][j]]
            cell_type_idxs = sim_cell_types.map(cell_type_to_idx)
            # Convert the cell_type_idx to a numpy array with num_cell_types elements
            # and a count of the number of times each cell type appears in the trajectory
            cell_type_counts = np.bincount(cell_type_idxs, minlength=num_cell_types)
            cell_type_trajectories[i, :, j] = cell_type_counts
    # Normalize the counts to get the proportion of each cell type in the trajectory
    # cell_type_trajectories = cell_type_trajectories / cell_type_trajectories.sum(axis=1)[:,None]
    combined_trajectories = np.zeros((num_cell_types*num_comparisons, len_trajectory), dtype=int)
    # Stack the trajectories on top of each other for the plot
    for i in range(num_comparisons):
        combined_trajectories[i::num_comparisons] = cell_type_trajectories[i]
    plt.imshow(combined_trajectories, aspect='auto', cmap='Blues', interpolation='none')
    # Label the y-axis with the cell type names
    # Add another set of ticks on the right side of the plot\
    spacing = 1/num_comparisons
    plt.yticks(np.arange(1,num_cell_types*num_comparisons,num_comparisons)-spacing, cell_type_to_idx);
    # plt.ylim(-spacing, num_cell_types*num_comparisons-spacing)
    for i in np.arange(0,num_cell_types*num_comparisons,num_comparisons)-spacing:
        plt.axhline(i, color='black', linewidth=spacing)
    for i in np.arange(0,num_cell_types*num_comparisons,1)-spacing:
        plt.axhline(i, color='black', linewidth=spacing/3)
    plt.twinx()
    plt.ylim(0, num_cell_types*num_comparisons)
    plt.yticks(ticks=np.arange(0,num_cell_types*num_comparisons,1)+spacing, 
               labels=labels[::-1]*num_cell_types, 
               fontsize=8);
    plt.title(f'{labels[0].capitalize()} vs {labels[1].capitalize()} cell type'
              ' proportion in trajectories')
    return cell_type_trajectories

def cell_type_proportions(proportions, cell_types, labels):
    w = .5
    num_comparisons = len(proportions)
    cell_type_idxs = np.arange(len(cell_types))

    spacing = np.linspace(start=-w/2, 
                          stop=w/2, 
                          num=num_comparisons)
    for i,x in enumerate(spacing):
        plt.bar(x=cell_type_idxs + x,
                height=proportions[i], 
                label=labels[i], 
                width=w/num_comparisons)
    for i in cell_type_idxs:
        plt.axvline(x=i+.5, color='black', linewidth=.5)
    plt.xticks(cell_type_idxs, 
               cell_types, rotation=90);
    plt.ylabel('Proportion of cells')
    plt.title('Overall Cell Type Proportions')
    # Put the legend in the upper right, outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')