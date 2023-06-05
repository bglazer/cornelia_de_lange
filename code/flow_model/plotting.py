import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
import torch
sys.path.append('..')
import util

def distribution(trajectories, pca):
    # Plot a scatter plot showing the overall distribution of points in the trajectories
    t = util.tonp(trajectories)
    t_emb = pca.transform(t.reshape(-1, t.shape[2]))[:,:2]
    t_emb = t_emb.reshape(t.shape[0], t.shape[1], -1)
    # Connect the scatter plot points
    # Color the points by the distance along the trajectory
    colormap = matplotlib.colormaps.get_cmap('gnuplot')
    len_trajectory = t.shape[0]
    colors = colormap(np.arange(len_trajectory)/len_trajectory)
    # plt.scatter(proj[:,0], proj[:,1], s=.1,
    #             c=cell_colors, cmap='tab20c')
    for i in range(t_emb.shape[1]):
        # plt.plot(t_emb[:,i,0], t_emb[:,i,1], color='black', linewidth=.5, alpha=.1)
        plt.scatter(t_emb[:,i,0], t_emb[:,i,1],
                    color=colors, 
                    marker='o', s=.5, alpha=0.5)
        
#%%
# Plot a sample of the individual trajectories
def sample_trajectories(trajectories, X, pca, genotype):
    # Get a 4x4 grid of plots
    nrow = 3
    ncol = 3
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20,20))
    # Choose a random sample of 16 trajectories
    num_trajectories = trajectories.shape[1]
    proj = np.array(pca.transform(X)[:,0:2])
    t = util.tonp(trajectories)
    t_emb = pca.transform(t.reshape(-1, t.shape[2]))[:,:2]
    t_emb = t_emb.reshape(t.shape[0], t.shape[1], -1)
    for i,idx in enumerate(random.sample(range(num_trajectories), nrow*ncol)):
        # Plot the scatter plot
        ax = axs[i//ncol, i%ncol]
        ax.scatter(proj[:,0], proj[:,1], color='grey', alpha=1, s=.1)
        # Plot the trajectory
        ax.plot(t_emb[:,idx,0], t_emb[:,idx,1], color='black', linewidth=.5)
        # ax.scatter(t_emb[:,idx,0], t_emb[:,idx,1], c=colors, 
        #            marker='o', s=5, zorder=100)
        # Remove the axis labels
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'{genotype.capitalize()} sampled trajectories', fontsize=30)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

def arrow_grid(data, pca, model, genotype):
    X = data.X.toarray()
    proj = np.array(pca.transform(X))[:,0:2]
    X = torch.tensor(X).float()
    # Find the extent of the scatter plot
    minX = np.min(proj[:,0])
    maxX = np.max(proj[:,0])
    minY = np.min(proj[:,1])
    maxY = np.max(proj[:,1])
    # Add some buffer to the sides of the plot
    xbuf = (maxX - minX) * 0.05
    ybuf = (maxY - minY) * 0.05
    # Set the granularity of the grid, i.e. how many grids to divide the space into
    # in both the x and y directions
    n_points = 20
    x_grid_points = np.linspace(minX, maxX, n_points)
    y_grid_points = np.linspace(minY, maxY, n_points)
    x_spacing = x_grid_points[1] - x_grid_points[0]
    y_spacing = y_grid_points[1] - y_grid_points[0]
    grid = np.array(np.meshgrid(x_grid_points, y_grid_points)).T.reshape(-1,2)
    velocity_grid = np.zeros_like(grid)
    velocity_grid[:] = np.nan

    # Find points inside each grid cell
    velocities = torch.zeros(len(grid), X.shape[1])
    variances = torch.zeros(len(grid), X.shape[1])
    grid_means = torch.zeros(len(grid), X.shape[1])
    for i,(x,y) in enumerate(grid):
        # Find the points inside the grid cell
        idx = np.where((proj[:,0] > x) & (proj[:,0] < x+x_spacing) & 
                        (proj[:,1] > y) & (proj[:,1] < y+y_spacing))[0]
        # If there are any points inside the grid cell
        if len(idx) > 0:
            # Get the average velocity vector for the points 
            # inside the grid cell
            velo, var = model(X[idx,:])
            velocities[i] = velo.mean(axis=0).reshape(-1)
            variances[i] = var.mean(axis=0).reshape(-1)
            grid_means[i] = X[idx,:].mean(axis=0)
    velocities = util.tonp(velocities)
    variances = util.tonp(variances)
    pca_embed = lambda x: np.array(pca.transform(x)[:,0:2])
    velocity_grid = util.embed_velocity(grid_means, velocities, pca_embed)

    fig, ax = plt.subplots(1,1, figsize=(10,10))

    cell_types = {name:i for i,name in enumerate(sorted(data.obs['cell_type'].unique()))}
    cell_colors = data.obs['cell_type'].map(cell_types)
    ax.set_xlim(minX-xbuf, maxX+xbuf)
    ax.set_ylim(minY-ybuf, maxY+ybuf)
    ax.set_facecolor('grey')
    # Add lines to show the grid
    for x in x_grid_points:
        ax.axvline(x, color='white', alpha=0.1)
    for y in y_grid_points:
        ax.axhline(y, color='white', alpha=0.1)
    #Plot the velocity vectors
    ax.scatter(proj[:,0], proj[:,1], c=cell_colors, s=.7)
    for i in range(len(grid)):
        velo = velocity_grid[i,:]
        # If the velocity is zero, don't plot it
        if np.abs(velo).sum() == 0:
            continue
        x,y = grid[i,:]
        # Plot an arrow in the center of the grid cell, 
        # pointing in the direction of the velocity
        ax.arrow(x + x_spacing/2, y + y_spacing/2, velo[0], velo[1], 
                width=0.025, head_width=0.1, head_length=0.05, 
                color='orange')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{genotype.capitalize()}', fontsize=14);

def cell_type_proportions(trajectories, data, kdtree, genotype):
    # Plot the proportion of each cell type in the trajectories
    len_trajectory = trajectories.shape[0]
    num_trajectories = trajectories.shape[1]
    # Convert the cell type names to integers
    cell_types = {name:i for i,name in enumerate(sorted(data.obs['cell_type'].unique()))}
    num_cell_types = len(cell_types)
    cell_type_trajectories = np.zeros((len_trajectory, num_cell_types), dtype=int)
    # For each point in every trajectory
    for i in range(len_trajectory):
        for j in range(num_trajectories):
            # Find the nearest cell in the dataset
            nearest_cell = kdtree.query(trajectories[i,j,:].reshape(1,-1))[1][0][0]
            # Increment the count for the cell type of the nearest cell
            cell_type = data.obs['cell_type'][nearest_cell]
            cell_type_trajectories[i,cell_types[cell_type]] += 1
        
    # %%
    plt.imshow(cell_type_trajectories.T, aspect='auto', cmap='Blues', interpolation='none')
    # Label the y-axis with the cell type names
    plt.yticks(range(num_cell_types), cell_types);
    plt.title(f'{genotype.capitalize()} cell type proportion in trajectories')