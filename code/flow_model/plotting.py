import sys
import matplotlib
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import random
import torch
sys.path.append('..')
import util
from scipy.stats import gaussian_kde as kde

def time_distribution(trajectories, pca, label, baseline=None):
    # Plot a scatter plot showing the overall distribution of points in the trajectories
    # Get the first two principal components of the data
    t_emb = pca.transform(trajectories.reshape(-1, trajectories.shape[2]))[:,:2]
    t_emb = t_emb.reshape(trajectories.shape[0], trajectories.shape[1], -1)
    # Connect the scatter plot points
    # Color the points by the distance along the trajectory
    colormap = matplotlib.colormaps.get_cmap('gnuplot')
    len_trajectory = trajectories.shape[0]
    colors = colormap(np.arange(len_trajectory)/len_trajectory)
    fig, axs = plt.subplots(1,1, figsize=(10,10))

    for i in range(t_emb.shape[1]):
        # plt.plot(t_emb[:,i,0], t_emb[:,i,1], color='black', linewidth=.5, alpha=.1)
        axs.scatter(t_emb[:,i,0], t_emb[:,i,1],
                    color=colors, 
                    marker='o', s=.5, alpha=0.5)
    if baseline is not None:
        baseline_emb = pca.transform(baseline)[:,0:2]
        axs.scatter(baseline_emb[:,0], baseline_emb[:,1], color='black', alpha=.2, s=.25)
    cax = fig.add_axes([1.01, .1, .02, .8])
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=colormap), cax=cax, label='Time')
    cbar.ax.tick_params(labelsize=15)
    # Increase the font size of the colorbar label
    cbar.ax.yaxis.label.set_size(18)
    # Remove the axis labels
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlabel('PC1', fontsize=16)
    axs.set_ylabel('PC2', fontsize=16)
    axs.set_title(f'{label.capitalize()}', fontsize=20)

def colorize_trajectory(nearest_idxs, data, cell_type_to_idx):
    cell_colors = util.distinct_colors(len(cell_type_to_idx))

    cell_type_traj = idx_to_cell_type(nearest_idxs, data, 
                                      cell_type_to_idx)
    cell_color_traj = np.array(cell_colors)[cell_type_traj]
    return cell_color_traj

def cell_colors(cell_type_to_idx):
    return util.distinct_colors(len(cell_type_to_idx))

def cell_type_distribution(trajectories, nearest_idxs, data, cell_type_to_idx, pca, label, baseline=None, s=1, scatter_alpha=0.1):
    # Plot a scatter plot showing the overall distribution of points in the trajectories
    # Get the first two principal components of the data
    t_emb = pca.transform(trajectories.reshape(-1, trajectories.shape[2]))[:,:2]
    t_emb = t_emb.reshape(trajectories.shape[0], trajectories.shape[1], -1)

    cell_colors = util.distinct_colors(len(cell_type_to_idx))

    cell_type_traj = idx_to_cell_type(nearest_idxs, data, 
                                      cell_type_to_idx)
    cell_color_traj = np.array(cell_colors)[cell_type_traj]

    # Connect the scatter plot points
    fig, axs = plt.subplots(1,1, figsize=(12,10))


    for i in range(t_emb.shape[1]):
        # plt.plot(t_emb[:,i,0], t_emb[:,i,1], color='black', linewidth=.5, alpha=.1)
        axs.scatter(t_emb[:,i,0], t_emb[:,i,1],
                    color=cell_color_traj[:,i], 
                    marker='.', s=s, alpha=scatter_alpha)
    # Add a legend outside the plot on the right side with the cell type colors
    plt.legend([matplotlib.patches.Patch(color=color) for color in cell_colors],
                list(cell_type_to_idx),
                bbox_to_anchor=(1.05, 1), loc='upper left',
                prop={'size': 18})
    
    if baseline is not None:
        baseline_emb = pca.transform(baseline)[:,0:2]
        x = baseline_emb[:,0]
        y = baseline_emb[:,1]
        xmin, xmax = axs.get_xlim()
        ymin, ymax = axs.get_ylim()
        k = kde([x,y])
        # 1j is the imaginary unit, which is used as a flag to mgrid to tell it 
        # a number of points to make in the grid
        xi, yi = np.mgrid[xmin:xmax:x.size**0.5*1j, 
                        ymin:ymax:y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        axs.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
        
        # # axs.scatter(baseline_emb[:,0], baseline_emb[:,1], color='black', alpha=.2, s=s*.25)
        # xedges = np.linspace(xlim[0], xlim[1], 100)
        # yedges = np.linspace(ylim[0], ylim[1], 100)
        # baseline_density, xedges, yedges = np.histogram2d(baseline_emb[:,0], baseline_emb[:,1], bins=[xedges, yedges])
        # # Plot the density as a contour plot
        # axs.contour(xedges[:-1], yedges[:-1], baseline_density.T, cmap='copper', levels=10, linewidths=1)
    
    # Remove the axis labels
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlabel('PC1', fontsize=16)
    axs.set_ylabel('PC2', fontsize=16)
    axs.set_title(f'{label}', fontsize=24)
    plt.tight_layout()
        
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
    ax.set_facecolor('#edf2f4')
    # Add lines to show the grid
    for x in x_grid_points:
        ax.axvline(x, color='white', alpha=.7)
    for y in y_grid_points:
        ax.axhline(y, color='white', alpha=.7)
    #Plot the velocity vectors
    ax.scatter(proj[0][:,0], proj[0][:,1], c=cell_colors, s=.7, alpha=.2)
    # TODO need to make a longer sequence of colors
    arrow_colors = ['#14213d', '#fca311']
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
                             color=arrow_colors[j])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # Manually specify handles and labels for the legend
    ax.legend([matplotlib.patches.Arrow(0,0,0,0, color=arrow_colors[i], width=.1) 
               for i in range(num_comparisons)],
              labels)
    ax.set_title(f'{" vs ".join([label.capitalize() for label in labels])}', fontsize=14);

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

def idx_to_cell_type(nearest_idxs, data, cell_type_to_idx):
    data_cell_types = data.obs['cell_type']
    
    # Convert nearest cell indexes to their corresponding cell types
    # map converts cell type name to an index
    # to_numpy converts the pandas series to a numpy array
    # reshape converts the 1D array to a 2D array with the same shape as nearest_idxs
    cell_types = data_cell_types.iloc[nearest_idxs.flatten()]\
                 .map(cell_type_to_idx).to_numpy()\
                 .reshape(nearest_idxs.shape)
    return cell_types

def calculate_cell_type_proportion(nearest_idxs, data, cell_type_to_idx, n_repeats, error=False):
    # Convert nearest cell indexes to their corresponding cell types
    cell_types = idx_to_cell_type(nearest_idxs, data, cell_type_to_idx)
    # Then count how many of each cell type appear in each trajectory
    cell_type_counts = np.apply_along_axis(np.bincount, 
                                           axis=0, arr=cell_types, 
                                           minlength=len(cell_type_to_idx))
    # Then compute the standard deviation of the cell type proportions
    cell_type_means = (cell_type_counts / cell_types.shape[0]).mean(axis=(1))
    # For the standard deviation, we calculate the means of the starting cells,
    #  then calculate the standard deviation across the repeats
    cell_type_stds = (cell_type_counts / cell_types.shape[0]).reshape((cell_type_counts.shape[0], -1, n_repeats)).mean(axis=1).std(axis=1)

    if error:
        return cell_type_means, cell_type_stds
    else:
        return cell_type_means
    

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
            sim_cell_types = data_cell_types[i].iloc[nearest_idxs[i][j]]
            cell_type_idxs = sim_cell_types.map(cell_type_to_idx)
            # Convert the cell_type_idx to a numpy array with num_cell_types elements
            # and a count of the number of times each cell type appears in the trajectory
            cell_type_counts = np.bincount(cell_type_idxs, minlength=num_cell_types)
            cell_type_trajectories[i, :, j] = cell_type_counts
    # Normalize the counts to get the proportion of each cell type in the trajectory
    cell_type_trajectories = cell_type_trajectories / cell_type_trajectories.sum(axis=(1,2))[:,None,None] * len_trajectory
    combined_trajectories = np.zeros((num_cell_types*num_comparisons, len_trajectory), dtype=float)
    # Stack the trajectories on top of each other for the plot
    for i in range(num_comparisons):
        combined_trajectories[i::num_comparisons] = cell_type_trajectories[i]
    plt.imshow(combined_trajectories, aspect='auto', cmap='viridis', interpolation='none')
    # Label the y-axis with the cell type names
    # Add another set of ticks on the right side of the plot\
    spacing = 1/num_comparisons
    plt.yticks(np.arange(1,num_cell_types*num_comparisons,num_comparisons)-spacing, cell_type_to_idx);
    # plt.ylim(-spacing, num_cell_types*num_comparisons-spacing)
    for i in np.arange(0,num_cell_types*num_comparisons,num_comparisons)-spacing:
        plt.axhline(i, color='black', linewidth=spacing)
    for i in np.arange(0,num_cell_types*num_comparisons,1)-spacing:
        plt.axhline(i, color='black', linewidth=spacing/3)
    plt.ylabel('Cell Type')
    plt.xlabel('Time')

    plt.twinx()
    plt.ylim(0, num_cell_types*num_comparisons)
    plt.yticks(ticks=np.arange(0,num_cell_types*num_comparisons,1)+spacing, 
               labels=labels[::-1]*num_cell_types, 
               fontsize=8);
    plt.title(f'{labels[0].capitalize()} vs {labels[1].capitalize()} cell type'
              ' proportion in trajectories')
    
    plt.colorbar(label='Proportion of cells', orientation='vertical', pad=0.15, aspect=30)
    return cell_type_trajectories

def cell_type_proportions(proportions, proportion_errors, cell_types, labels, colors=None):
    w = .5
    num_comparisons = len(proportions)
    cell_type_idxs = np.arange(len(cell_types))

    spacing = np.linspace(start=-w/2, 
                          stop=w/2, 
                          num=num_comparisons)
    if colors is None:
        colors = plt.cm.tab20(np.linspace(0,1,num_comparisons))

    for i,x in enumerate(spacing):
        plt.bar(x=cell_type_idxs + x,
                height=proportions[i], 
                label=labels[i], 
                width=w/num_comparisons,
                # Set the color of the bar
                color=colors[i])
        y_high = proportion_errors[i]*2
        y_low = proportion_errors[i]*2
        lt0 = (proportions[i] - proportion_errors[i]*2) < 0
        y_low[lt0] = proportions[i][lt0]
        plt.errorbar(x=cell_type_idxs + x,
                     y=proportions[i],
                     yerr=(y_low, y_high),
                     c='grey',
                     fmt='none')
    for i in cell_type_idxs:
        plt.axvline(x=i+.5, color='black', linewidth=.5)
    plt.xticks(cell_type_idxs, 
               cell_types, rotation=90);
    plt.ylabel('Proportion of cells')
    plt.title('Overall Cell Type Proportions')
    # Put the legend in the upper right, outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

#%%
from itertools import combinations, combinations_with_replacement
import pickle
from mip import Model, MINIMIZE, BINARY, xsum
protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}

# Plot the paths
def make_levels(paths):
    path_lens = [len(path) for path in paths]
    max_len = max(path_lens)
    levels = [set() for i in range(max_len)]
    for path in paths:
        for i,node in enumerate(path[::-1]):
            levels[i].add(node)
    widths = [len(level) for level in levels]
    max_width = max(widths)
    for j,level in enumerate(levels):
        level = list(level)
        levels[j] = {pid:i for i,pid in enumerate(level)}
    return levels, widths

def plot_paths(levels, paths, center=False):
    _, widths = make_levels(paths)
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    xy = {}
    for y,level in enumerate(levels):
        ys = [y for _ in range(len(level))]
        xs = level.values()
        if center:
            xs = [x - widths[y]/2 for x in xs]
        xy.update({pid:(x,y) for pid,x in zip(level.keys(), xs)})
        
    colors= plt.cm.tab20(np.linspace(0,1,len(paths)))
    starts = {path[0]:0 for path in paths}
    for j, path in enumerate(paths):
        # Color the starting node the same color as the path
        if len(path) > 1:
            pid0 = path[0]
            ax.scatter(xy[pid0][0], xy[pid0][1])#, s=300-starts[pid0]*90, color=colors[j], zorder=starts[pid0])
            # print('starts', protein_id_name[pid0], 300-starts[pid0]*90)
            starts[pid0] += 1
        path = path[::-1]
        for i in range(len(path)-1):
            pid1 = path[i]
            pid2 = path[i+1]
            x1,y1 = xy[pid1]
            x2,y2 = xy[pid2]
            ax.plot([x1,x2],[y1,y2], c=colors[j])
        for i in range(0,len(path)-1):
            x,y = xy[path[i]]
            # ax.scatter(x, y, s=100, c='grey')

    for pid,(x,y) in xy.items():
        name = protein_id_name[pid]
        # ax.text(x-len(name)/15, y+.05, name, fontsize=10, fontdict={'family':'monospace'})
        ax.text(x, y, name, fontsize=10, fontdict={'family':'monospace'})
    ax.set_xticks([])
    ax.set_yticks([]);

# Based on https://doi.org/10.1109/PacificVis.2018.00025
def optimize_placement(paths, max_count=None, verbose=False):
    levels, widths = make_levels(paths)
    model = Model(sense=MINIMIZE)
    above = [{} for level in levels]

    if verbose:
        print('Adding variables', flush=True)
    # Decision variables
    for j, level in enumerate(levels):
        combos = list(combinations_with_replacement(level, r=2))
        for i, (pid1, pid2) in enumerate(combos):
            # Add a binary variable for each pair of nodes, 
            # indicating whether pid1 is above pid2
            above[j][f'x_{pid1}_{pid2}'] = model.add_var(var_type=BINARY)
            above[j][f'x_{pid2}_{pid1}'] = model.add_var(var_type=BINARY)
            # Add the constraint that one of the nodes must be above the other
            model.add_constr(above[j][f'x_{pid1}_{pid2}'] + above[j][f'x_{pid2}_{pid1}'] == 1)
    for j, level in enumerate(levels):
        combos = list(combinations(level, 3))
        for i, (pid1, pid2, pid3) in enumerate(combos):
            # Add a second order transitivity constraint, 
            # so that if pid1 is above pid2, and pid2 is above pid3,
            # then pid1 is above pid3
            model.add_constr(above[j][f'x_{pid3}_{pid1}'] >= above[j][f'x_{pid3}_{pid2}'] + above[j][f'x_{pid2}_{pid1}'] - 1)
    # Convert paths to individual links
    links = [[] for i in range(len(levels)-1)]
    for path in paths:
        path = path[::-1]
        for i in range(len(path)-1):
            pid1 = path[i]
            pid2 = path[i+1]
            links[i].append((pid1, pid2))
    # Create variables indicating whether two links are crossing
    crossings = {}
    for i, level in enumerate(links):
        combos = list(combinations(level, 2))
        for j, (link1, link2) in enumerate(combos):
            u1, v1 = link1
            u2, v2 = link2
            crossings[f'c_({u1}_{v1})_({u2}_{v2})'] = model.add_var(var_type=BINARY)
            # Add constraint that activates the crossing variable if the links are crossing
            model.add_constr(above[i][f'x_{u2}_{u1}'] + above[i+1][f'x_{v1}_{v2}'] + crossings[f'c_({u1}_{v1})_({u2}_{v2})'] >= 1)
            model.add_constr(above[i][f'x_{u1}_{u2}'] + above[i+1][f'x_{v2}_{v1}'] + crossings[f'c_({u1}_{v1})_({u2}_{v2})'] >= 1)
            if f'c_({u1}_{v1})_({u2}_{v2})' in crossings and f'c_({u1}_{v2})_({u2}_{v1})' in crossings:
                model.add_constr(crossings[f'c_({u1}_{v1})_({u2}_{v2})'] + crossings[f'c_({u1}_{v2})_({u2}_{v1})'] == 1)
    # Add objective function
    if verbose:
        print('Adding objective')
    model.objective = xsum(list(crossings.values()))
    # Optimize
    if verbose:
        print('Optimizing', flush=True)
    if max_count is None:
        status = model.optimize()
    else:
        status = model.optimize(max_seconds=max_count)
    if verbose:
        print(status)
    # Sort the nodes by their rank, i.e. how many other nodes they're above
    # this gives the overall ordering of the nodes that minimizes crossings
    ordering = []
    for i,level in enumerate(above):
        above_counts = {k:0 for k in levels[i].keys()}
        for var, value in level.items():
            n1, n2 = var.split('_')[1:]
            v = int(value.x)
            above_counts[n1] += v
        sorted_counts = sorted(above_counts.items(), key=lambda x: x[1], reverse=True)
        ordering.append({pid: rank for rank,(pid,count) in enumerate(sorted_counts)})
    return ordering
# %%
