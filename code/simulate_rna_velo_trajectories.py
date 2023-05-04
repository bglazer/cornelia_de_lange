#%%
import scanpy as sc
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random
import scvelo as scv

#%% 
# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

#%%
genotype = 'mutant'
adata = sc.read_h5ad(f'../data/{genotype}_net.h5ad')
# %%
# TODO need to get all the RNA velo matrices into the adata object
# TODO but first we need to make the prepocessing consistent between my analysis and Stephens'
scv.tl.velocity_graph(adata)
T = scv.utils.get_transition_matrix(adata)
# %%
initial_idxs = adata.uns['initial_points_via']
len_trajectory = 500
num_trajectories = 1000
n_cells = T.shape[0]
#%%
def trajectory():
    traj = np.zeros(len_trajectory, dtype=int)
    initial = np.random.choice(initial_idxs)
    traj[0] = initial
    for j in range(1, len_trajectory):
        p = T[traj[j-1]].toarray().flatten()
        next_step = np.random.choice(range(n_cells), p=p)
        traj[j] = next_step
    return traj
# trajectory()
#%%
trajectories = []
with Parallel(n_jobs=30, verbose=10) as parallel:
    trajectories = parallel(delayed(trajectory)() for i in range(num_trajectories))
#%%
trajectories = np.asarray(trajectories)
# %%
# Get a list of the cell type names
cell_types = {name:i for i,name in enumerate(sorted(adata.obs['cell_type'].unique()))}
num_cell_types = len(cell_types)

#%%
cell_type_trajectories = np.zeros((len_trajectory, num_cell_types), dtype=int)
for i in range(num_trajectories):
    for j in range(len_trajectory):
        cell_type = adata.obs['cell_type'][trajectories[i,j]]
        cell_type_trajectories[j, cell_types[cell_type]] += 1
    
# %%
plt.imshow(cell_type_trajectories.T, aspect='auto', cmap='Blues', interpolation='none')
# Label the y-axis with the cell type names
plt.yticks(range(num_cell_types), cell_types);
plt.title(f'{genotype.capitalize()} cell type proportion in trajectories')
plt.savefig(f'../figures/{genotype}_celltype_trajectories.png', dpi=300)

# %%
# Get the archetype of each cell in the trajectories
archetypes = adata.obs['specialists_pca_diffdist'].tolist()
# Replace the nan with 'Generalist'
for i,archetype in enumerate(archetypes):
    if type(archetype) == float:
        archetypes[i] = 'Generalist'

archetype_idx = {name:i for i,name in enumerate(sorted(set(archetypes)))}
#%%
archetype_trajectories = np.zeros((len_trajectory, len(archetype_idx)), dtype=int)
for i in range(num_trajectories):
    for j in range(len_trajectory):
        archetype = archetypes[trajectories[i,j]]
        archetype_trajectories[j, archetype_idx[archetype]] += 1
# %%
plt.imshow(archetype_trajectories.T, aspect='auto', cmap='Blues', interpolation='none')
# Label the y-axis with the archetype names
plt.yticks(range(len(archetype_idx)), archetype_idx);
plt.title(f'{genotype.capitalize()} archetype proportion in trajectories')
# %%
# Get a color map with len_trajectory colors
cmap = plt.get_cmap('viridis')
colors = [cmap(i/len_trajectory) for i in range(len_trajectory)]
# Plot a scatter plot showing individual trajectories
umap = adata.obsm['X_umap']
# Get a 4x4 grid of plots
nrow = 3
ncol = 3
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20,20))
# Choose a random sample of 16 trajectories
for i,idx in enumerate(random.sample(range(num_trajectories), nrow*ncol)):
    # Plot the scatter plot
    ax = axs[i//ncol, i%ncol]
    ax.scatter(umap[:,0], umap[:,1], color='grey', alpha=0.1, s=.1)
    # Plot the trajectory
    for j in range(len_trajectory):
        idx = trajectories[i,j]
        prev_idx = trajectories[i,j-1]
        # print(j, pct)
        # Draw lines between points in the trajectory
        if j > 0:
            ax.plot([umap[idx,0], umap[prev_idx,0]], 
                    [umap[idx,1], umap[prev_idx,1]], c=colors[j], alpha=0.4)
    # Remove the axis labels
    ax.set_xticks([])
    ax.set_yticks([])

# Add a title
fig.suptitle(f'{genotype.capitalize()} sampled trajectories', fontsize=30)
plt.savefig(f'../figures/trajectories/{genotype}_sampled_trajectories.png', dpi=300)
