#%%
import pickle
import scanpy as sc
from plotting import idx_to_cell_type
import sys
sys.path.append('../')
import util

#%%
genotype = 'wildtype'
tmstp = '20230607_165324'
outdir = f'../../output/{tmstp}'
ko_dir = f'{outdir}/knockout_simulations'
pltdir = f'{outdir}/knockout_simulations/figures'
data = sc.read_h5ad(f'../../data/wildtype_net.h5ad')


#%%
with open(f'{outdir}/baseline_trajectories_{genotype}.pickle', 'rb') as f:
    trajectories = pickle.load(f)
with open(f'{outdir}/baseline_nearest_cell_idxs_{genotype}.pickle', 'rb') as f:
    nearest_idxs = pickle.load(f)
# %%
# For each simulation, find the cell type of the nearest cell in the baseline
cell_types = {c:i for i,c in enumerate(sorted(set(data.obs['cell_type'])))}
nearest_cell_types = idx_to_cell_type(nearest_idxs, data, cell_types)

# %%
# Calculate the distance between the pairs of trajectory vectors
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
#%%
# Get all unique pairs of trajectories
num_trajectories = trajectories.shape[1]
dists = np.zeros((num_trajectories, num_trajectories))
progress = tqdm(total=num_trajectories*(num_trajectories+1)/2)
pairs = np.triu_indices(num_trajectories)
for i in range(num_trajectories):
    for j in range(i+1,num_trajectories):
        # dist = ((trajectories[:,i,:] - trajectories[:,j,:])**2).sum()
        dist = 1-((nearest_cell_types[:,i] == nearest_cell_types[:,j])).sum()/(nearest_cell_types.shape[0])
        dists[i,j] = dist
        # dists[j,i] = dist
        tqdm.update(progress, 1)
# Make the distance matrix symmetric
dists = dists + dists.T

# # %%
#%%
# UMAP embed the trajectories using distance matrix
import umap
reducer = umap.UMAP(metric='precomputed')
embedding = reducer.fit_transform(dists)
# %%
n_repeats = 10
n_cells = 82
distinct_colors = np.array(util.distinct_colors(n_cells))

colors = distinct_colors[np.arange(n_cells).repeat(n_repeats)]
#%%
from matplotlib import pyplot as plt
plt.scatter(embedding[:,0], embedding[:,1], s=.2, c=colors)
#%%
# %%
