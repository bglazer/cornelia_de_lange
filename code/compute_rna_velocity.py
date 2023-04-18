# %%
import numpy as np
import scanpy as sc
from util import umap_axes
import scvelo as scv

# %%
# Load the RNA velocity data
wt_vel = sc.read('../data/wildtype_velocity_network.h5ad')
mut_vel = sc.read('../data/mutant_velocity_network.h5ad')
# Filter to only the genes that are in the network data
wt_vel = wt_vel[:, wt_vel.var_names.isin(wt.var_names)]
mut_vel = mut_vel[:, mut_vel.var_names.isin(mut.var_names)]

# %%
# Add the precomputed UMAP to the velocity data
wt_vel.obsm['X_umap'] = wt_emb
mut_vel.obsm['X_umap'] = mut_emb



#%%
# Plot the velocity embedding from the original, unfiltered data
scv.pl.velocity_embedding_stream(wt_vel)
scv.pl.velocity_embedding_stream(mut_vel)

#%%
# Recompute the transition matrix using only the cells that are in the network
scv.tl.velocity(wt_vel)
scv.tl.velocity(mut_vel)
scv.tl.transition_matrix(wt_vel)
scv.tl.transition_matrix(mut_vel)
scv.tl.velocity_pseudotime(wt_vel)
scv.tl.velocity_pseudotime(mut_vel)
# Replot the velocity embedding stream
scv.pl.velocity_embedding_stream(wt_vel)
scv.pl.velocity_embedding_stream(mut_vel)

# %%
# Plot the RNA-velocity pseudotime
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].scatter(wt_emb[:,0], wt_emb[:,1], 
            c=wt_vel.obs['velocity_pseudotime'], cmap='gnuplot',
            s=.5, alpha=.5)
axs[1].scatter(mut_emb[:,0], mut_emb[:,1], 
            c=mut_vel.obs['velocity_pseudotime'], cmap='gnuplot',
            s=.5, alpha=.5)
umap_axes(axs)
# Add a colorbar
fig.colorbar(plt.cm.ScalarMappable(cmap='gnuplot'), ax=axs.ravel().tolist(), label='Pseudotime')

# %%
scv.pl.scatter(wt_vel, color='velocity_pseudotime', cmap='gnuplot')
scv.pl.scatter(mut_vel, color='velocity_pseudotime', cmap='gnuplot')

# %%
# Compute the cosine similarity between the velocity pseudotime and 
# the VIA pseudotime velocity vectors
from util import velocity_vectors
wt_X = wt_via.data
wt_T = wt_via.sc_transition_matrix(smooth_transition=1)

wt_V_via = velocity_vectors(wt_T, wt_X)
wt_V_rna = wt_vel.layers['velocity']
wt_V_nrm_via = wt_V_via / np.linalg.norm(wt_V_via, axis=1)[:,None]
wt_V_nrm_rna = wt_V_rna / np.linalg.norm(wt_V_rna, axis=1)[:,None]
# Cosine similarity
wt_cos_sim = np.sum(wt_V_nrm_via * wt_V_nrm_rna, axis=1)
wt_eucl_dist = np.linalg.norm(wt_V_via - wt_V_rna, axis=1)

# Repeat for the mutant
mut_X = mut_via.data
mut_T = mut_via.sc_transition_matrix(smooth_transition=1)

mut_V_via = velocity_vectors(mut_T, mut_X)
mut_V_rna = mut_vel.layers['velocity']
mut_V_nrm_via = mut_V_via / np.linalg.norm(mut_V_via, axis=1)[:,None]
mut_V_nrm_rna = mut_V_rna / np.linalg.norm(mut_V_rna, axis=1)[:,None]
# Cosine similarity
mut_cos_sim = np.sum(mut_V_nrm_via * mut_V_nrm_rna, axis=1)
mut_eucl_dist = np.linalg.norm(mut_V_via - mut_V_rna, axis=1)

# %%
wt_cos_sim_nrm = (wt_cos_sim + 1)/2
mut_cos_sim_nrm = (mut_cos_sim + 1)/2



# %%
fig, axs = plt.subplots(1,2, figsize=(10,5))
# Set the color range to be the same for both plots
vmin = -1
vmax = 1
axs[0].scatter(wt_emb[:,0], wt_emb[:,1], c=wt_cos_sim, cmap='gnuplot', s=.2, alpha=.5, vmin=vmin, vmax=vmax)
axs[1].scatter(mut_emb[:,0], mut_emb[:,1], c=mut_cos_sim, cmap='gnuplot', s=.2, alpha=.5, vmin=vmin, vmax=vmax)
fig.colorbar(plt.cm.ScalarMappable(cmap='gnuplot'), ax=axs.ravel().tolist())
umap_axes(axs)

# %%
# UMAP embedding of the euclidean distances between the RNA velocity and VIA velocity vectors
fig, axs = plt.subplots(1,2, figsize=(10,5))
# Set the color range to be the same for both plots
vmin = 0
vmax = np.max((wt_eucl_dist.max(), mut_eucl_dist.max()))
axs[0].scatter(wt_emb[:,0], wt_emb[:,1], c=wt_eucl_dist, cmap='gnuplot', s=.2, alpha=.5, vmin=vmin, vmax=vmax)
axs[1].scatter(mut_emb[:,0], mut_emb[:,1], c=mut_eucl_dist, cmap='gnuplot', s=.2, alpha=.5, vmin=vmin, vmax=vmax)
# Add a colorbar with the range vmin to vmax
fig.colorbar(plt.cm.ScalarMappable(cmap='gnuplot'), ax=axs.ravel().tolist(), 
             label='Euclidean distance')
umap_axes(axs)

# %%
# Plot the distribution of the cosine similarity
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].hist(wt_cos_sim, bins=100);
axs[1].hist(mut_cos_sim, bins=100);

# Plot the distribution of the euclidean distance
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].hist(wt_eucl_dist, bins=100);
axs[1].hist(mut_eucl_dist, bins=100);

# %%
# Compare RNA velocity between the mutant and its closest wildtype neighbor in VIA velocity space
# First, find the closest wildtype neighbor
wt_neighb_vel = wt_V_nrm_via[neighbor_idxs,:]
# Compute the cosine similarity between the RNA velocity vectors
wt_neighb_cos_sim = np.sum(mut_V_nrm_rna * wt_neighb_vel, axis=1)
wt_neighb_cos_sim_nrm = (wt_neighb_cos_sim + 1)/2
# Plot the distribution of the cosine similarity, 
# then the scatter plot of the UMAP embedding colored by the cosine similarity
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].hist(wt_neighb_cos_sim, bins=100);
axs[1].scatter(mut_emb[:,0], mut_emb[:,1], c=wt_neighb_cos_sim_nrm, cmap='gnuplot', s=.1, alpha=.5)
umap_axes(axs[1:])
# Add a colorbar
fig.colorbar(plt.cm.ScalarMappable(cmap='gnuplot'), ax=axs.ravel().tolist(), label='Cosine similarity');

# %%
# Plot the similarity between the mutant and its closest wildtype neighbor in RNA velocity
# make a dense matrix of expression
wt_rna_neighb_vel = wt_V_nrm_rna[neighbor_idxs,:]
wt_rna_neighb_cos_sim = np.sum(mut_V_nrm_rna * wt_rna_neighb_vel, axis=1)
wt_rna_neighb_cos_sim_nrm = (wt_rna_neighb_cos_sim + 1)/2
# Plot the distribution of the cosine similarity, then the scatter plot colored by velocity similarity
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].hist(wt_rna_neighb_cos_sim, bins=100);
axs[1].scatter(mut_emb[:,0], mut_emb[:,1], c=wt_rna_neighb_cos_sim_nrm, cmap='gnuplot', s=.1, alpha=.5, vmin=0, vmax=1)
umap_axes(axs[1:])
# Add a colorbar
fig.colorbar(plt.cm.ScalarMappable(cmap='gnuplot'), ax=axs.ravel().tolist(), label='Cosine similarity');
