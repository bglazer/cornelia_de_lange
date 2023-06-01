#%%
import sys
sys.path.append('..')
import util
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%%
wt = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut = sc.read_h5ad(f'../../data/mutant_net.h5ad')
adata = wt.concatenate(mut, batch_key='genotype', batch_categories=['wildtype', 'mutant'])
# %%
# Arrow grid
cell_types = {c:i for i,c in enumerate(set(wt.obs['cell_type']))}
proj = np.array(adata.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = wt.uns['pca_mean']
pca.components_ = wt.uns['PCs']
minX = np.min(proj[:,0])
maxX = np.max(proj[:,0])
minY = np.min(proj[:,1])
maxY = np.max(proj[:,1])
xbuf = (maxX - minX) * 0.05
ybuf = (maxY - minY) * 0.05

n_points = 20
x_grid_points = np.linspace(minX, maxX, n_points)
y_grid_points = np.linspace(minY, maxY, n_points)
x_spacing = x_grid_points[1] - x_grid_points[0]
y_spacing = y_grid_points[1] - y_grid_points[0]
grid = np.array(np.meshgrid(x_grid_points, y_grid_points)).T.reshape(-1,2)
velocity_grid = {genotype: np.zeros_like(grid) for genotype in ['wildtype', 'mutant']}
velocity_grid['wildtype'][:] = np.nan
velocity_grid['mutant'][:] = np.nan

#%%
for i, genotype in enumerate(['wildtype', 'mutant']):
    d = adata[adata.obs['genotype']==genotype,:]
    proj = np.array(pca.transform(d.X.toarray()))[:,0:2]
    X = d.X.toarray()
    T = d.obsm['transition_matrix']

    V = util.velocity_vectors(T, X)
    V_emb = util.embed_velocity(X, V, lambda x: np.array(pca.transform(x)[:,0:2]))

    # Find points inside each grid cell
    for i,(x,y) in enumerate(grid):
        idx = np.where((proj[:,0] > x) & (proj[:,0] < x+x_spacing) & 
                       (proj[:,1] > y) & (proj[:,1] < y+y_spacing))[0]
        if len(idx) > 0:
            # Get the average velocity vector
            v = V_emb[idx,:].mean(axis=0).reshape(-1)
            velocity_grid[genotype][i] = v

#%%
# Compute the cosine similarity between the velocity vectors
cosine_sim = np.zeros(len(grid))
cosine_sim[:] = np.nan
for i in range(len(grid)):
    wt_v = velocity_grid['wildtype'][i,:]
    mut_v = velocity_grid['mutant'][i,:]
    if np.isnan(wt_v).any() or np.isnan(mut_v).any():
        continue
    cos_sim = np.dot(wt_v, mut_v) / (np.linalg.norm(wt_v) * np.linalg.norm(mut_v))
    cosine_sim[i] = cos_sim

# Rescale cosine similarities from (-1,1) -> (0,1)
cosine_sim = (cosine_sim + 1) / 2

#%%
fig, axs = plt.subplots(1,2, figsize=(10,5))

for i, genotype in enumerate(['wildtype', 'mutant']):
    ax = axs[i]
    d = adata[adata.obs['genotype']==genotype,:]
    proj = np.array(pca.transform(d.X.toarray()))[:,0:2]
    cell_labels = [cell_types[c] for c in d.obs['cell_type']]
    # Make the scatter plot have the same range for both genotypes
    # ax.scatter(proj[:,0], proj[:,1], s=1, alpha=0.2, c=cell_labels, cmap='tab20')
    ax.set_xlim(minX-xbuf, maxX+xbuf)
    ax.set_ylim(minY-ybuf, maxY+ybuf)
    ax.set_facecolor('grey')
    # add a grid to the plot
    for x in x_grid_points:
        ax.axvline(x, color='black', alpha=0.1)
    for y in y_grid_points:
        ax.axhline(y, color='black', alpha=0.1)
    # Plot the velocity vectors
    for i in range(len(grid)):
        v = velocity_grid[genotype][i,:]
        if np.isnan(v).any():
            continue
        x,y = grid[i,:]
        # Color the arrows by the cosine similarity

        ax.arrow(x + x_spacing/2, y + y_spacing/2, v[0], v[1], 
                 width=0.025, head_width=0.1, head_length=0.05, 
                 color=plt.cm.viridis(cosine_sim[i]))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{genotype.capitalize()}', fontsize=14)
plt.suptitle('Velocity vectors (colored by cosine similarity)', fontsize=18)
plt.tight_layout()

#%%
plt.hist(cosine_sim, bins=20);

# %%
# Plot the data colored by the cosine similarity of the velocity vector grid
from matplotlib.patches import Rectangle
fig, axs = plt.subplots(1,2, figsize=(10,5))

for i, genotype in enumerate(['wildtype', 'mutant']):
    ax = axs[i]
    d = adata[adata.obs['genotype']==genotype,:]
    proj = np.array(pca.transform(d.X.toarray()))[:,0:2]
    cell_labels = [cell_types[c] for c in d.obs['cell_type']]
    colors = plt.cm.tab20(cell_labels)
    # Assign an alpha corresponding to the cosine similarity of the cells in each grid
    ax.scatter(proj[:,0], proj[:,1], s=.6, alpha=1, c=colors)
    alpha = np.zeros(len(proj))
    for i in range(len(grid)):
        x,y = grid[i,:]
        idx = np.where((proj[:,0] > x) & (proj[:,0] < x+x_spacing) & 
                       (proj[:,1] > y) & (proj[:,1] < y+y_spacing))[0]
        if len(idx) > 0:
            sim = cosine_sim[i]
            # Draw a box around the grid on the plot if the cosine similarity is less than .9
            if sim < np.percentile(cosine_sim[~np.isnan(cosine_sim)], 10):
                rect = Rectangle((x,y), x_spacing, y_spacing, linewidth=1, 
                                 edgecolor='black', facecolor='none', alpha=.5)
                ax.add_patch(rect)

    # Make the scatter plot have the same range for both genotypes
    ax.set_xlim(minX-xbuf, maxX+xbuf)
    ax.set_ylim(minY-ybuf, maxY+ybuf)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{genotype.capitalize()}', fontsize=14)

plt.suptitle('Cells colored by cosine similarity of velocity vector grid', fontsize=18)
# Convert the integer cell type labels to strings and make a legend for the plot
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=cell_type,
                       markerfacecolor=plt.cm.tab20(idx), markersize=10)
                   for cell_type, idx in cell_types.items()]
plt.legend(handles=legend_elements, labels=cell_types, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# %%
# Load the model 