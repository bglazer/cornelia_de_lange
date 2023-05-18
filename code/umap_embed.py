#%%
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from util import umap_axes
from umap import UMAP
from sklearn.decomposition import PCA
from random import seed

#%%
# Set the random seed for reproducibility
np.random.seed(42)
seed(42)

#%%
dataset = 'net'
wt = sc.read_h5ad(f'../data/wildtype_{dataset}.h5ad')
mut = sc.read_h5ad(f'../data/mutant_{dataset}.h5ad')
adata = wt.concatenate(mut, batch_key='genotype', batch_categories=['wildtype', 'mutant'])

#%%
# UMAP embedding of the combined data
# Use the label information from the batches 
umap  = UMAP(n_components=2, n_neighbors=30, random_state=42)
umap_embedding = umap.fit_transform(adata.X)

#%% 
# Plot the UMAP embedding of the combined data
plt.scatter(umap_embedding[:,0], umap_embedding[:,1], 
            s=1, c=adata.obs['genotype'].cat.codes)
umap_axes(plt.gca())
plt.title('UMAP embedding of wildtype and mutant data');
# %%
# Add the umap embedding to the combined dataset
adata.obsm['X_umap'] = umap_embedding

# Add the umap embedding to the WT and mutant datasets
wt.obsm['X_umap'] = umap_embedding[adata.obs['genotype']=='wildtype',:]
mut.obsm['X_umap'] = umap_embedding[adata.obs['genotype']=='mutant',:]

#%%
# PCA embed the combined data
pca = PCA(n_components=50, random_state=42)
proj = pca.fit_transform(adata.X.toarray())
wt.obsm['X_pca'] = proj[adata.obs['genotype']=='wildtype',:]
wt.uns['PCs'] = pca.components_
wt.uns['pca_mean'] = pca.mean_
#%%
mut.obsm['X_pca'] = proj[adata.obs['genotype']=='mutant',:]
mut.uns['PCs'] = pca.components_
mut.uns['pca_mean'] = pca.mean_

# %%
# Save the combined dataset
# adata.write_h5ad(f'../data/combined_{dataset}.h5ad')
# Save the WT and network datasets
wt.write_h5ad(f'../data/wildtype_{dataset}.h5ad')
mut.write_h5ad(f'../data/mutant_{dataset}.h5ad')


# %%
