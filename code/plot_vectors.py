#%%
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
from util import tonp, plot_arrows, velocity_vectors, embed_velocity, get_plot_limits, is_notebook
from sklearn.decomposition import PCA

#%%
dataset = 'net'
#%% 
wt = sc.read_h5ad(f'../data/wildtype_{dataset}.h5ad')
mut = sc.read_h5ad(f'../data/mutant_{dataset}.h5ad')

#%%
pcs = wt.varm['PCs']
pca = PCA()
pca.components_ = pcs.T
pca.mean_ = wt.X.mean(axis=0)

#%%
def embed(X, pcs=[0,1]):
    return pca.transform(X)[:,pcs]

def plot_velocity(adata):
    # Get the transition matrix from the VIA graph
    X = adata.X.toarray()
    T = adata.obsm['transition_matrix']

    V = velocity_vectors(T, X)
    embedding = embed(X)
    V_emb = embed_velocity(X, V, embed)

    x_limits, y_limits = get_plot_limits(embedding)
    
    plot_arrows(idxs=range(len(embedding)), 
                points=np.asarray(embedding), 
                V=V_emb, 
                sample_every=50, 
                c=adata.obs['pseudotime'],
                xlimits=x_limits,
                ylimits=y_limits,
                aw=0.015,
                )
    
plot_velocity(wt)
plot_velocity(mut)
# %%
import pyVIA.core as via
# %%
def streamplot(adata):
    X = adata.X.toarray()
    T = adata.obsm['transition_matrix']

    V = velocity_vectors(T, X)
    embedding = embed(X)
    via.via_streamplot(v0, embedding, 
                       density_grid=0.8, scatter_size=30, 
                       scatter_alpha=0.3, linewidth=0.5)