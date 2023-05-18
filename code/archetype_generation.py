# Archetype analysis
#%%
import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from py_pcha import PCHA
# import magic

#%%
genotype = 'wildtype'
dataset = 'net'
adata = sc.read_h5ad(f'../data/{genotype}_{dataset}.h5ad')

#%%
pca_var = adata.uns['pca']['variance_ratio']

#%%
# Plot the explained variance vs number of archetypes
max_dim = 10
from joblib import Parallel, delayed

parallel = Parallel(n_jobs=30, verbose=11)
def pcha(i,j):
    XC, S, C, SSE, varexpl = PCHA(adata.obsm['X_pca'][:,:i].T, noc=j, delta=0.1)
    return (i,j,varexpl)
    # print(f"PCA dim {i} Num archtypes: {j}, PCA variance explained: {varexpl:.4f} archetype variance explained: {SSE:.4f}")

jobs = []
for i in range(3, max_dim):
    max_archetypes = i + 10
    for j in range(i+1, max_archetypes):
        jobs.append(delayed(pcha)(i,j))

ev_per_arc = parallel(jobs)

#%%
def find_knee(var):
    farthestk = 0
    n = len(var)
    for i in range(n):
        #find distance of point p3 from line between p1 and p2
        p1=np.array([0,var[0]])
        p2=np.array([n-1,var[n-1]])
        p3=np.array([i,var[i]])
        k = (np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

        #knee is farthest away from line between p1 and p2
        # print(i, k)
        if k > farthestk: 
            farthestk = k
            knee = i
    return knee, farthestk # number of components is 1 more than index
#%%
# Plot explained variance per archetype as a heatmap
ev_per_arc_arr = np.array(ev_per_arc)
# Total variance explained by the archetypes is the product of variance
# explained by each archetype and the variance explained by the PCs
cum_pca_var = np.cumsum(pca_var)
total_var_expl = ev_per_arc_arr[:,2].reshape(7,-1) * cum_pca_var[3:10].reshape(-1,1)
# Create an array of 3-10 repeated 7 times along the row
#%%
knees = []
for i in range(7):
    knee, knee_dist = find_knee(total_var_expl[i])
    knees.append((knee_dist, knee))
    # print(i+3, knee, knee_dist)
knees = np.array(knees)
# Add three because we started at 3 PCs
farthest_knee_idx = np.argmax(knees[:,0])
farthest_knee = 3+farthest_knee_idx
farthest_knee_archetype = int(knees[farthest_knee_idx,1])
pc_knee = farthest_knee
archetype_knee = farthest_knee + farthest_knee_archetype + 1
print(pc_knee, archetype_knee)

# plt.plot(np.arange(7).repeat(9).reshape(7,9), total_var_expl, marker='o')
idx = farthest_knee_archetype
plt.plot(total_var_expl[:,idx], marker='o')
plt.plot([0,6],[total_var_expl[0,idx], total_var_expl[-1,idx]])
plt.xlabel("Number of PCs")
plt.ylabel("Total variance explained")
plt.xticks(np.arange(7), np.arange(3,10));
plt.title("Total variance explained for different numbers of archetypes and PC dimensions")
plt.axvline(farthest_knee_idx)
plt.axhline(total_var_expl[farthest_knee_idx, int(farthest_knee_archetype)])

#%%
XC, S, C, SSE, varexpl = PCHA(adata.obsm['X_pca'][:,0:(pc_knee)].T, 
                              noc=archetype_knee, delta=0)
XC = np.array(XC)
for components in ['1,2','1,3','2,3']:
    scv.pl.pca(adata, color = 'cell_type', components=components, show=False, figsize= (6,6), frameon=True)
    comp = components.split(',')
    plt.scatter(XC[int(comp[0])-1], XC[int(comp[1])-1], color = 'green', s = 200)
    plt.xlabel("PC"+comp[0])
    plt.ylabel("PC"+comp[1])
    plt.show()

#%%
# XC is the matrix of archetypes, shape (num_pcs, num_archetypes)
XC_df = pd.DataFrame(XC)
XC_df.columns = [f'Arc_{i}' for i in range(1,archetype_knee+1)]

# Make a dataframe with the PCA (n_components=knee) projection of the data
X_pca_df = pd.DataFrame(adata.obsm['X_pca'][:,0:pc_knee], index = adata.obs_names)
# Append the archetype locations 
X_full_df = X_pca_df.append(XC_df.T)

# Compute the distance matrix between all cells and archetypes
pdx = squareform(pdist(X_full_df, metric='euclidean')) 
pdx_df = pd.DataFrame(pdx, index=X_full_df.index, columns=X_full_df.index)
# Keep only the distances between cells and archetypes
pdx_df = pdx_df.loc[XC_df.columns].drop(XC_df.columns, axis = 1)
adata.obsm['arc_distance'] = pdx_df.T

#%%
# Now that we have a euclidean distance to each archetype, \
# we can find a neighborhood with arbritrary radius to classify cells closest to each archetype as 
# specialists. In Van Dijk et al., they choose a radius (distance on diffusion map) that 
# is 1/2 the minimum of the distance between archetypes.

# Compute the distance matrix between all archetypes
pdx_archetypes = squareform(pdist(XC_df.T, metric='euclidean')) 

# Find half the minimum distance between archetypes
radius = .5*pdx_archetypes[pdx_archetypes > 0].min()

tmp = adata.obsm['arc_distance'].copy()

# percent_radius = .1 # in percentage of datapoints; radius = .1 means 10% closest cells to each archetype

for arc in  adata.obsm['arc_distance'].columns:
    # Find the set of points within radius of each archetype
    closest = adata.obsm['arc_distance'].loc[adata.obsm['arc_distance'][arc].sort_values() < radius]
    # Label the closest points as specialists
    tmp.loc[closest.index,'specialist'] = arc

adata.obs['specialists_pca_diffdist'] = tmp.specialist
#%%
# ## Labeling by PCHA
#%%
S_df = pd.DataFrame(S.T)

S_df.index = adata.obs_names

adata.obsm['py_pcha_S'] = S.T

adata.obsm['arc_distance'] = pdx_df.T

S_df.columns = [f'Arc_{i}' for i in range(1,archetype_knee+1)]

adata.obsm['py_pcha_S'] = S_df

for c in adata.obsm['py_pcha_S']:
    adata.obs[f"{c}_PCHA_Score"] = adata.obsm['py_pcha_S'][c]

adata.uns['archetype_vertices'] = XC
#%%
# # Write out data
adata.write_h5ad(f'../data/{genotype}_{dataset}.h5ad')

# %%
genotype = 'mutant'
dataset = 'net'
adata = sc.read_h5ad(f'../data/{genotype}_{dataset}.h5ad')
# %%
# XC is the matrix of archetypes, shape (num_pcs, num_archetypes)
XC_df = pd.DataFrame(XC)
XC_df.columns = [f'Arc_{i}' for i in range(1,archetype_knee+1)]

# Make a dataframe with the PCA (n_components=knee) projection of the data
X_pca_df = pd.DataFrame(adata.obsm['X_pca'][:,0:pc_knee], index = adata.obs_names)
# Append the archetype locations 
X_full_df = X_pca_df.append(XC_df.T)

# Compute the distance matrix between all cells and archetypes
pdx = squareform(pdist(X_full_df, metric='euclidean')) 
pdx_df = pd.DataFrame(pdx, index=X_full_df.index, columns=X_full_df.index)
# Keep only the distances between cells and archetypes
pdx_df = pdx_df.loc[XC_df.columns].drop(XC_df.columns, axis = 1)
adata.obsm['arc_distance'] = pdx_df.T

#%%
# Now that we have a euclidean distance to each archetype, \
# we can find a neighborhood with arbritrary radius to classify cells closest to each archetype as 
# specialists. In Van Dijk et al., they choose a radius (distance on diffusion map) that 
# is 1/2 the minimum of the distance between archetypes.

# Compute the distance matrix between all archetypes
pdx_archetypes = squareform(pdist(XC_df.T, metric='euclidean')) 

# Find half the minimum distance between archetypes
radius = .5*pdx_archetypes[pdx_archetypes > 0].min()

tmp = adata.obsm['arc_distance'].copy()

# percent_radius = .1 # in percentage of datapoints; radius = .1 means 10% closest cells to each archetype

for arc in  adata.obsm['arc_distance'].columns:
    # Find the set of points within radius of each archetype
    closest = adata.obsm['arc_distance'].loc[adata.obsm['arc_distance'][arc].sort_values() < radius]
    # Label the closest points as specialists
    tmp.loc[closest.index,'specialist'] = arc

adata.obs['specialists_pca_diffdist'] = tmp.specialist
#%%
# ## Labeling by PCHA
#%%
# S_df = pd.DataFrame(S.T)

# S_df.index = adata.obs_names

# adata.obsm['py_pcha_S'] = S.T

adata.obsm['arc_distance'] = pdx_df.T

# S_df.columns = [f'Arc_{i}' for i in range(1,archetype_knee+1)]

# adata.obsm['py_pcha_S'] = S_df

# for c in adata.obsm['py_pcha_S']:
#     adata.obs[f"{c}_PCHA_Score"] = adata.obsm['py_pcha_S'][c]

adata.uns['archetype_vertices'] = XC
#%%
# # Write out data
adata.write_h5ad(f'../data/{genotype}_{dataset}.h5ad')

# %%
