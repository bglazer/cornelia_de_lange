#!/usr/bin/env python
# coding: utf-8

# # Unsupervised and supervised archetype analysis on combined dataset (raw and scanorama)

# import mazebox as mb
import scvelo as scv
import scanpy as sc
import os.path as op
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import dropkick as dk
import mazebox as mb
import cellrank as cr
import scanorama
from scipy.spatial.distance import squareform
from scipy.sparse import issparse, csr_matrix, find
from scipy.spatial.distance import pdist
from py_pcha import PCHA
from sklearn.utils import shuffle
from matplotlib.colors import CenteredNorm

adata = sc.read_h5ad('../../data/combined/adata_01_filtered.h5ad')

adata

# ## MAGIC

# For stronger PCHA, we'll run MAGIC.

import magic
# import pandas as pd
# import matplotlib.pyplot as plt
magic_operator = magic.MAGIC(solver='approximate')
X_magic = magic_operator.fit_transform(adata)

sc.pp.pca(X_magic)

sc.pl.pca(X_magic, color=['sample_x','identity'])

scv.pp.neighbors(X_magic, random_state=0)
scv.tl.umap(X_magic, random_state=0)

scv.pl.umap(X_magic,color=['sample_x','identity'])

sc.pl.pca_variance_ratio(X_magic)

pca_var = X_magic.uns['pca']['variance_ratio']
var_explained = .90
tot_exp_var = 0
n = 0
for i in pca_var:
    n +=1
    tot_exp_var += i
    if tot_exp_var > var_explained: 
        print(n+1, "PCs explain at least", var_explained*100, "percent of the variance")
        break

def cumulative(var):
    cum_var = []
    tot_sum = 0
    for i in var:
        tot_sum += i
        cum_var.append(tot_sum)
    return cum_var

cum_var = cumulative(pca_var)

def find_knee_varexpl(cum_var):
    farthestk = 0
    for i in range(50):
        #find distance of point p3 from line between p1 and p2
        p1=np.array([0,cum_var[0]])
        p2=np.array([49,cum_var[49]])
        p3=np.array([i,cum_var[i]])
        k = (np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

        #knee is farthest away from line between p1 and p2
        if k > farthestk: 
            farthestk = k
            knee = i
    return knee +1 # number of components is 1 more than index
print("Knee of EV vs PC plot: ",find_knee_varexpl(cum_var))

X_magic.write_h5ad('../../data/combined/X_magic.h5ad')

# ## PCHA prep

def find_knee(ev_per_arc, kmax):
    farthestk = 0
    for i in range(3,kmax):
        #find distance of point p3 from line between p1 and p2
        p1=np.array([3,ev_per_arc[0]])
        p2=np.array([kmax,ev_per_arc[kmax-3]])
        p3=np.array([i,ev_per_arc[i-3]])
        k = (np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

        #knee is farthest away from line between p1 and p2
        if k > farthestk: 
            farthestk = k
            knee = i
    print("K* =",knee, "archetypes, distance between k* and line: ",np.round(farthestk,2))
    return knee, farthestk

# # PCHA

# pca_var = X_magic_no_doublets.uns['pca']['variance_ratio']
# cum_var = cumulative(pca_var)

# print("Knee of EV vs PC plot: ",find_knee_varexpl(cum_var))

ev_per_arc = []
for i in range(3,11):
    # use the number of components found above as the knee of the EV plot for PCA
    XC, S, C, SSE, varexpl = PCHA(X_magic.obsm['X_pca'][:,0:7].T, noc=i, delta=0.1)
    ev_per_arc.append(varexpl)
    print(varexpl)
plt.scatter(x = [3,4,5,6,7,8,9,10], y = ev_per_arc)
plt.title(f"EV per $k^*$ archetypes on 10 component PCA projection, delta = 0.1")
plt.xlabel("Number of archetypes")
plt.ylabel("EV fraction")
print("Knee in EV vs k plot for different k_max:")
for kmax in range(8,11):
    print('k_max =', kmax)
    knee, farthestk = find_knee(ev_per_arc, kmax=kmax)
plt.axvline(x=knee, linestyle = "--")
plt.show()

#change the number of components here to match the above code block
XC, S, C, SSE, varexpl = PCHA(X_magic.obsm['X_pca'][:,0:7].T, noc=6, delta=0)
XC = np.array(XC)
for components in ['1,2','1,3','2,3']:
    scv.pl.pca(X_magic, color = 'identity', components=components, show=False, figsize= (6,6), frameon=True)
    comp = components.split(',')
    plt.scatter(XC[int(comp[0])-1], XC[int(comp[1])-1], color = 'green', s = 200)
    plt.xlabel("PC"+comp[0])
    plt.ylabel("PC"+comp[1])

    plt.show()

XC_df = pd.DataFrame(XC)
XC_df.columns = ['Arc_1','Arc_2','Arc_3','Arc_4','Arc_5', "Arc_6"]

X_magic_pca_df = pd.DataFrame(X_magic.obsm['X_pca'][:,0:7], index = X_magic.obs_names)
X_magic_full_df = X_magic_pca_df.append(XC_df.T)
X_magic_full_df.head()

pdx = squareform(pdist(X_magic_full_df, metric='euclidean')) # compute distances on pca
pdx_df = pd.DataFrame(pdx, index=X_magic_full_df.index, columns=X_magic_full_df.index)
pdx_df = pdx_df.loc[XC_df.columns].drop(XC_df.columns, axis = 1)
X_magic.obsm['arc_distance'] = pdx_df.T

# Now that we have a euclidean distance to each archetype on the MAGIC imputed data, we can find a neighborhood with arbritrary radius to classify cells closest to each archetype as specialists. In Van Dijk et al., they choose a radius (distance on diffusion map) that is 1/2 the minimum of the distance between archetypes.

pdx_archetypes = squareform(pdist(XC_df.T, metric='euclidean')) # compute distances on pca

radius = .5*pdx_archetypes[pdx_archetypes > 0].min()

tmp = X_magic.obsm['arc_distance'].copy()

# percent_radius = .1 # in percentage of datapoints; radius = .1 means 10% closest cells to each archetype

for arc in  X_magic.obsm['arc_distance'].columns:
    closest = X_magic.obsm['arc_distance'].loc[X_magic.obsm['arc_distance'][arc].sort_values() < radius]
    tmp.loc[closest.index,'specialist'] = arc

X_magic.obs['specialists_pca_diffdist'] = tmp.specialist

# ## Labeling by PCHA

scv.pl.umap(X_magic, c = 'identity', components='1,2', show=False, figsize= (5,5), frameon=True, cmap = 'RdBu')

for i in range(6):
    scv.pl.umap(X_magic, c = [S[i,:].T], components='1,2', show=False, figsize= (5,5), frameon=True, cmap = 'RdBu')
    plt.title(f"Archetype {i}")
    # plt.savefig(f'./figures/unsupervised_AA_magic/{i}_scPCHA_pca.pdf')
    

S_df = pd.DataFrame(S.T)

S_df.index = X_magic.obs_names

X_magic.obsm['py_pcha_S'] = S.T

adata.obsm['arc_distance'] = pdx_df.T

S_df.columns = ['Arc_1','Arc_2','Arc_3','Arc_4','Arc_5', 'Arc_6']

adata.obsm['py_pcha_S'] = S_df

for c in adata.obsm['py_pcha_S']:
    adata.obs[f"{c}_PCHA_Score"] = adata.obsm['py_pcha_S'][c]

for i in range(6):
    scv.pl.scatter(adata, c = S[i,:].T, basis = "umap_wnn",components='1,2', show=False, figsize= (5,5), frameon=True, cmap = 'RdBu_r', smooth = True)
    plt.title(f"Archetype {i}")
    plt.show()

# # Write out data
adata.write_h5ad('../../data/combined/adata_02_filtered.h5ad')

X_magic.write_h5ad("../../data/combined/X_magic_02_filtered.h5ad")

