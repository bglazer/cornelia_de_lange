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
# Perform PCA
# sc.pp.pca(adata, random_state=42)
# Compute the neighborhood graph
scv.pp.neighbors(adata, random_state=0)

#%%
pca_var = adata.uns['pca']['variance_ratio']
sc.pl.pca_variance_ratio(adata)
var_explained = .90
tot_exp_var = 0
n = 0
for _var in pca_var:
    n +=1
    tot_exp_var += _var
    if tot_exp_var > var_explained: 
        print(n+1, "PCs explain at least", var_explained*100, "percent of the variance")
        break
#%%
def cumulative(var):
    cum_var = []
    tot_sum = 0
    for i in var:
        tot_sum += i
        cum_var.append(tot_sum)
    return cum_var

cum_var = cumulative(pca_var)
#%%
def find_knee_varexpl(cum_var):
    farthestk = 0
    n = len(cum_var)
    for i in range(n):
        #find distance of point p3 from line between p1 and p2
        p1=np.array([0,cum_var[0]])
        p2=np.array([(n-1),cum_var[n-1]])
        p3=np.array([i,cum_var[i]])
        k = (np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

        #knee is farthest away from line between p1 and p2
        if k > farthestk: 
            farthestk = k
            knee = i
    return knee +1 # number of components is 1 more than index
pc_knee = find_knee_varexpl(cum_var)
print("Knee of EV vs PC plot: ",find_knee_varexpl(cum_var))
#%%
# Find the knee in explained variance vs number of archetypes
def find_knee(ev_per_arc, kmax):
    farthestk = 0
    for i in range(3,kmax):
        #find distance of point p3 from line between p1 and p2
        p1=np.array([3,ev_per_arc[0]])
        p2=np.array([kmax,ev_per_arc[kmax-3]])
        p3=np.array([i,ev_per_arc[i-3]])
        # k is the distance from the line between p1 and p2
        k = (np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

        #knee is farthest away from line between p1 and p2
        if k > farthestk: 
            farthestk = k
            knee = i
    if knee is None:
        raise Exception("No knee found")
    print("K* =",knee, "archetypes, distance between k* and line: ",np.round(farthestk,2))
    return knee, farthestk

# Plot the explained variance vs number of archetypes
max_archetypes = 12
ev_per_arc = []
for i in range(3, max_archetypes):
    # pc_knee is the the number of principal components found above as the knee of the EV plot for PCA
    XC, S, C, SSE, varexpl = PCHA(adata.obsm['X_pca'][:,0:(pc_knee)].T, noc=i, delta=0.1)
    ev_per_arc.append(varexpl)
    print(f"Num archtypes: {i}, variance explained: {varexpl:.4f}")

plt.scatter(x = range(3,max_archetypes), y = ev_per_arc)
plt.title(f"EV per $k^*$ archetypes on {pc_knee} component PCA projection, delta = 0.1")
plt.xlabel("Number of archetypes")
plt.ylabel("EV fraction");

# kmax is the maximum number of archetypes to consider
print("Knee in EV vs k plot for different k_max:")
for kmax in range(8,max_archetypes):
    print('k_max =', kmax)
    knee, farthestk = find_knee(ev_per_arc, kmax=kmax)
plt.axvline(x=knee, linestyle = "--")
plt.show()

#%%
XC, S, C, SSE, varexpl = PCHA(adata.obsm['X_pca'][:,0:(pc_knee)].T, noc=knee, delta=0)
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
XC_df.columns = [f'Arc_{i}' for i in range(1,knee+1)]

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

S_df.columns = [f'Arc_{i}' for i in range(1,knee+1)]

adata.obsm['py_pcha_S'] = S_df

for c in adata.obsm['py_pcha_S']:
    adata.obs[f"{c}_PCHA_Score"] = adata.obsm['py_pcha_S'][c]

adata.uns['archetype_vertices'] = XC
#%%
# # Write out data
adata.write_h5ad(f'../data/{genotype}_{dataset}.h5ad')

# %%
