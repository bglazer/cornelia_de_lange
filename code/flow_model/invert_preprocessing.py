#%%
import scanpy as sc
import numpy as np
import pickle
from matplotlib import pyplot as plt

#%%
# Load the data
wt_data = sc.read_h5ad('../../data/wildtype_full.h5ad')
mut_data = sc.read_h5ad('../../data/mutant_full.h5ad')

#%%
# Load utility data
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}

# %%
wt_raw = np.array(wt_data.layers['raw'].todense())
mut_raw = np.array(mut_data.layers['raw'].todense())
# %%
# Compute the normalization factors
wt_sum = np.sum(wt_raw, axis=1, keepdims=True)
mut_sum = np.sum(mut_raw, axis=1, keepdims=True)
# %%
# Normalize and log transform the data
wt_nrm = wt_raw / wt_sum * 1e4
mut_nrm = mut_raw / mut_sum * 1e4
wt_log = np.log1p(wt_nrm)
mut_log = np.log1p(mut_nrm)
# %%
wt_X = np.array(wt_data.X.todense())
mut_X = np.array(mut_data.X.todense())
#%%
plt.hist(wt_log[wt_log>0].flatten(), bins=100, alpha=0.5, label='WT')
plt.hist(mut_log[mut_log>0].flatten(), bins=100, alpha=0.5, label='Mut')
plt.hist(wt_X[wt_X>0].flatten(), bins=100, alpha=0.5, label='WT data')
plt.hist(mut_X[mut_X>0].flatten(), bins=100, alpha=0.5, label='Mut data')
plt.legend()
#%%
# Check whether the manually transformed raw data matches the data in the .h5ad file\
np.allclose(wt_log, wt_X)
# %%
# Try to invert the preprocessing steps using the normalization factor and exponentiation
wt_raw_reconstructed = np.expm1(wt_X) * wt_sum / 1e4
mut_raw_reconstructed = np.expm1(mut_X) * mut_sum / 1e4

# %%
print('wt close:', np.allclose(wt_raw, wt_raw_reconstructed))
print('mut close:', np.allclose(mut_raw, mut_raw_reconstructed))

# %%
# Subset the normalization factor to only the genes in the network
def get_graph_nodes(adata):
    graph = pickle.load(open('../../data/filtered_graph.pickle', 'rb'))
    protein_id_to_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
    protein_name_to_ids = pickle.load(open('../../data/protein_names.pickle', 'rb'))
    indices_of_nodes_in_graph = []
    data_ids = {}
    id_new_row = {}
    new_row = 0
    for i,name in enumerate(adata.var_names):
        name = name.upper()
        if name in protein_name_to_ids:
            for id in protein_name_to_ids[name]:
                if id in graph.nodes:
                    indices_of_nodes_in_graph.append(i)
                    if id in data_ids:
                        print('Duplicate id', id, name, data_ids[id])
                    data_ids[id] = name
                    id_new_row[id] = new_row
                    new_row += 1
    return indices_of_nodes_in_graph
    
wt_graph_idxs = get_graph_nodes(wt_data)
mut_graph_idxs = get_graph_nodes(mut_data)
for i in range(len(wt_graph_idxs)):
    if wt_graph_idxs[i] != mut_graph_idxs[i]:
        print('Inconsistent:', i, wt_graph_idxs[i], mut_graph_idxs[i])
        break
graph_idxs = wt_graph_idxs
# %%
# Load the baseline trajectories
wt_baseline_trajectories = pickle.load(open('../../output/20230607_165324/baseline_trajectories_wildtype.pickle', 'rb'))
mut_baseline_trajectories = pickle.load(open('../../output/20230608_093734/baseline_trajectories_mutant.pickle','rb'))

# %%
# Get the indexes of the NMP cells used in the simulations
wt_net_data = sc.read_h5ad('../../data/wildtype_net.h5ad')
mut_net_data = sc.read_h5ad('../../data/mutant_net.h5ad')
#%%
wt_initial_idxs = wt_net_data.uns['initial_points_nmp']
mut_initial_idxs = mut_net_data.uns['initial_points_nmp']
# %%
# Get the normalization factors for the NMP cells
wt_nmp_sum = wt_sum[wt_initial_idxs]
mut_nmp_sum = mut_sum[mut_initial_idxs]
# %%
# Calculate the Fano factor for the non-denormalized cells, 
# i.e. var/mean for np.expm1(X) without the sum normalization
wt_fano_genes_not_nrm = np.var(np.expm1(wt_X), axis=0) / np.mean(np.expm1(wt_X), axis=0)
wt_fano_genes_raw = np.var(wt_raw, axis=0) / np.mean(wt_raw, axis=0)

print(wt_fano_genes_not_nrm[~np.isnan(wt_fano_genes_not_nrm)].mean())
print(wt_fano_genes_raw[~np.isnan(wt_fano_genes_raw)].mean())
# %%
# Which genes are all zero?
for idx in np.where(np.all(wt_raw == 0, axis=0))[0]:
    print(idx_to_node[graph_idxs[idx]])

# %%
wt_net_X = np.array(wt_net_data.X.todense())
mut_net_X = np.array(mut_net_data.X.todense())
from sklearn.decomposition import PCA
wt_proj = np.array(wt_net_data.obsm['X_pca'])
mut_proj = np.array(mut_net_data.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = wt_net_data.uns['pca_mean']
pca.components_ = wt_net_data.uns['PCs']
wt_proj = np.array(pca.transform(wt_net_X))[:,0:2]
mut_proj = np.array(pca.transform(mut_net_X))[:,0:2]
#%%
# Scatter plot colored by normalizing factor
plt.scatter(wt_proj[:,0], wt_proj[:,1], alpha=1.0, c=wt_sum, label='WT',s=1)
#%%
plt.scatter(mut_proj[:,0], mut_proj[:,1], alpha=1.0, c=mut_sum, label='Mut',s=1)
# %%
