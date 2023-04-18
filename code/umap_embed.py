#%%
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from util import umap_axes
from umap import UMAP
import pickle

#%%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%%
wt = sc.read_h5ad(f'../data/wildtype_net.h5ad')
mut = sc.read_h5ad(f'../data/mutant_net.h5ad')
adata = wt.concatenate(mut, batch_key='genotype', batch_categories=['wildtype', 'mutant'])

#%%
# Check that the id_row is the same for both datasets
for key, nm in mut.uns['id_row'].items():
    if wt.uns['id_row'][key]!=nm:
        raise Exception('id_row is not the same for both datasets')

#%%
# Add the id_row to the combined dataset
adata.uns['id_row'] = wt.uns['id_row']

#%%
# Import gene network from Tiana et al paper
graph = pickle.load(open('../data/filtered_graph.pickle', 'rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
protein_name_to_ids = pickle.load(open('../data/protein_names.pickle', 'rb'))
indices_of_nodes_in_graph = []

for proteinid, datarow in adata.uns['id_row'].items():
    if proteinid in graph.nodes:
        indices_of_nodes_in_graph.append(datarow)

#%%
# Filter the data to only include the genes in the Nanog regulatory network
network_data = adata[:,indices_of_nodes_in_graph]
network_data.var_names = [adata.var_names[i] for i in indices_of_nodes_in_graph]

#%%
# UMAP embedding of the combined data
# Use the label information from the batches 
umap  = UMAP(n_components=2, n_neighbors=30, random_state=42)
umap_embedding = umap.fit_transform(network_data.X)

#%% 
# Plot the UMAP embedding of the combined data
plt.scatter(umap_embedding[:,0], umap_embedding[:,1], 
            s=1, c=network_data.obs['genotype'].cat.codes)
umap_axes(plt.gca())
plt.title('UMAP embedding of wildtype and mutant data');
# %%
# Add the umap embedding to the combined dataset
adata.obsm['X_umap'] = umap_embedding

# Add the umap embedding to the WT and mutant datasets
wt.obsm['X_umap'] = umap_embedding[adata.obs['genotype']=='wildtype',:]
mut.obsm['X_umap'] = umap_embedding[adata.obs['genotype']=='mutant',:]

# %%
# Save the combined dataset
adata.write_h5ad('../data/combined_net.h5ad')
# Save the WT and network datasets
wt.write_h5ad('../data/wildtype_net.h5ad')
mut.write_h5ad('../data/mutant_net.h5ad')

# %%
