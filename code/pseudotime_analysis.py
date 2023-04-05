#%%
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import sklearn
import umap
from util import plot_qc_distributions
import pyVIA.core as via

#%%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%%
genotype = 'mutant'
adata = sc.read_h5ad(f'../data/{genotype}.h5ad')

#%%
sc.pl.highest_expr_genes(adata, n_top=20)

#%% Filter out genes that are not expressed in at least 3 cells
sc.pp.filter_genes(adata, min_cells=3)

#%% Filter out cells that have less than 200 genes expressed
sc.pp.filter_cells(adata, min_genes=200)

#%% Plot the percentage of mitochondrial genes expressed per cell
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

#%% 
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

#%%
plot_qc_distributions(adata, genotype, 'all_genes', '../figures')

#%%
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
# The regress_out and scale operations gave me results that were difficult to interpret
# (i.e. negative values for the gene expression). They're not universally 
# recommended for pseudotime analysis, so I'm skipping them for now
# sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
# sc.pp.scale(adata, max_value=10)

#%% Save the processed data
adata.write(f'../data/{genotype}_processed.h5ad')

#%%
import pickle
# Import gene network from Tiana et al paper
graph = pickle.load(open('../data/filtered_graph.pickle', 'rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
protein_name_to_ids = pickle.load(open('../data/protein_names.pickle', 'rb'))
indices_of_nodes_in_graph = []
data_ids = {}
id_row = {}
for i,name in enumerate(adata.var_names):
    name = name.upper()
    if name in protein_name_to_ids:
        for id in protein_name_to_ids[name]:
            if id in graph.nodes:
                indices_of_nodes_in_graph.append(i)
                if id in data_ids:
                    print('Duplicate id', id, name, data_ids[id])
                data_ids[id] = name
                id_row[id] = i

#%%
# Filter the data to only include the genes in the Nanog regulatory network
network_data = adata[:,indices_of_nodes_in_graph]
network_data.var_names = [adata.var_names[i] for i in indices_of_nodes_in_graph]

#%%
# Rerun PCA and UMAP on the filtered data
pca = sklearn.decomposition.PCA(n_components=30)
network_data.obsm['X_pca'] = pca.fit_transform(network_data.X)
# Run UMAP on the PCA embedding
umap_ = umap.UMAP(n_components=2, random_state=42)
umap_embedding = umap_.fit_transform(network_data.obsm['X_pca'])
network_data.obsm['X_umap'] = umap_embedding
 #%%
# Plot the UMAP embedding of the filtered data
sc.pl.umap(network_data, size=10, show=False)

#%%
# Save the network data
pickle.dump(network_data, open(f'../data/network_data_{genotype}.pickle', 'wb'))
# Save the umap object
pickle.dump(umap_, open(f'../data/umap_{genotype}.pickle', 'wb'))
# Save the embedding
pickle.dump(umap_embedding, open(f'../data/umap_embedding_{genotype}.pickle', 'wb'))
# Save the pca object
pickle.dump(pca, open(f'../data/pca_{genotype}.pickle', 'wb'))

#%%
plot_qc_distributions(network_data, genotype, 'network_genes', '../figures')

#%%
# Import the cluster assigments of each gene from the Tiana et al paper
cluster_assignments = pickle.load(open('../data/louvain_clusters.pickle', 'rb'))
# Convert the cluster assignments to indices of the genes in the filtered data
cluster_indexes = []
# Calculate the sum of the expression of each gene in each cluster
cluster_sums = np.zeros((network_data.n_obs, len(cluster_assignments))) 

for i,gene_ids in enumerate(cluster_assignments):
    # Get all the rows in the data that correspond to the current cluster
    for gene_id in gene_ids:
        if gene_id in id_row:
            cluster_sums[:,i] += adata.X[:, id_row[gene_id]]
#%%
# Assign initial points to be cells with high expression of cluster 0
# Get the index of 99th percentile of cluster 0 expression
top_percentile = np.percentile(cluster_sums[:,0], 99)
# Get the cells with expression above the 99th percentile
initial_points = np.where(cluster_sums[:,0] > top_percentile)[0]

#%%
pseudotime = via.VIA(network_data.X, 
                     knn=30,
                     cluster_graph_pruning_std=0.5,
                     jac_std_global=.15,
                     too_big_factor=.2,
                     root_user=initial_points,
                     random_seed=42)
pseudotime.run_VIA()
# Dump the VIA object to a pickle file
pickle.dump(pseudotime, open(f'../data/{genotype}_pseudotime.pickle', 'wb'))

# %%
plt.figure(figsize=(10,10))
plt.scatter(network_data.obsm['X_umap'][:,0], network_data.obsm['X_umap'][:,1], 
            c = pseudotime.single_cell_pt_markov,
            cmap = 'plasma',
            s=1, alpha=.5)
plt.title(f'{genotype.capitalize()} pseudotime')
# Remove x and y axis labels and ticks
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.xticks([])
plt.yticks([])
# Add a colorbar
plt.colorbar()

plt.savefig(f'../figures/pseudotime/{genotype}_pseudotime.png', dpi=300)


# %%
via.draw_trajectory_gams(via_object=pseudotime, embedding=network_data.obsm['X_umap'], 
                         draw_all_curves=False)
plt.savefig(f'../figures/pseudotime/{genotype}_pseudotime_gams.png', dpi=300)

# %%
pseudotime.embedding = network_data.obsm['X_umap']
# edge plots can be made with different edge resolutions. Run hammerbundle_milestone_dict() to recompute the edges for plotting. Then provide the new hammerbundle as a parameter to plot_edge_bundle()
hammerbundle_milestone_dict = via.make_edgebundle_milestone(via_object=pseudotime,
                                                             n_milestones=100,
                                                             global_visual_pruning=0.5,
                                                             decay=0.7, initial_bandwidth=0.05)

fig, ax=via.plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='externally computed edgebundle', headwidth_bundle=0.4)
plt.savefig(f'../figures/pseudotime/{genotype}_pseudotime_edgebundle.png', dpi=300)
# fig.set_size_inches(15,15)
# %%
