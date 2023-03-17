#%%
import pandas as pd
import numpy as np
import scanpy as sc
import umap

#%%
f = open('../data/raw-counts-mes-wildtype.csv','r')

#%%
# Read the first line of the file
header = f.readline()

# %%
expression = []
names_in_data = []
for line in f:
    # Split line by commas
    sp = line.split(',')
    gene = sp[0]
    exp = [int(x) for x in sp[1:]]
    expression.append(exp)
    names_in_data.append(gene.strip('"'))

#%%
expression = np.ndarray(expression)

#%%
adata = sc.AnnData(expression.T)
adata.var_names = names_in_data
# adata.obsm['X'] = expression

#%%
# use UMAP or PHate to obtain embedding that is used for single-cell level visualization
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=5)
sc.tl.umap(adata)

#%%
# Plot the UMAP embedding
sc.pl.umap(adata, size=10, show=False)

#%%
import pickle
# Import gene network from Tiana et al paper
graph = pickle.load(open('../data/filtered_graph.pickle', 'rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
protein_name_to_ids = pickle.load(open('../data/protein_names.pickle', 'rb'))
indices_of_nodes_in_graph = []
data_ids = {}
id_row = {}
for i,name in enumerate(names_in_data):
    name = name.upper()
    if name in protein_name_to_ids:
        for id in protein_name_to_ids[name]:
            if id in graph.nodes:
                indices_of_nodes_in_graph.append(i)
                data_ids[id] = name
                id_row[id] = i

#%%
# Filter the data to only include the genes in the Nanog regulatory network
network_data = adata[:,indices_of_nodes_in_graph]
network_data.var_names = [names_in_data[i] for i in indices_of_nodes_in_graph]
# Rerun PCA and UMAP on the filtered data
sc.tl.pca(network_data, svd_solver='arpack')
sc.pp.neighbors(network_data, n_neighbors=15, n_pcs=50)
sc.tl.umap(network_data)

#%%
# Plot the UMAP embedding of the filtered data
sc.pl.umap(network_data, size=10, show=False)

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
            cluster_sums[:,i] += adata.X[:,id_row[gene_id]]

#%%
# Get the GO enrichment associated with each cluster
cluster_enrichment = pickle.load(open('../data/cluster_enrichments_louvain.pickle', 'rb'))

#%%
# Plot the UMAP embedding of the filtered data colored by the total gene expression of each cluster
# Normalize the expression of each cluster
expression_color = cluster_sums[:,0]/(cluster_sums[:,0].max())
sc.pl.umap(network_data, color=expression_color, show=False)

#%%
print('finished embedding')
# list marker genes or genes of interest if known in advance. otherwise marker_genes = []
marker_genes = ['Igll1', 'Myc', 'Slc7a5', 'Ldha', 'Foxo1', 'Lig4', 'Sp7']  # irf4 down-up
# call VIA. We identify an early (suitable) start cell root = [42]. Can also set an arbitrary value
via.via_wrapper(adata, true_label, embedding, knn=knn, ncomps=ncomps, jac_std_global=0.15, root=[42], dataset='',
            random_seed=1,v0_toobig=0.3, v1_toobig=0.1, marker_genes=marker_genes, piegraph_edgeweight_scalingfactor=1, piegraph_arrow_head_width=.1)
