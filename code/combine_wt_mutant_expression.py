#%%
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
from umap import UMAP

#%% 
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%% 
# Load the processed data
wt = sc.read_h5ad('../data/wildtype_processed.h5ad')
mut = sc.read_h5ad('../data/mutant_processed.h5ad')
# Combine the wildtype and mutant data
adata = wt.concatenate(mut, batch_categories=['wildtype', 'mutant'])
adata.obs['batch'] = adata.obs['batch'].astype('category')

#%%
import pickle
def filter_to_network(adata, min_cells=3):    
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
    # Filter the data to only include the genes in the Nanog regulatory network
    network_data = adata[:,indices_of_nodes_in_graph]
    network_data.var_names = [adata.var_names[i] for i in indices_of_nodes_in_graph]
    return network_data, id_row

network_data, id_row = filter_to_network(adata)
wt_net, _ = filter_to_network(wt)
mut_net, _ = filter_to_network(mut)

#%%
# UMAP embedding of the combined data
# Use the label information from the batches 
umap  = UMAP(n_components=2, n_neighbors=30, random_state=42)
umap_embedding = umap.fit_transform(network_data.X)

#%% 
# Plot the UMAP embedding of the combined data
plt.scatter(umap_embedding[:,0], umap_embedding[:,1], 
            s=1, c=network_data.obs['batch'].cat.codes)
# Remove the ticks and labels from the plot
plt.xticks([])
plt.yticks([])
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP embedding of wildtype and mutant data')

# %%
# Find the minimum distance between each wild type cell and mutant cell
# Import the kd-tree data structure
from scipy.spatial import KDTree
kdtree = KDTree(wt_net.X)
results = kdtree.query(mut_net.X)

# %%
dists, idxs = results
# Plot the distribution of distances
plt.hist(dists, bins=100);
# %%
# Plot the scatter plot of the UMAP embedding of the combined data
# colored by the distance between the wildtype and mutant cells
wt_emb = umap_embedding[:wt.shape[0]]
mut_emb = umap_embedding[wt.shape[0]:]

plt.figure(figsize=(10,10))
plt.scatter(mut_emb[:,0], mut_emb[:,1],
            s=.5, c=dists)
# Remove the ticks and labels from the plot
plt.xticks([])
plt.yticks([])
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP embedding of mutant data colored by distance to nearest wildtype cell')
# Remove the gridlines
plt.grid(False)
# Add a colorbar
plt.colorbar()

# %%
# Load the pseudotime data for the wildtype and mutant data
wt_via = pickle.load(open('../data/wildtype_pseudotime.pickle', 'rb'))
mut_via = pickle.load(open('../data/mutant_pseudotime.pickle', 'rb'))

#%%
# Plot the mutant and wildtype cells side by side 
# colored by pseudotime
wt_pseudotime = wt_via.single_cell_pt_markov
mut_pseudotime = mut_via.single_cell_pt_markov
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].scatter(wt_emb[:,0], wt_emb[:,1],
            s=.5, c=wt_pseudotime)
axs[0].set_title('Wildtype')
axs[1].scatter(mut_emb[:,0], mut_emb[:,1],
            s=.5, c=mut_pseudotime)
axs[1].set_title('Mutant')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.grid(False)

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
# Get the GO enrichment associated with each cluster
cluster_enrichment = pickle.load(open('../data/cluster_enrichments_louvain.pickle', 'rb'))
# Convert to a dictionary of clusters and terms with only the p<.01 terms
cluster_terms = {}
for cluster,term,pval,genes in cluster_enrichment:
    if pval < .01:
        if cluster not in cluster_terms: cluster_terms[cluster] = []
        cluster_terms[cluster].append((term,pval,genes))

#%%
# Plot the UMAP embedding of the filtered data colored by the total gene expression of each cluster
# Normalize the expression of each cluster
expression_sum_nrm = cluster_sums/(cluster_sums.max(axis=0))

# Filter words that are relevant to development
positive_words = ['develop', 'signal', 'matrix', 
                'organization', 'proliferation', 'stem', 'pathway', 'epithel', 'mesenchym',
                'morpho', 'mesoderm', 'endoderm', 'different', 'specification']

# Plot the cluster sums for wildtype and mutant side by side
for i in range(cluster_sums.shape[1]):
    fig,axs = plt.subplots(1,2, figsize=(10,10))
    print(i, flush=True)
    if i in cluster_terms:
        for genotype in ['wildtype', 'mutant']:
            if genotype == 'wildtype':
                ax = axs[0]
                X = wt_emb
                idxs = np.arange(wt.shape[0])
            else:
                ax = axs[1]
                X = mut_emb
                idxs = np.arange(wt.shape[0], wt.shape[0]+mut.shape[0])
            # Plot the UMAP embedding of the filtered data colored by
            #  the total gene expression of each cluster
            ax.scatter(X[:,0], X[:,1], 
                       c=expression_sum_nrm[idxs,i], 
                       s=10, alpha=.5, cmap='viridis', vmin=0, vmax=1)
            # Title the plot
            ax.set_title(f'Cluster {i} {genotype} expression')
            # Add the enrichment terms to the side of the plot as a legend
            legend = ['Terms:']
            # Sort the terms by p-value
            for cluster in cluster_terms:
                cluster_terms[cluster] = sorted(cluster_terms[cluster], key=lambda x: x[1])

            enriched_genes = set()
            c = 0
            for term,pval,genes in cluster_terms[i]:
                if any([word in term.lower() for word in positive_words]) and pval<.05:
                    legend.append(f'• {term} {pval:.2e}')
                    enriched_genes.update(genes)
                    c += 1
                    if c > 10: break

            legend.append('\n')
            legend.append('Genes:')
            legend.append(', '.join(enriched_genes))
            # Add blank space to the bottom of the plot to accomodate the text
            # Remove the axis labels and ticks
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove the gridlines
            ax.grid(False)

        plt.subplots_adjust(bottom=.4, left=.05)
        # Add the terms to the plot as text below the plot
        plt.text(0,-.1, '\n'.join(legend), wrap=True, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                transform=plt.gca().transAxes)
        # Save the plot
        plt.savefig(f'../figures/gene_cluster_heatmaps/cluster_{i}_expression.png', dpi=300)
        # Clear the plot
        plt.clf()
# %%