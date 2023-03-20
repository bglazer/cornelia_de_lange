#%%
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

#%%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%%
f = open('../data/raw-counts-mes-mutant.csv','r')
genotype = 'mutant'

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
expression = np.array(expression)


#%%
adata = sc.AnnData(expression.T)
adata.var_names = names_in_data
# adata.obsm['X'] = expression
#%%
# Plot the overall distribution of total gene expression
plt.hist(adata.X.sum(axis=1), bins=100)
plt.title('Distribution of total gene expression per cell across all genes');
plt.savefig(f'../figures/all_genes_total_expression_per_cell_{genotype}.png', dpi=300)

#%%
# Plot the distribution of gene expression for each gene
plt.hist(np.log10(adata.X.sum(axis=0)+1), bins=100)
plt.title('Log Distribution of total expression per gene across all cells');
plt.savefig(f'../figures/all_genes_log_expression_per_gene_{genotype}.png', dpi=300)

#%%
# Plot the number of genes with expression > 0 per cell
plt.hist((adata.X>0).sum(axis=0), bins=100);
plt.title('Distribution of number of cells with expression > 0 per gene');
plt.savefig(f'../figures/all_genes_nonzero_expression_per_gene_{genotype}.png', dpi=300)

#%% 
# Plot the cumulative distribution of total gene expression per cell
plt.hist(adata.X.sum(axis=1), bins=100, cumulative=True);
plt.title('Cumulative distribution of total gene expression per cell');
plt.savefig(f'../figures/all_genes_cumulative_expression_per_cell_{genotype}.png', dpi=300)

#%%
# use UMAP or PHate to obtain embedding that is used for single-cell level visualization
# sc.tl.pca(adata, svd_solver='arpack')
# sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
# sc.tl.umap(adata)
# Plot the UMAP embedding
# sc.pl.umap(adata, size=10, show=False)

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
sc.pp.neighbors(network_data, n_neighbors=15, n_pcs=30)
sc.tl.umap(network_data)

#%%
# Plot the overall distribution of total gene expression
plt.hist(network_data.X.sum(axis=1), bins=100)
plt.title('Distribution of total gene expression per cell across all genes');
plt.savefig(f'../figures/network_genes_total_expression_per_cell_{genotype}.png', dpi=300)

#%%
# Plot the distribution of gene expression for each gene
plt.hist(np.log10(network_data.X.sum(axis=0)+1), bins=100)
plt.title('Log Distribution of total expression per gene across all cells');
plt.savefig(f'../figures/network_genes_log_expression_per_gene_{genotype}.png', dpi=300)

#%%
# Plot the number of genes with expression > 0 per cell
plt.hist((network_data.X>0).sum(axis=0), bins=100);
plt.title('Distribution of number of cells with expression > 0 per gene');
plt.savefig(f'../figures/network_genes_nonzero_expression_per_gene_{genotype}.png', dpi=300)

#%% 
# Plot the cumulative distribution of total gene expression per cell
plt.hist(network_data.X.sum(axis=1), bins=100, cumulative=True);
plt.title('Cumulative distribution of total gene expression per cell');
plt.savefig(f'../figures/network_genes_cumulative_expression_per_cell_{genotype}.png', dpi=300)

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
# Convert to a dictionary of clusters and terms with only the p<.01 terms
cluster_terms = {}
for cluster,term,pval,genes in cluster_enrichment:
    if pval < .01:
        if cluster not in cluster_terms: cluster_terms[cluster] = []
        cluster_terms[cluster].append((term,pval,genes))

#%%
# Plot the UMAP embedding of the filtered data colored by the total gene expression of each cluster
# Normalize the expression of each cluster
from matplotlib import pyplot as plt
expression_sum_nrm = cluster_sums/(cluster_sums.max(axis=0))

# Filter words that are relevant to development
positive_words = ['develop', 'signal', 'matrix', 
                'organization', 'proliferation', 'stem', 'pathway', 'epithel', 'mesenchym',
                'morpho', 'mesoderm', 'endoderm', 'different', 'specification']

plt.figure(figsize=(20,10))
for i in range(cluster_sums.shape[1]):
    if i in cluster_terms:
        plt.scatter(network_data.obsm['X_umap'][:,0], 
                    network_data.obsm['X_umap'][:,1], 
                    c=expression_sum_nrm[:,i], 
                    s=10, alpha=.5, cmap='viridis', vmin=0, vmax=1)
        # Title the plot
        plt.title(f'Cluster {i} {genotype} expression')
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
        # Add blank space to the right side of the plot
        plt.subplots_adjust(right=.7)
        # Add the terms to the plot as a box to the right side
        plt.text(1.05, .5, '\n'.join(legend), verticalalignment='center', wrap=True, fontsize=12,
                 transform=plt.gca().transAxes)
        # Remove the axis labels and ticks
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])

        # Save the plot
        plt.savefig(f'../figures/gene_cluster_heatmaps/{genotype}_cluster_{i}_expression.png', dpi=300)
        # Clear the plot
        plt.clf()


#%%
# Assign initial points to be cells with high expression of cluster 0
# Get the index of 99th percentile of cluster 0 expression
top_percentile = np.percentile(cluster_sums[:,0], 99)
# Get the cells with expression above the 99th percentile
initial_points = np.where(cluster_sums[:,0] > top_percentile)[0]

#%%
import pyVIA.core as via
pseudotime = via.VIA(network_data.X, random_seed=42, root_user=initial_points)
pseudotime.run_VIA() 

# %%
f, ax = via.plot_scatter(embedding = network_data.obsm['X_umap'], 
                         labels = pseudotime.single_cell_pt_markov,
                         cmap = 'plasma')
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
