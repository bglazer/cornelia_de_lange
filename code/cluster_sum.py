#%%
import scanpy as sc
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util import umap_axes

#%%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%%
# network_data = sc.read_h5ad('../data/combined_net.h5ad')
wt = sc.read_h5ad('../data/wildtype_net.h5ad')
mut = sc.read_h5ad('../data/mutant_net.h5ad')

#%%
def cluster_sum(adata):
    # Import the cluster assigments of each gene from the Tiana et al paper
    cluster_assignments = pickle.load(open('../data/louvain_clusters.pickle', 'rb'))
    # Convert the cluster assignments to indices of the genes in the filtered data
    cluster_indexes = []
    # Calculate the sum of the expression of each gene in each cluster
    cluster_sums = np.zeros((adata.n_obs, len(cluster_assignments))) 

    id_row = adata.uns['id_row']

    for i,gene_ids in enumerate(cluster_assignments):
        # Get all the rows in the data that correspond to the current cluster
        for gene_id in gene_ids:
            if gene_id in id_row:
                # todense() returns a matrix, A1 returns a 1d array
                cluster_sums[:,i] += adata.X[:, id_row[gene_id]].todense().A1
    return cluster_sums

wt_cluster_sums = cluster_sum(wt)
mut_cluster_sums = cluster_sum(mut)

#%%
# Add the cluster sums to the data sets
wt.obsm['cluster_sums'] = wt_cluster_sums
mut.obsm['cluster_sums'] = mut_cluster_sums
# %%
# Save the data sets
# wt.write_h5ad('../data/wildtype_net.h5ad')
# mut.write_h5ad('../data/mutant_net.h5ad')


#%%
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))

#%%
# Get the GO enrichment associated with each cluster
cluster_enrichment = pickle.load(open('../data/cluster_enrichments_louvain.pickle', 'rb'))
# Convert to a dictionary of clusters and terms with only the p<.01 terms
cluster_terms = {}
for cluster,term,pval,genes in cluster_enrichment:
    if pval < .01:
        if cluster not in cluster_terms: cluster_terms[cluster] = []
        cluster_terms[cluster].append((term,pval,genes))
cluster_assignments = pickle.load(open('../data/louvain_clusters.pickle', 'rb'))
#%%
# Setup the PCA 
from sklearn.decomposition import PCA
pca = PCA()
# Set the PC mean and components
pca.mean_ = mut.uns['pca_mean']
pca.components_ = mut.uns['PCs']
X_pca_wt = pca.transform(wt.X.toarray()) 
X_pca_mut = pca.transform(mut.X.toarray())
#%%
# Plot the UMAP embedding of the filtered data colored by the total gene expression of each cluster
# Normalize the expression of each cluster
max_expression = np.max(np.vstack((wt.obsm['cluster_sums'].max(axis=0), mut.obsm['cluster_sums'].max(axis=0))), axis=0)
wt_expression_sum_nrm = wt.obsm['cluster_sums']/max_expression
mut_expression_sum_nrm = mut.obsm['cluster_sums']/max_expression

# Filter words that are relevant to development
positive_words = ['develop', 'signal', 'matrix', 
                'organization', 'proliferation', 'stem', 'pathway', 'epithel', 'mesenchym',
                'morpho', 'mesoderm', 'endoderm', 'different', 'specification']

# Plot the cluster sums for wildtype and mutant side by side
n_clusters = wt.obsm['cluster_sums'].shape[1]
for i in range(n_clusters):
    if i in cluster_terms:
        print(i, flush=True)
        fig,axs = plt.subplots(1,2, figsize=(11,11.5));
        for genotype in ['wildtype', 'mutant']:
            if genotype == 'wildtype':
                ax = axs[0]
                X = X_pca_wt
                expression_sum_nrm = wt_expression_sum_nrm
            else:
                ax = axs[1]
                X = X_pca_mut
                expression_sum_nrm = mut_expression_sum_nrm
            # Plot the UMAP embedding of the filtered data colored by
            #  the total gene expression of each cluster
            _=ax.scatter(X[:,0], X[:,1], 
                       c=expression_sum_nrm[:,i], 
                       s=10, alpha=.5, cmap='viridis', vmin=0, vmax=1)
            # Title the plot
            _=ax.set_title(f'Cluster {i} {genotype} expression')
            # Add the enrichment terms to the side of the plot as a legend
            legend = ['Terms:']
            # Sort the terms by p-value
            for cluster in cluster_terms:
                cluster_terms[cluster] = sorted(cluster_terms[cluster], key=lambda x: x[1])

            enriched_genes = set()
            all_genes = ['\\'.join(protein_id_to_name[protein_id]) for protein_id in cluster_assignments[i]]
            c = 0
            for term,pval,genes in cluster_terms[i]:
                if any([word in term.lower() for word in positive_words]) and pval<.05:
                    legend.append(f'• {term} {pval:.2e}')
                    enriched_genes.update(genes)
                    c += 1
                    if c > 10: break

            legend.append('\n')
            legend.append('Matched Genes:')
            legend.append(', '.join(enriched_genes))
            legend.append('\n')
            legend.append('All Genes in Cluster:')
            legend.append(', '.join(all_genes))
            # Add blank space to the bottom of the plot to accomodate the text
            # Remove the axis labels and ticks

        umap_axes(axs);

        _=plt.subplots_adjust(bottom=.4, left=.05)
        # Add the terms to the plot as text in the middle of the figure below the axes
        _=plt.figtext(.05, .35, '\n'.join(legend), 
                      horizontalalignment='left',
                      verticalalignment='top',
                      fontsize=12, wrap=True)
        # Save the plot
        # _=plt.savefig(f'../figures/gene_cluster_heatmaps/cluster_{i}_expression.png', dpi=300)
        # Clear the plot
        _=plt.clf();

# %%
