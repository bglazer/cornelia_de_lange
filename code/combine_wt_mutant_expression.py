#%%
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
from umap import UMAP
from util import filter_to_network

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

network_data, id_row, id_new_row = filter_to_network(adata)
wt_net, wt_id_row, wt_id_new_row = filter_to_network(wt)
mut_net, mut_id_row, mut_id_new_row = filter_to_network(mut)

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
    fig,axs = plt.subplots(1,2, figsize=(10,10));
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
            all_genes = set()
            c = 0
            for term,pval,genes in cluster_terms[i]:
                if any([word in term.lower() for word in positive_words]) and pval<.05:
                    legend.append(f'• {term} {pval:.2e}')
                    enriched_genes.update(genes)
                    all_genes.update(genes)
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
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove the gridlines
            ax.grid(False)

        plt.subplots_adjust(bottom=.4, left=.05)
        # Add the terms to the plot as text in the middle of the figure below the axes
        plt.figtext(.05, .35, '\n'.join(legend), 
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=12, wrap=True)
        # Save the plot
        plt.savefig(f'../figures/gene_cluster_heatmaps/cluster_{i}_expression.png', dpi=300)
        # Clear the plot
        plt.clf();

# %%
# Plot pseudotime vs cluster expression for WT and mutant side by side
from sklearn.linear_model import BayesianRidge
import scipy

n_clusters = cluster_sums.shape[1]
fig, axs = plt.subplots(n_clusters, 3, figsize=(15,5*n_clusters))
regressions = []
for i in range(n_clusters):
    # Rmove the vertical gridlines
    axs[i,0].grid(axis='x')
    axs[i,1].grid(axis='x')
    # Set the y axis limits to be the same for both plots
    axs[i,0].set_ylim(0, 1)
    axs[i,1].set_ylim(0, 1)
    # Set the x axis limits to be the same for both plots
    axs[i,0].set_xlim(0, 1)
    axs[i,1].set_xlim(0, 1)

    # Fit a linear regression of the relationship between the pseudotime and cluster expression
    regression = BayesianRidge()
    wt_pseudotime = np.array(wt_via.single_cell_pt_markov).reshape(-1,1)
    mut_pseudotime = np.array(mut_via.single_cell_pt_markov).reshape(-1,1)
    wt_expr = expression_sum_nrm[:wt.shape[0],i]
    mut_expr = expression_sum_nrm[wt.shape[0]:,i]
    fit = regression.fit(wt_pseudotime, wt_expr)
    regressions.append(fit)
    x = np.linspace(0,1,100).reshape(-1,1)
    wt_y, wt_std = fit.predict(x, return_std=True)
    axs[i,0].plot(x, wt_y, c='r')

    # Plot the WT regression predictions on the mutant plot
    # For each point in the mutant plot calculate the number of standard deviations away from the regression line
    mut_y, mut_std = fit.predict(mut_pseudotime, return_std=True)
    axs[i,1].plot(mut_pseudotime, mut_y, c='r')
    # Given the standard deviation of the WT regression line, 
    # calculate the probability of the mutant data
    error = mut_y - mut_expr
    mut_prob = scipy.stats.norm.pdf(error, loc=0, scale=mut_std)
    
        
    axs[i,0].scatter(wt_pseudotime, expression_sum_nrm[:wt.shape[0],i], s=1, alpha=1)
    axs[i,1].scatter(mut_pseudotime, expression_sum_nrm[wt.shape[0]:,i], 
                     s=1, alpha=1, c=dists)
    axs[i,0].set_ylabel(f'Cluster {i} expression', fontsize=8)
    # Plot standard error bars of the WT regression line on the mutant data
    for j in range(3):
        axs[i,0].fill_between(x.flatten(), wt_y-wt_std*(j+1), wt_y+wt_std*(j+1), color='grey', alpha=.2)
        axs[i,1].fill_between(x.flatten(), wt_y-wt_std*(j+1), wt_y+wt_std*(j+1), color='grey', alpha=.2)

    # Plot the mutant data colored by the probability of the data given the WT regression line
    gt3std = np.abs(error) > mut_std*3
    gt2std = np.abs(error) > mut_std*2
    # Convert boolean array to int array so we can add them for the color map
    outliers = gt2std.astype(int) + gt3std.astype(int)
    axs[i,2].set_title(f'Cells >2 std from WT pseudotime regression')
    axs[i,2].scatter(mut_emb[:,0], mut_emb[:,1], s=1, alpha=.75, c=outliers, cmap='Reds')
    # Print the percentage of points outside 2 and 3 standard deviations
    # as text in the upper right corner of the plot
    xlim = axs[i,2].get_xlim()[1]
    ylim = axs[i,2].get_ylim()[1]
    std_text = f'>2 std: {gt2std.sum()/mut_y.shape[0]*100:.2f}%'
    std_text += f'\n>3 std: {gt3std.sum()/mut_y.shape[0]*100:.2f}%'
    axs[i,2].text(xlim*.95, ylim*.95, 
                  horizontalalignment='right',
                  verticalalignment='top',
                  s=std_text)
    # Remove the axis labels and ticks
    axs[i,2].set_xlabel('UMAP 1')
    axs[i,2].set_ylabel('UMAP 2')
    axs[i,2].set_xticks([])
    axs[i,2].set_yticks([])
    axs[i,2].grid(False)
    
axs[0,0].set_title('Wildtype')
axs[0,1].set_title('Mutant')
axs[-1,0].set_xlabel('Pseudotime')
axs[-1,0].set_xlabel('Pseudotime');
# Save the plot
plt.savefig('../figures/pseudotime_vs_cluster_expression.png', dpi=300)


#%%
#  Plot the UMAP with the cell type labels
wt_cell_types = wt_net.obs['cell_type']
mut_cell_types = mut_net.obs['cell_type']

cell_ints = list(set(wt_cell_types) | set(mut_cell_types))
wt_cell_ints = [cell_ints.index(cell_type) for cell_type in wt_cell_types]
mut_cell_ints = [cell_ints.index(cell_type) for cell_type in mut_cell_types]
fig, axs = plt.subplots(1,2, figsize=(10,5))
# Plot each cell type individiually so we can add a legend
for i, cell_type in enumerate(cell_ints):
    wt_mask = np.array(wt_cell_ints) == i
    mut_mask = np.array(mut_cell_ints) == i
    axs[0].scatter(wt_emb[wt_mask,0], wt_emb[wt_mask,1], s=.5, alpha=1, 
                   c=plt.cm.tab20(i), label=cell_type)
    axs[1].scatter(mut_emb[mut_mask,0], mut_emb[mut_mask,1], s=.5, alpha=1, 
                   c=plt.cm.tab20(i), label=cell_type)

axs[0].set_title('Wildtype')
axs[1].set_title('Mutant')
# Remove the ticks
for i in range(2):
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_xlabel('UMAP 1')
    axs[i].set_ylabel('UMAP 2')
    axs[i].grid(False)

# Add a legend for the cell types
handles = [mpatches.Patch(color=plt.cm.tab20(i), label=cell_ints[i]) 
           for i in range(len(cell_ints))]
axs[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);
# Save the plot
plt.savefig(f'../figures/umap_cell_types.png', dpi=300)

# %%
#  Plot the UMAP with the cell line labels
wt_cell_lines = wt_net.obs['cell_line']
mut_cell_lines = mut_net.obs['cell_line']

cell_ints = list(set(wt_cell_lines) | set(mut_cell_lines))
wt_cell_ints = [cell_ints.index(cell_type) for cell_type in wt_cell_lines]
mut_cell_ints = [cell_ints.index(cell_type) for cell_type in mut_cell_lines]
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].scatter(wt_emb[:,0], wt_emb[:,1], s=1, alpha=.75,
               c=wt_cell_ints, cmap='tab20')
axs[1].scatter(mut_emb[:,0], mut_emb[:,1], s=1, alpha=.75, 
               c=mut_cell_ints, cmap='tab20')
axs[0].set_title('Wildtype')
axs[1].set_title('Mutant')
# Add a legend for the cell types
handles = []
from matplotlib import patches as mpatches
for i, cell_type in enumerate(cell_ints):
    handles.append(mpatches.Patch(color=plt.cm.tab20(i), label=cell_type))

axs[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);
# Save the plot
plt.savefig(f'../figures/umap_cell_types.png', dpi=300)

# %%
# Make a heatmap of the expression fo each gene in the cluster across time
# Get the expression of genes in each cluster
protein_id_to_row = pickle.load(open('../data/wildtype_id_row.pickle', 'rb'))
wt_gene_expr = []
for cluster, gene_ids in enumerate(cluster_assignments):
    wt_gene_expr.append(np.zeros((wt_net.shape[0], len(gene_ids))))
    # Sort cells by pseudotime
    wt_pseudotime_sorted = wt.X[np.argsort(wt_pseudotime.flatten()),:]
    # Get all the rows in the data that correspond to the current cluster   
    for i,gene_id in enumerate(gene_ids):
        if gene_id in protein_id_to_row:
            row = protein_id_to_row[gene_id]
            wt_gene_expr[cluster][:,i] = wt_pseudotime_sorted[:, row]
    mean = wt_gene_expr[cluster].mean(axis=1)
    wt_gene_expr[cluster] = np.hstack((wt_gene_expr[cluster], mean[:,None]))

# %%
# Plot the heatmap
# make a directory to save the heatmaps
import os
if not os.path.exists('../figures/expression_heatmap'):
    os.makedirs('../figures/expression_heatmap')

for cluster in range(len(cluster_assignments)):
    plt.figure(figsize=(10,5));
    plt.imshow(wt_gene_expr[cluster].T, interpolation='none', aspect='auto', cmap='viridis');
    plt.ylabel('Gene');
    plt.xlabel('Cell');
    plt.title('Wildtype');
    # Remove the vertical grid
    plt.colorbar();
    # Add gene name labels to the x axis
    protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
    gene_ids = [list(protein_id_to_name[gene_id])[0] for gene_id in cluster_assignments[cluster]] + ['Mean']
    plt.yticks(np.arange(len(gene_ids)), gene_ids, fontsize=8);
    plt.savefig(f'../figures/expression_heatmap/cluster_{i}_gene_expression_heatmap.png', dpi=300);
    plt.clf();

# %%
plt.plot(wt_gene_expr[0][:,-1])
# %%
