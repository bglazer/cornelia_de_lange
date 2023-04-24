#%%
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
from util import umap_axes
from matplotlib import patches as mpatches
from random import seed
import scvelo as scv
import pickle

#%% 
# Set the random seed for reproducibility
np.random.seed(42)
seed(42)

#%% 
# Load the processed data
# wt = sc.read_h5ad('../data/wildtype_processed.h5ad')
# mut = sc.read_h5ad('../data/mutant_processed.h5ad')
# adata = wt.concatenate(mut, batch_categories=['wildtype', 'mutant'])
# adata.obs['batch'] = adata.obs['batch'].astype('category')

wt = sc.read_h5ad('../data/wildtype_net.h5ad')
mut = sc.read_h5ad('../data/mutant_net.h5ad')
network_data = wt.concatenate(mut, batch_categories=['wildtype', 'mutant'])

# %%
# Find the minimum distance between each wild type cell and mutant cell
# Import the kd-tree data structure
from scipy.spatial import KDTree
kdtree = KDTree(wt.X.toarray())
results = kdtree.query(mut.X.toarray())

# %%
neighbor_dists, neighbor_idxs = results
# Plot the distribution of distances
plt.hist(neighbor_dists, bins=100);
plt.xlabel('Distance to nearest wildtype cell')
plt.ylabel('Number of mutant cells')

#%%
mut_emb = mut.obsm['X_umap']
wt_emb = wt.obsm['X_umap']

plt.figure(figsize=(10,10))
plt.scatter(mut_emb[:,0], mut_emb[:,1],
            s=4, c=neighbor_dists/neighbor_dists.max(),
            cmap='Purples')
# Remove the ticks and labels from the plot
umap_axes(plt.gca())
plt.title('UMAP embedding of mutant data colored by distance to nearest wildtype cell')
# Remove the gridlines
plt.grid(False)
# Add a colorbar
plt.colorbar()

#%%
# Plot the mutant and wildtype cells side by side 
# colored by pseudotime
wt_pseudotime = np.array(wt.obs['pseudotime'])
mut_pseudotime = np.array(mut.obs['pseudotime'])
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].scatter(wt_emb[:,0], wt_emb[:,1],
            s=.5, c=wt_pseudotime)
axs[0].set_title('Wildtype')
axs[1].scatter(mut_emb[:,0], mut_emb[:,1],
            s=.5, c=mut_pseudotime)
axs[1].set_title('Mutant')
umap_axes(axs)

# %%
# Plot pseudotime vs cluster expression for WT and mutant side by side
from sklearn.linear_model import BayesianRidge
import scipy

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
    wt_expr = wt_expression_sum_nrm[:,i].reshape(-1,1)
    mut_expr = mut_expression_sum_nrm[:,i].reshape(-1,1)
    fit = regression.fit(wt_pseudotime.reshape(-1,1), wt_expr)
    regressions.append(fit)
    x = np.linspace(0,1,100).reshape(-1,1)
    wt_y, wt_std = fit.predict(x, return_std=True)
    axs[i,0].plot(x, wt_y, c='r')

    # Plot the WT regression predictions on the mutant plot
    # For each point in the mutant plot calculate the number of standard deviations away from the regression line
    mut_y, mut_std = fit.predict(mut_pseudotime.reshape(-1,1), return_std=True)
    axs[i,1].plot(mut_pseudotime, mut_y, c='r')
    # Given the standard deviation of the WT regression line, 
    # calculate the probability of the mutant data
    error = mut_y - mut_expr[:,0]
    mut_prob = scipy.stats.norm.pdf(error, loc=0, scale=mut_std)
    
        
    axs[i,0].scatter(wt_pseudotime, wt_expression_sum_nrm[:,i], s=1, alpha=1)
    axs[i,1].scatter(mut_pseudotime, mut_expression_sum_nrm[:,i], 
                     s=1, alpha=1, c=neighbor_dists)
    axs[i,0].set_ylabel(f'Cluster {i} expression', fontsize=16)
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
wt_cell_types = wt.obs['cell_type']
mut_cell_types = mut.obs['cell_type']

cell_ints = list(set(wt_cell_types) | set(mut_cell_types))
wt_cell_ints = [cell_ints.index(cell_type) for cell_type in wt_cell_types]
mut_cell_ints = [cell_ints.index(cell_type) for cell_type in mut_cell_types]
fig, axs = plt.subplots(1,2, figsize=(10,5))
# Plot each cell type individiually so we can add a legend
for i, cell_type in enumerate(cell_ints):
    wt_mask = np.array(wt_cell_ints) == i
    mut_mask = np.array(mut_cell_ints) == i
    axs[0].scatter(wt_emb[wt_mask,0], wt_emb[wt_mask,1], s=.5, alpha=1, 
                   color=plt.cm.tab20(i), label=cell_type)
    axs[1].scatter(mut_emb[mut_mask,0], mut_emb[mut_mask,1], s=.5, alpha=1, 
                   color=plt.cm.tab20(i), label=cell_type)

axs[0].set_title('Wildtype')
axs[1].set_title('Mutant')
# Remove the ticks
umap_axes(axs)

# Add a legend for the cell types
handles = [mpatches.Patch(color=plt.cm.tab20(i), label=cell_ints[i]) 
           for i in range(len(cell_ints))]
axs[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);
# Save the plot
plt.savefig(f'../figures/umap_cell_types.png', dpi=300)

# %%
#  Plot the UMAP with the cell line labels
wt_cell_lines = wt.obs['cell_line']
mut_cell_lines = mut.obs['cell_line']

cell_ints = list(set(wt_cell_lines) | set(mut_cell_lines))
wt_cell_ints = [cell_ints.index(cell_type) for cell_type in wt_cell_lines]
mut_cell_ints = [cell_ints.index(cell_type) for cell_type in mut_cell_lines]
fig, axs = plt.subplots(1,2, figsize=(10,5))
# Plot each cell type individiually so we can add a legend
for i, cell_line in enumerate(cell_ints):
    wt_mask = np.array(wt_cell_ints) == i
    mut_mask = np.array(mut_cell_ints) == i
    axs[0].scatter(wt_emb[wt_mask,0], wt_emb[wt_mask,1], s=.5, alpha=1, 
                   color=plt.cm.tab20(i), label=cell_line)
    axs[1].scatter(mut_emb[mut_mask,0], mut_emb[mut_mask,1], s=.5, alpha=1, 
                   color=plt.cm.tab20(i), label=cell_line)

axs[0].set_title('Wildtype')
axs[1].set_title('Mutant')
# Add a legend for the cell types
handles = []

for i, cell_type in enumerate(cell_ints):
    handles.append(mpatches.Patch(color=plt.cm.tab20(i), label=cell_type))

axs[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);
# Save the plot
plt.savefig(f'../figures/umap_cell_types.png', dpi=300)
umap_axes(axs)

#%%
# Plot the UMAP of each mutant mouse line separately
# Get a list of cell lines
mut_cell_lines = list(set(mut.obs['cell_line']))
# Make a plot with 3 columns and 2 rows
fig, axs = plt.subplots(2,3, figsize=(10,5))
# Plot each mouse line separately
# Get the x and y limits of the UMAP
x_min = np.min(mut_emb[:,0])
x_max = np.max(mut_emb[:,0])
y_min = np.min(mut_emb[:,1])
y_max = np.max(mut_emb[:,1])
# Add a buffer around the limits
x_buffer = (x_max - x_min) * .1
y_buffer = (y_max - y_min) * .1
x_min -= x_buffer
x_max += x_buffer
y_min -= y_buffer
y_max += y_buffer

for i, cell_line in enumerate(mut_cell_lines):
    # Get the cells that are from the current mouse line
    mask = mut.obs['cell_line'] == cell_line
    # Plot the UMAP
    axs[i//3, i%3].scatter(mut_emb[mask,0], mut_emb[mask,1], s=.5, alpha=1, 
                           color=plt.cm.tab20(i), label=cell_line)
    # Set the x and y limits
    axs[i//3, i%3].set_xlim(x_min, x_max)
    axs[i//3, i%3].set_ylim(y_min, y_max)
    axs[i//3, i%3].set_title(cell_line)
    # Remove the ticks
    axs[i//3, i%3].set_xticks([])
    axs[i//3, i%3].set_yticks([])

# %%
# Make a heatmap of the expression for each gene in the cluster across time
# Get the expression of genes in each cluster
protein_id_to_row = wt.uns['id_row']
wt_gene_expr = []
for cluster, gene_ids in enumerate(cluster_assignments):
    wt_gene_expr.append(np.zeros((wt.shape[0], len(gene_ids))))
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
# *******************
# TODO THIS IS NOT WORKING
# *******************
# Plot the heatmap
# make a directory to save the heatmaps
# import os
# if not os.path.exists('../figures/expression_heatmap'):
#     os.makedirs('../figures/expression_heatmap')

# for cluster in range(len(cluster_assignments)):
#     _=plt.figure(figsize=(10,5));
#     plt.imshow(wt_gene_expr[cluster].T, interpolation='none', aspect='auto', cmap='viridis');
#     plt.ylabel('Gene');
#     plt.xlabel('Cell');
#     plt.title('Wildtype');
#     # Remove the vertical grid
#     plt.colorbar();
#     # Add gene name labels to the x axis
#     gene_ids = [list(protein_id_to_name[gene_id])[0] for gene_id in cluster_assignments[cluster]] + ['Mean']
#     plt.yticks(np.arange(len(gene_ids)), gene_ids, fontsize=8);
#     plt.savefig(f'../figures/expression_heatmap/cluster_{i}_gene_expression_heatmap.png', dpi=300);
#     plt.clf();


# %%
# Import the Bayesian node ranking results
ranking = pickle.load(open('../data/graph_ranked_genes.pickle', 'rb')) 
# %%
# Plot the expression of each quintile of the ranking
# First, get the ids of the genes in each quintile
n_groups = 10
n_genes = len(ranking)
percentile_size = n_genes // n_groups

fig, axs = plt.subplots(n_groups,2, figsize=(10,2*n_groups))
for i in range(n_groups):
    # Get the genes in the current quintile
    genes = ranking[i*percentile_size:(i+1)*percentile_size]
    mut_rows = []
    wt_rows = []
    for id,name in genes:
        if id in mut_id_new_row:
            mut_rows.append(mut_id_new_row[id])
        if id in wt_id_new_row:
            wt_rows.append(wt_id_new_row[id])
    # Get the expression of the genes in the current quintile
    mut_gene_expr = mut_X[:, mut_rows].sum(axis=1)
    wt_gene_expr = wt_X[:, wt_rows].sum(axis=1)
    # Plot the expression of the genes in the current quintile on the UMAP embedding
    axs[i,0].scatter(wt_emb[:,0], wt_emb[:,1], c=wt_gene_expr, s=.1, alpha=.5)
    axs[i,1].scatter(mut_emb[:,0], mut_emb[:,1], c=mut_gene_expr, s=.1, alpha=.5)
    # Remove the ticks
    axs[i,0].set_xticks([])
    axs[i,0].set_yticks([])
    axs[i,1].set_xticks([])
    axs[i,1].set_yticks([])
    
# %%
# Cluster gene expression in the mutant and wildtype samples, using only the network genes
sc.pp.neighbors(mut, n_neighbors=10, n_pcs=40)
sc.pp.neighbors(wt, n_neighbors=10, n_pcs=40)
sc.tl.leiden(mut, key_added='leiden', resolution=0.5)
sc.tl.leiden(wt, key_added='leiden', resolution=0.5)

# %%
# Plot the clustering of the network genes in the mutant and wildtype samples
fig, axs = plt.subplots(1,2, figsize=(10,5))
# Plot the leiden clusters on the UMAP embedding
wt_cluster_colors = np.array(wt.obs['leiden'], dtype=int)
mut_cluster_colors = np.array(mut.obs['leiden'], dtype=int)

# Plot the clusters
for i in range(wt_cluster_colors.max()+1):
    axs[0].scatter(wt_emb[wt_cluster_colors==i,0], wt_emb[wt_cluster_colors==i,1], s=.1, label=f'Cluster {i}')
for i in range(mut_cluster_colors.max()+1):
    axs[1].scatter(mut_emb[mut_cluster_colors==i,0], mut_emb[mut_cluster_colors==i,1], s=.1, label=f'Cluster {i}')
axs[0].set_title('Wildtype')
axs[1].set_title('Mutant')
umap_axes(axs)
# Add legends to the plots
# Increase the size of the legend markers
axs[0].legend(fontsize=6, markerscale=10);
axs[1].legend(fontsize=6, markerscale=10);

# %%
# Identify the cluster that has the highest distance from mutant to WT cells
cluster_dists = []
for i in range(mut_cluster_colors.max()+1):
    mut_cluster_mask = mut_cluster_colors==i
    # Get the closest WT neighbor for each cell in the cluster
    cluster_dists.append(neighbor_dists[mut_cluster_mask].mean())
    
# %%
# Calculate the most highly expressed genes in the 3 clusters with the highest 
# distances from mutant to WT cells
cluster_ids = np.argsort(cluster_dists)[::-1]
for i in cluster_ids:
    mut_cluster_mask = mut_cluster_colors==i
    # Get the expression of the genes in the current cluster
    mut_gene_expr = mut[mut_cluster_mask, :].X.mean(axis=0)
    # Sort to get the most highly expressed genes
    mut_gene_expr_sorted = np.argsort(mut_gene_expr)[::-1]
    # Get the names of the most highly expressed genes
    mut_gene_names = [mut.var_names[i] for i in mut_gene_expr_sorted]
    print(f'Cluster {i} most highly expressed genes:')
    for i in range(10):
         print(f'{mut_gene_names[i]:10s}: {mut_gene_expr[mut_gene_expr_sorted[i]]:.2f}')
    print('---------')

# %%
from itertools import combinations
from scipy.stats import ttest_ind
# For every pair of clusters, calculate the differentially expressed genes
# between the clusters
cluster_diff_expr = {}
cluster_diff_expr_pval = {}
cluster_diff_expr_pval_corrected = {}
num_tests = 0
ngenes = mut.shape[1]

for cluster_a, cluster_b in combinations(range(mut_cluster_colors.max()), r=2):
    # Get the cells in each cluster
    cluster_a_mask = mut_cluster_colors==cluster_a
    cluster_b_mask = mut_cluster_colors==cluster_b
    # Get the expression of the genes in each cluster
    cluster_a_expr = mut[cluster_a_mask, :].X
    cluster_b_expr = mut[cluster_b_mask, :].X
    # Calculate the mean expression of each gene in each cluster
    cluster_a_mean = cluster_a_expr.mean(axis=0)
    cluster_b_mean = cluster_b_expr.mean(axis=0)
    # Calculate the fold change between the clusters
    cluster_diff_expr[(cluster_a, cluster_b)] = (cluster_a_mean - cluster_b_mean)
    # Calculate the p-value for each difference of means using a t-test
    # TODO why are some of the p-values nan?
    cluster_diff_expr_pval[(cluster_a, cluster_b)] = ttest_ind(cluster_a_expr, cluster_b_expr, axis=0)[1]
    # Increment the number of tests
    num_tests += len(mut.var_names)

#%%


# Correct the p-values for multiple testing
from statsmodels.stats.multitest import multipletests
# Get the p-values  
pvals = np.array(list(cluster_diff_expr_pval.values())).flatten()
# Correct the p-values
# TODO redo this with fdr_bh when I figure out how to remove nans
pvals_corrected = multipletests(pvals, method='bonferroni')[1]
# Add the corrected p-values to the dictionary
for i, (cluster_a, cluster_b) in enumerate(cluster_diff_expr_pval.keys()):
    cluster_diff_expr_pval_corrected[(cluster_a, cluster_b)] = pvals_corrected[i*ngenes:(i+1)*ngenes]

# For each cluster pair, get the most significantly differentially expressed genes
for cluster_a, cluster_b in cluster_diff_expr_pval.keys():
    # Get the p-values for the current cluster pair
    pvals = cluster_diff_expr_pval_corrected[(cluster_a, cluster_b)]
    # Get the fold change of the most significant genes
    fold_change = cluster_diff_expr[(cluster_a, cluster_b)]
    # Sort the fold changes
    fc_sort = np.argsort(np.abs(fold_change))[::-1]
    # Filter to only the significant genes (pval < 0.05)
    fc_sort = fc_sort[pvals[fc_sort] < 0.05]
    # Get the names of the most significant genes
    gene_names = [mut.var_names[i] for i in fc_sort]
    # Filter to only the significant genes (pval < 0.05)
    pvals = pvals[fc_sort]
    fold_change = fold_change[fc_sort]
    # Print the most significant genes
    print(f'Cluster {cluster_a} vs {cluster_b} most significant genes:')
    print(f'{"Gene":10s}: Mean Expr. Diff, P-value (corrected)')
    for i in range(10):
        print(f'{gene_names[i]:10s}: {fold_change[i]:< 15.3f}  {pvals[i]:.2e}')
    print('---------')


# %%
