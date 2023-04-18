#%%
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from util import umap_axes
import pyVIA.core as via
import pickle
import scipy

#%%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%%
network_data = sc.read_h5ad('../data/combined_net.h5ad')
wt_data = sc.read_h5ad('../data/wildtype_net.h5ad')
mut_data = sc.read_h5ad('../data/mutant_net.h5ad')

#%%
def cluster_sum(adata):
    # Import the cluster assigments of each gene from the Tiana et al paper
    cluster_assignments = pickle.load(open('../data/louvain_clusters.pickle', 'rb'))
    # Convert the cluster assignments to indices of the genes in the filtered data
    cluster_indexes = []
    # Calculate the sum of the expression of each gene in each cluster
    cluster_sums = np.zeros((adata.n_obs, len(cluster_assignments))) 

    id_row = network_data.uns['id_row']

    for i,gene_ids in enumerate(cluster_assignments):
        # Get all the rows in the data that correspond to the current cluster
        for gene_id in gene_ids:
            if gene_id in id_row:
                # todense() returns a matrix, A1 returns a 1d array
                cluster_sums[:,i] += adata.X[:, id_row[gene_id]].todense().A1
    return cluster_sums

wt_cluster_sums = cluster_sum(wt_data)
mut_cluster_sums = cluster_sum(mut_data)

#%%
# Add the cluster sums to the data sets
wt_data.uns['cluster_sums'] = wt_cluster_sums
mut_data.uns['cluster_sums'] = mut_cluster_sums

#%%
# Assign initial points to be cells with high expression of cluster 0
# Get the index of 99th percentile of cluster 0 expression
wt_top_percentile = np.percentile(wt_cluster_sums[:,0], 99)
mut_top_percentile = np.percentile(mut_cluster_sums[:,0], 99)
# Get the cells with expression above the 99th percentile
wt_initial_points = np.where(wt_cluster_sums[:,0] > wt_top_percentile)[0]
mut_initial_points = np.where(mut_cluster_sums[:,0] > mut_top_percentile)[0]
wt_data.uns['initial_points_via'] = wt_initial_points
mut_data.uns['initial_points_via'] = mut_initial_points
# Add the initial points to the data sets

#%%
def calculate_pseudotime(adata, initial):
    genotype = adata.uns['genotype']
    # Check if the data is in scipy sparse package
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    pseudotime = via.VIA(X, 
                        knn=30,
                        cluster_graph_pruning_std=0.5,
                        jac_std_global=.15,
                        too_big_factor=.2,
                        root_user=initial,
                        random_seed=42)
    pseudotime.run_VIA()
    # Dump the VIA object to a pickle file
    pickle.dump(pseudotime, open(f'../data/{genotype}_pseudotime.pickle', 'wb'))
    return pseudotime

# wt_data.X = wt_data.X.toarray()
# mut_data.X = mut_data.X.toarray()
wt_pseudotime = calculate_pseudotime(wt_data, wt_initial_points)
mut_pseudotime = calculate_pseudotime(mut_data, mut_initial_points)

#%%
for genotype in ['wildtype', 'mutant']:
    if genotype == 'wildtype':
        network_data = wt_data
        pseudotime = wt_pseudotime
    else:
        network_data = mut_data
        pseudotime = mut_pseudotime
    plt.figure(figsize=(10,10))
    plt.scatter(network_data.obsm['X_umap'][:,0], network_data.obsm['X_umap'][:,1], 
                c = pseudotime.single_cell_pt_markov,
                cmap = 'plasma',
                s=1, alpha=.5)
    plt.title(f'{genotype.capitalize()} pseudotime')
    umap_axes(plt.gca())
    # Add a colorbar
    plt.colorbar()

    plt.savefig(f'../figures/pseudotime/{genotype}_pseudotime.png', dpi=300)

    via.draw_trajectory_gams(via_object=pseudotime, embedding=network_data.obsm['X_umap'], 
                            draw_all_curves=False)
    plt.savefig(f'../figures/pseudotime/{genotype}_pseudotime_gams.png', dpi=300)

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

#%%
# Add the pseudotime to the wildtype and mutant datasets
# then save them as h5ad files
wt_data.obs['pseudotime'] = wt_pseudotime.single_cell_pt_markov
mut_data.obs['pseudotime'] = mut_pseudotime.single_cell_pt_markov
wt_data.obsm['transition_matrix'] = wt_pseudotime.sc_transition_matrix(smooth_transition=1)
mut_data.obsm['transition_matrix'] = mut_pseudotime.sc_transition_matrix(smooth_transition=1)
wt_data.write_h5ad('../data/wildtype_net_pseudotime.h5ad')
mut_data.write_h5ad('../data/mutant_net_pseudotime.h5ad')
# %%
