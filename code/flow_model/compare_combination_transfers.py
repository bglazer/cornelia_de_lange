#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import util
from tabulate import tabulate
import scanpy as sc
import networkx as nx
from itertools import combinations
import plotting
#%%
# Load the data
source_genotype = 'mutant'
target_genotype = 'wildtype'

src_tmstp = '20230607_165324' if source_genotype == 'wildtype' else '20230608_093734'
tgt_tmstp = '20230607_165324' if target_genotype == 'wildtype' else '20230608_093734'
tgt_data = sc.read_h5ad(f'../../data/{target_genotype}_net.h5ad')
src_data = sc.read_h5ad(f'../../data/{source_genotype}_net.h5ad')
tgt_outdir = f'../../output/{tgt_tmstp}'
src_outdir = f'../../output/{src_tmstp}'
transfer = f'{source_genotype}_to_{target_genotype}'
transfer_dir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations'
pltdir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/figures'
datadir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/data'

node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}
#%%
n_repeats = 10
cell_types = {c:i for i,c in enumerate(sorted(set(tgt_data.obs['cell_type'])))}
idx_to_cell_type = {v:k for k,v in cell_types.items()}
src_baseline_trajectories = pickle.load(open(f'{src_outdir}/baseline_trajectories_{source_genotype}.pickle', 'rb'))
src_baseline_trajectories_np = src_baseline_trajectories
src_baseline_idxs = pickle.load(open(f'{src_outdir}/baseline_nearest_cell_idxs_{source_genotype}.pickle', 'rb'))
src_baseline_cell_proportions, src_baseline_cell_errors = plotting.calculate_cell_type_proportion(src_baseline_idxs, src_data, cell_types, n_repeats, error=True)

tgt_baseline_trajectories = pickle.load(open(f'{tgt_outdir}/baseline_trajectories_{target_genotype}.pickle', 'rb'))
tgt_baseline_trajectories_np = tgt_baseline_trajectories
tgt_baseline_idxs = pickle.load(open(f'{tgt_outdir}/baseline_nearest_cell_idxs_{target_genotype}.pickle', 'rb'))
tgt_baseline_cell_proportions, tgt_baseline_cell_errors = plotting.calculate_cell_type_proportion(tgt_baseline_idxs, tgt_data, cell_types, n_repeats, error=True)

#%%
individual_proportions = {}
individual_distances = {}
individual_errors = {}
# Load all the individual transfer simulation results (cell type proportions)
# Plus one for the empty (null) transfer
for i in range(1, len(all_genes)):
    result = pickle.load(open(f'{datadir}/individual_combination_{i}_mutant_to_wildtype_transfer_cell_type_proportions.pickle', 'rb'))
    gene = tuple(result['transfer_genes'])[0]
    proportions = result['perturb_proportions']
    error = result['perturb_errors']
    individual_proportions[gene] = proportions
    individual_errors[gene] = error
    individual_distances = {gene: np.abs(proportions - src_baseline_cell_proportions) 
                            for gene, proportions in individual_proportions.items()}
#%%
# Load the baseline transfer (null, no transfer) simulation results
baseline_result = pickle.load(open(f'{datadir}/baseline_combination_0_mutant_to_wildtype_transfer_cell_type_proportions.pickle', 'rb'))
baseline_proportions = baseline_result['perturb_proportions']
baseline_error = baseline_result['perturb_errors']
baseline_distance = np.abs(baseline_proportions - src_baseline_cell_proportions).sum()
#%%
# Calculate a ranking of the genes by their mean expression
mean_expression = np.array(tgt_data.X.mean(axis=0)).flatten()
mean_expression_sort = np.argsort(mean_expression)[::-1]
mean_expression_rank = {idx_to_node[node_idx]:rank for rank,node_idx in enumerate(mean_expression_sort)}
# Calculate a ranking of the genes by their variance
var_expression = np.var(tgt_data.X.toarray(), axis=0).flatten()
var_expression_rank = np.argsort(var_expression)[::-1]
var_expression_rank = {idx_to_node[node_idx]:rank for rank,node_idx in enumerate(var_expression_rank)}
tgt_graph = pickle.load(open(f'{tgt_outdir}/optimal_{target_genotype}_graph.pickle', 'rb'))
src_graph = pickle.load(open(f'{src_outdir}/optimal_{source_genotype}_graph.pickle', 'rb'))

#%%
headers = ['Gene', 'Cell Proportion Dist', 'WT Num Targets Rank', 'WT Mean Rank',  'WT Var Rank', 'Jaccard Index of Targets']
rows = []
diffs = []
sorted_distances = sorted(individual_distances.items(), key=lambda x: x[1].sum())
for i, (transfer_gene, distances) in enumerate(sorted_distances):
    transfer_gene = transfer_gene
    # Get the outgoing edges from the transfer gene from the networkx graph
    out = set(src_graph.out_edges(transfer_gene))
    diffs.append(distances)

    transfer_names = protein_id_name[transfer_gene]

    rows.append((f'{transfer_names}',
                 f'{distances.sum():.3f}',
    ))
                #  f'{var_expression_rank[transfer_idx]:5d}'))
print(tabulate(rows, headers=headers))

#%%
best_gene_combination = pickle.load(open(f'{datadir}/top_transfer_combination.pickle', 'rb'))
#%%
best_subgraph_tgt = tgt_graph.subgraph(best_gene_combination)
best_subgraph_src = src_graph.subgraph(best_gene_combination)

# Get the connected components in the both the source and target graphs
tgt_connected_components = list(nx.strongly_connected_components(best_subgraph_tgt))
src_connected_components = list(nx.strongly_connected_components(best_subgraph_src))

# Print the number of connected components in the source and target graphs
print(f'Number of connected components in target graph: {len(tgt_connected_components)}')
print(f'Number of connected components in source graph: {len(src_connected_components)}')

# For each pair of genes in the best combination, check if they are reachable 
# from each other in the source and target graphs
for gene1, gene2 in combinations(best_gene_combination, 2):
    if gene1 in src_graph and gene2 in src_graph and gene1 in tgt_graph and gene2 in tgt_graph:
        tgt_connected = nx.has_path(best_subgraph_tgt, gene1, gene2)
        src_connected = nx.has_path(best_subgraph_src, gene1, gene2)
        if tgt_connected != src_connected:
            print(f'{protein_id_name[gene1]} and {protein_id_name[gene2]} are connected in the target graph: {tgt_connected}')
            print(f'{protein_id_name[gene1]} and {protein_id_name[gene2]} are connected in the source graph: {src_connected}')
            print('-'*30)

#%%
def print_components(subgraph):
    for connected_component in list(nx.strongly_connected_components(subgraph)):
        targets = set()
        for node in sorted(connected_component):
            out_edges = subgraph.out_edges(node)
            print(protein_id_name[node], len(out_edges))
            for src, dst in out_edges:
                targets.add(dst)
        print('=---')
print('--------------------------')
print('Components in target graph')
print('--------------------------')
print_components(best_subgraph_tgt)
print('--------------------------')
print('Components in source graph')
print('--------------------------')
print_components(best_subgraph_src)
#%%
# Print the rank of the genes in the best combination, with their rank
# in the individual transfer simulations. Also print their mean and variance ranks
headers = ['Gene', 'Cell Proportion Dist', 'Mut-WT Single Transfer Rank', 'WT Mean Rank',  'WT Var Rank', 'Number of Targets Rank']
rows = []
individual_transfer_rank = {gene: i for i, (gene, distances) in enumerate(sorted_distances)}
num_targets = {gene: len(src_graph.out_edges(gene)) for gene in all_genes}
sorted_by_targets = sorted(all_genes, key=lambda x: num_targets[x], reverse=True)
num_targets_rank = {gene: i for i, gene in enumerate(sorted_by_targets)}
for i, transfer_gene in enumerate(best_gene_combination):
    # Get the outgoing edges from the transfer gene from the networkx graph
    targets_rank = num_targets_rank[transfer_gene]
    # Get the distance of the transfer gene from the baseline
    distances = individual_distances[transfer_gene]

    transfer_names = protein_id_name[transfer_gene]

    rows.append((f'{transfer_names}',
                 f'{distances.sum():.3f}',
                 f'{individual_transfer_rank[transfer_gene]:5d}',
                 f'{mean_expression_rank[transfer_gene]:5d}',
                 f'{var_expression_rank[transfer_gene]:5d}',
                 f'{targets_rank:5d}'))
print(tabulate(rows, headers=headers))

#%%
# Make a heatmap of how the cell types are affected by the transfers
cell_types = sorted(tgt_data.obs['cell_type'].cat.categories)
all_diffs = np.zeros((len(all_genes), len(cell_types)))
transfer_gene_names = [protein_id_name[transfer_gene] for transfer_gene in individual_proportions]
for i, (transfer_gene, distances) in enumerate(sorted_distances):
    all_diffs[i] = distances
n_genes = 20

plt.figure(figsize=(5, 7))
plt.imshow(all_diffs[:n_genes], interpolation='none', aspect='auto', cmap='bwr')
plt.xticks(np.arange(len(cell_types)), cell_types, rotation=90)
plt.yticks(np.arange(n_genes), transfer_gene_names[:n_genes])
# Put xticks at the top too
ax = plt.gca()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
# plt.colorbar()
# Add more ticks to the colorbar
cbar = plt.colorbar(shrink=.5)
# Get the colorbar as an axis object
cbar_ax = plt.gcf().axes[-1]
ticks = np.linspace(all_diffs.min(), all_diffs.max(), 9)
ticklabels = [f'{tick:.3f}' for tick in ticks]
cbar_ax.set_yticks(ticks, ticklabels);
# Label the colorbar
cbar.set_label('\nCell Type Proportion Difference', fontsize=12)
# plt.tight_layout()

#%%
# Plot the individual transfers
plt.figure(figsize=(5, 7))
ds = np.array([distances.sum() for gene, distances in sorted_distances])
# The errors are standard deviations, so we aggregate them across the cell types
# with the following eqn: sqrt(sum_i (std_i)^2)
errors = [np.sqrt((error**2).sum()) for gene, error in individual_errors.items()]
plt.axhline(baseline_distance, color='black', linestyle='--')
baseline_error = np.sqrt((baseline_error**2).sum())
plt.axhline(baseline_distance + baseline_error, color='green', linestyle='-', alpha=.3)
plt.axhline(baseline_distance - baseline_error, color='green', linestyle='-', alpha=.3)
x = np.arange(len(sorted_distances))
plt.errorbar(x, ds, yerr=errors, fmt='none', ecolor='black', capsize=2, alpha=.2)
plt.scatter(x, ds, s=10)
baseline_low = baseline_distance - baseline_error
baseline_high = baseline_distance + baseline_error
# # If the error bars don't overlap with the baseline, plot them in red
# for i in range(len(sorted_distances)):
#     if ds[i] - errors[i] > baseline_high or ds[i] + errors[i] < baseline_low:
#         plt.scatter(x[i], ds[i], color='red', s=10)
#         # Label the significant genes
#         plt.text(x[i]+3, ds[i], protein_id_name[sorted_distances[i][0]], fontsize=8, rotation=0)
best_combo_ds = np.array([individual_distances[gene].sum() for gene in best_gene_combination])
n_above = (best_combo_ds > baseline_distance).sum()
n_below = (best_combo_ds < baseline_distance).sum()

# TODO this works alright but I think the ideal strategy would be to split the 
# space in the plot into vertical and horizontal non-overlapping chunks with enough space for each
# label, then arrange the labels into the chunks. We would have to change to using data offsets instead of 
# fontsize offsets, and we would have to calculate the number of chunks and the size of each chunk
# Generate a bounding box the dimensions of the plot (xlim, ylim), then split it into chunks
# Each chunk would then be given sequential non-overlapping rectangles to place the labels in

above = []
below = []
for i,gene in enumerate(sorted(best_gene_combination, key=lambda x: individual_transfer_rank[x])):
    gene_i = sorted_distances.index((gene, individual_distances[gene]))
    is_above = ds[gene_i] > baseline_distance
    if is_above:
        above.append((gene_i, gene))
    else:
        below.append((gene_i, gene))
plt_height = plt.ylim()[1] - plt.ylim()[0]
plt_height *= .9
plt_width = plt.xlim()[1] - plt.xlim()[0]
slot_height = plt_height / len(best_gene_combination)

def add_labels(slots, base_height, above=True):
    h = base_height 
    if above:
        relpos = (1,0)
        xbuf = 0
    else:
        relpos = (0,0)
        xbuf = plt_width * .05
    print(h)
    
    for i in range(len(slots)-1):
        gene_idx, gene_name = slots[i]
        next_gene_idx  = slots[i+1][0]
        diff = np.abs(ds[gene_idx] - ds[next_gene_idx])
        if diff < slot_height:
            h += slot_height * np.sign(diff)
        else:
            h = ds[gene_idx] 
        ytext = h
        xtext = x[gene_idx] + xbuf

        print(xtext, ytext, diff, slot_height, h)
        plt.annotate(text=protein_id_name[gene_name], 
             xy=(x[gene_idx], ds[gene_idx]), xycoords='data',
             xytext=(xtext, ytext), textcoords='data',
             arrowprops=dict(arrowstyle='simple, head_width=0.2, tail_width=0.05', relpos=relpos),
             fontsize=8, rotation=0)

add_labels(above, baseline_distance, above=True)
add_labels(below, baseline_distance, above=False)


plt.xlabel('Transfer Gene')
plt.xticks([])
plt.ylabel('Cell Type Proportion Difference')
# TODO 
# %%