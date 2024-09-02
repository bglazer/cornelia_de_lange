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
import plotting
import random
import matplotlib
import glob
from collections import Counter
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
for i in range(0, len(all_genes)):
    result = pickle.load(open(f'{datadir}/individual_combination_{i}_{transfer}_transfer_cell_type_proportions.pickle', 'rb'))
    gene = tuple(result['transfer_genes'])[0]
    proportions = result['perturb_proportions']
    error = result['perturb_errors']
    individual_proportions[gene] = proportions
    individual_errors[gene] = error
    individual_distances = {gene: np.abs(proportions - src_baseline_cell_proportions) 
                            for gene, proportions in individual_proportions.items()}
sorted_distances = sorted(individual_distances.items(), key=lambda x: x[1].sum())
individual_transfer_rank = {gene: i for i, (gene, distances) in enumerate(sorted_distances)}

#%%
# Load the baseline transfer (null, no transfer) simulation results
baseline_result = pickle.load(open(f'{datadir}/baseline_combination_0_{transfer}_transfer_cell_type_proportions.pickle', 'rb'))
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
#%%
best_gene_combinations = []
# TODO use the label when we want to select only the VIM first transfers
label = ''
for file in glob.glob(f'{datadir}/top_{label}{transfer}_combination*.pickle'):
    combo = pickle.load(open(file, 'rb'))
    best_gene_combinations.append(combo)

combo_gene_counts = Counter()
for combo in best_gene_combinations:
    combo_gene_counts.update(combo)

# Simulate selecting random combinations of genes of the same size as the best combinations
n_repeats = 10_000
random_occurences = {gene: np.zeros(n_repeats) for gene in all_genes}
for i in range(n_repeats):
    random_combo_gene_counts = Counter()
    for combo in best_gene_combinations:
        random_combo = random.sample(list(all_genes), len(combo))
        random_combo_gene_counts.update(random_combo)
    for gene, count in random_combo_gene_counts.items():
        random_occurences[gene][i] = count

p_gene_counts = {}
for gene, count in combo_gene_counts.most_common():
    p = (random_occurences[gene] > count).sum()/n_repeats
    p_gene_counts[gene] = p
    print(f'{count:2d}/{len(best_gene_combinations):2d} {p:.4f} {protein_id_name[gene]}')


all_core_genes = list(combo_gene_counts.keys())

#%%
p_components = []
for combo in best_gene_combinations:
    # Compute the probability of finding N randomly selected nodes in a strongly connected component
    combo_components = nx.strongly_connected_components(src_graph.subgraph(combo))
    combo_max_component = max(combo_components, key=lambda x: len(x))
    n_repeats = 10000
    big_component_sizes = np.zeros(n_repeats)

    for i in range(n_repeats):
        nodes = np.random.choice(list(src_graph.nodes), len(combo), replace=False)
        components = nx.strongly_connected_components(src_graph.subgraph(nodes))
        big_component_sizes[i] = max([len(c) for c in components])

    p_larger_component = (big_component_sizes > len(combo_max_component)).sum() / n_repeats
    p_components.append(p_larger_component)
    print(f'Probability of finding a larger component by chance: {p_larger_component}', flush=True)
    print(f'Nodes in largest component / Nodes selected: {len(combo_max_component)}/{len(combo)}, {len(combo_max_component)/len(combo)}', flush=True)
    print('-')
#%%
p_components = np.array(p_components)
print(f'p-values less than .05: {(p_components < .05).sum()} out of {len(p_components)}')
#%%
combined_components = nx.DiGraph()
combo_components = []
for combo in best_gene_combinations:
    components = nx.strongly_connected_components(src_graph.subgraph(combo))
    largest_component_genes = max(components, key=lambda x: len(x))
    combo_components.append(largest_component_genes)
    largest_component = src_graph.subgraph(largest_component_genes)
    assert(len(largest_component_genes) == len(largest_component))
    for edge in largest_component.edges:
        if edge not in combined_components.edges:
            combined_components.add_edge(*edge, count=1)
        else:
            combined_components[edge[0]][edge[1]]['count'] += 1

    # combined_components.add_edges_from(largest_component.edges)
    combined_components.add_nodes_from(largest_component.nodes)

assert(nx.is_strongly_connected(combined_components))

print(f'Number of nodes in combined components: {len(combined_components)}')
print(f'Number of edges in combined components: {len(combined_components.edges)}')
#%%
# Save the combined components
pickle.dump(combined_components, open(f'{datadir}/core_circuit.pickle', 'wb'))
#%%
core_subgraph_genes = list(combined_components.nodes)
#%%
# Frequency of core genes in the best combinations
core_gene_counts = Counter()
for gene in combined_components.nodes:
    for combo in best_gene_combinations:
        if gene in combo:
            core_gene_counts[gene] += 1
for i,(gene, count) in enumerate(core_gene_counts.most_common()):
    print(f'{i} {count}/{len(best_gene_combinations)} {protein_id_name[gene]}')


#%%
# Draw the combined components
layout = nx.kamada_kawai_layout(combined_components)

node_labels = {node:protein_id_name[node] for node in combined_components.nodes}
# Color nodes by their connectivity
colormap = matplotlib.cm.viridis
# pagerank = nx.pagerank(combined_components)=
centrality = np.array([combo_gene_counts[node] for node in combined_components.nodes])
centrality_scaled = (centrality / centrality.max())
node_colors = [colormap(centrality_scaled[i]) for i in range(len(combined_components.nodes))]
# node_colors = ['yellow' if centrality[i] else 'blue' for i in range(len(combined_components.nodes))]
node_scale = np.array([core_gene_counts[gene]/len(core_gene_counts)
                       for gene in combined_components.nodes])
node_sizes = (node_scale * 1500)
edge_alphas = np.array([combined_components[u][v]['count'] for u,v in combined_components.edges])
edge_alphas = (edge_alphas) / (edge_alphas.max())
edge_colors = []
for alpha in edge_alphas:
    edge_colors.append([.4,.4,.4,alpha]) #colormap(alpha)[:3] + (alpha,))
# Increase the dpi of the figure so the text is not blurry
fig = plt.figure(dpi=300)
no_self_loops = combined_components.copy()
no_self_loops.remove_edges_from(nx.selfloop_edges(no_self_loops))
nx.draw(no_self_loops, pos=layout,
        node_size=centrality+100,
        # edge_color=[(0,0,0,alpha) for alpha in edge_alphas],
        edge_color=edge_colors,
        # Change the node border color to blue
        node_color=node_colors,
        # alpha=node_scale+0.5,
        with_labels=False,
        )

plt_height = plt.ylim()[1] - plt.ylim()[0]
plt_width = plt.xlim()[1] - plt.xlim()[0]
# Get figure size in inches and dpi
fig_width_in = fig.get_figwidth()
fig_height_in = fig.get_figheight()
dpi = fig.get_dpi()
# Convert node size from points**2 to the fig data scale
node_size_in = np.sqrt(node_sizes) / dpi
# Convert node size from fig data scale to axes data scale
node_size_ax = node_size_in*2 * plt_height / fig_height_in

for i in range(len(combined_components.nodes)):
    gene = list(combined_components.nodes)[i]
    # Check if the gene was selected a statistically significant number of times
    if gene in p_gene_counts and p_gene_counts[gene] < .001:
        gene_name = protein_id_name[gene]+'*'
    else:
        gene_name = protein_id_name[gene]
    plt.text(layout[gene][0], layout[gene][1]-node_size_ax[i], gene_name, 
             horizontalalignment='center', verticalalignment='top',
             fontsize=8, color='black',
             bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.1'))
plt.title('Subgraph of Mutant Transfer Genes')
plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=centrality_scaled.min()*.95, 
                                                                           vmax=centrality_scaled.max()), 
                                          cmap=colormap),
             shrink=.5, 
             ticks=[tick.round(2) for tick in np.linspace(centrality_scaled.min(), centrality_scaled.max(), 6)],
             label='Proportion of transfer sets')

#%%
# Which clusters in the regulatory graph are the transfer genes in?
# graph = pickle.load(open(f'../../data/filtered_graph.pickle', 'rb'))
cluster_assignments = pickle.load(open('../../data/louvain_clusters.pickle', 'rb'))

cluster_matches = np.zeros(len(cluster_assignments), dtype=int)
for i,cluster in enumerate(cluster_assignments):
    cluster_genes = set(cluster_assignments[i])
    transfer_genes = set(core_subgraph_genes)
    intersection = cluster_genes.intersection(transfer_genes)
    if len(intersection) > 0:
        cluster_matches[i] += len(intersection)
        print(f'Cluster {i} has {len(intersection)} transfer genes')
        for gene in intersection:
            print(protein_id_name[gene])
        print('-'*30)
#%%
print(f'{(cluster_matches>0).sum()} out of {len(cluster_assignments)} clusters have transfer genes')

#%%
# Get a random set of len(best_gene_combination) genes
# and test how many clusters they are in
n_repeats = 10000
n_clusters = len(cluster_assignments)
n_cluster_matches = np.zeros((n_repeats, n_clusters), dtype=int)
for i in range(n_repeats):
    random_genes = random.sample(list(all_genes), len(core_subgraph_genes))
    for j,cluster in enumerate(cluster_assignments):
        cluster_genes = set(cluster_assignments[j])
        transfer_genes = set(random_genes)
        intersection = cluster_genes.intersection(transfer_genes)
        if len(intersection) > 0:
            n_cluster_matches[i][j] += len(intersection)
p_cluster_matches = (n_cluster_matches >= cluster_matches).sum(axis=0) / n_repeats
print(f'Probability of finding more cluster matches by chance: ')
for i,p in enumerate(p_cluster_matches):
    print(f'Cluster {i:3d}: {cluster_matches[i]} {p:.3f} {"*" if p < .05 else ""}')
#%%
# Find the probability of exceeding the overall number of cluster matches
p_cluster_matches = ((n_cluster_matches>0).sum(axis=1) > (cluster_matches>0).sum()).sum() / n_repeats
print(f'Probability of finding {(cluster_matches>0).sum()} cluster matches with {len(core_subgraph_genes)} genes by chance: ', p_cluster_matches)
#%%
# How many genes do the core set regulate?
regulated_genes = set()
for gene in core_subgraph_genes:
    for src, dst in src_graph.out_edges(gene):
        regulated_genes.add(dst)
print(f'{len(regulated_genes)} out of {len(all_genes)} genes are regulated by the core set')

#%%
# Print the rank of the genes in the best combination, with their rank
# in the individual transfer simulations. Also print their mean and variance ranks
headers = ['Gene', 'Cell Proportion Dist', 'Mut-WT Single Transfer Rank', 'WT Mean Rank',  'WT Var Rank', 'Number of Targets Rank']
rows = []
num_targets = {gene: len(src_graph.out_edges(gene)) for gene in all_genes}
sorted_by_targets = sorted(all_genes, key=lambda x: num_targets[x], reverse=True)
num_targets_rank = {gene: i for i, gene in enumerate(sorted_by_targets)}
for i, transfer_gene in enumerate(core_subgraph_genes):
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
sorted_gene_names = [protein_id_name[gene] for gene, distances in sorted_distances]
for i, (transfer_gene, distances) in enumerate(sorted_distances):
    all_diffs[i] = distances
n_genes = 10

fig, axs = plt.subplots(2,1, figsize=(5, 8))
ax0 = axs[0].imshow(all_diffs[:n_genes], interpolation='none', aspect='auto', cmap='Blues', vmin=all_diffs.min(), vmax=all_diffs.max())
axs[0].set_xticks(np.arange(len(cell_types)), cell_types, rotation=90)
axs[0].set_yticks(np.arange(n_genes), sorted_gene_names[:n_genes])
# Put xticks at the top too
axs[0].xaxis.tick_top()
axs[0].xaxis.set_label_position('top')
axs[0].set_ylabel('Transfer Gene (Top 10)')

ax1 = axs[1].imshow(all_diffs[-n_genes:], interpolation='none', aspect='auto', cmap='Blues', vmin=all_diffs.min(), vmax=all_diffs.max())
axs[1].set_xticks(np.arange(len(cell_types)), cell_types, rotation=90)
axs[1].set_yticks(np.arange(n_genes), sorted_gene_names[-n_genes:])
# Put xticks at the top too
# axs[1].xaxis.tick_top()
# axs[1].xaxis.set_label_position('top')
axs[1].set_xlabel('Cell Type')
axs[1].set_ylabel('Transfer Gene (Bottom 10)')
cb0 = plt.colorbar(ax0, ax=axs[0])
cb1 = plt.colorbar(ax1, ax=axs[1])
plt.tight_layout()


# Get the colorbar as an axis object
# ticks = np.linspace(all_diffs.min(), all_diffs.max(), 9)
# ticklabels = [f'{tick:.3f}' for tick in ticks]
# cb0.set_ticks(ticks)#, ticklabels);
# cb1.set_ticks(ticks)#, ticklabels);
# Label the colorbar
cb0.set_label('\nPercentage point difference from mutant', fontsize=12)
# plt.tight_layout()


#%%
def sequential_labeled_points(best_gene_combination, individual_transfers, baseline):
    # Plot the individual transfers
    plt.figure(figsize=(5, 7))
    baseline_distance, baseline_error = baseline
    individual_distances, individual_errors = individual_transfers
    sorted_distances = sorted(individual_distances.items(), key=lambda x: x[1].sum())
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
    slot_height = (plt_height / len(best_gene_combination))/2 * 1.25

    def add_half_labels(genes, base_height, above=True, fontsize=8):
        if above:
            relpos = (1,0)
            alignment = 'right'
            xbuf = -plt_width * .05
            ybuf = plt_height* .05
        else:
            genes.reverse() 
            relpos = (0,0)
            alignment = 'left'
            xbuf = plt_width * .05
            ybuf = -plt_height* .05

        h = base_height + ybuf
        last_d = base_height
        
        for i in range(len(genes)):
            gene_idx, gene_name = genes[i]
            diff = ds[gene_idx] - last_d
            last_d = ds[gene_idx]
            # If the difference is greater than the slot height, 
            # and the point is farther from the baseline than the previous point
            # then we can put the label directly on the point
            solo_point = np.abs(diff) > slot_height
            farther_than_previous = np.abs(h - base_height) < np.abs(ds[gene_idx] - base_height)
            if solo_point and farther_than_previous:
                h = ds[gene_idx]
            # Otherwise just put the label in the next slot
            else:
                h += slot_height*np.sign(diff)
            ytext = h
            xtext = x[gene_idx] + xbuf

            plt.annotate(text=protein_id_name[gene_name], 
                         xy=(x[gene_idx], ds[gene_idx]), xycoords='data',
                         xytext=(xtext, ytext), textcoords='data',
                         arrowprops=dict(arrowstyle='simple, head_width=0.2, tail_width=0.05', relpos=relpos),
                         fontsize=fontsize, rotation=0, horizontalalignment=alignment)

    add_half_labels(above, baseline_distance, above=True, fontsize=8)
    add_half_labels(below, baseline_distance, above=False, fontsize=8)
#%%
sorted_transfers = sorted(individual_distances.items(), key=lambda x: x[1].sum())
up_transfers = [g for g,d in sorted_transfers][:5]
down_transfers = [g for g,d in sorted_transfers][-5:]

sequential_labeled_points(up_transfers + down_transfers,
                          (individual_distances, individual_errors), 
                          (baseline_distance, baseline_error))

plt.xlabel('Transfer Gene')
plt.xticks([])
plt.title('Individual transfer difference from mutant, sum of cell types')
plt.ylabel('Cell Proportions, percentage point difference')


#%%
sequential_labeled_points(core_subgraph_genes, 
                          (individual_distances, individual_errors), 
                          (baseline_distance, baseline_error))

plt.xlabel('Transfer Gene')
plt.xticks([])
plt.title('Individual transfer difference from mutant, best combination')
plt.ylabel('Cell Proportions, percentage point difference')

# %%
