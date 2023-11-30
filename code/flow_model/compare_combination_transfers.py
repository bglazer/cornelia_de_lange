#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import util
from tabulate import tabulate
import scanpy as sc
from itertools import combinations

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
transfer_dir = f'{tgt_outdir}/{transfer}_targeted_transfer_simulations'
pltdir = f'{tgt_outdir}/{transfer}_targeted_transfer_simulations/figures'
datadir = f'{tgt_outdir}/{transfer}_targeted_transfer_simulations/data'

node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}
#%%
proportion_distance_individual_genes = pickle.load(open(f'{tgt_outdir}/{transfer}_transfer_simulations/data/proportion_distance.pickle', 'rb'))

#%%
n_genes = 10
transfer_genes = [gene for gene,distance in proportion_distance_individual_genes[:n_genes]]
transfer_gene_names = [protein_id_name[gene] for gene in transfer_genes]

all_combos = []
for i in range(2, len(transfer_genes)+1):
    combos = list(combinations(transfer_genes, r=i))
    all_combos.extend(combos)
#%%
cell_type_trajectories = {}
cell_type_proportions = {}
mean_trajectories = {}

for i, genes in enumerate(all_combos):
    combo_ids = tuple(genes)
    combo_names = "_".join([protein_id_name[gene] for gene in genes])
    with open(f'{transfer_dir}/data/{combo_names}_{transfer}_transfer_cell_type_proportions.pickle', 'rb') as f:
        cell_type_proportions[combo_ids] = pickle.load(f)

    with open(f'{transfer_dir}/data/{combo_names}_{transfer}_transfer_cell_type_trajectories.pickle', 'rb') as f:
        cell_type_trajectories[combo_ids] = pickle.load(f)

    with open(f'{transfer_dir}/data/{combo_names}_{transfer}_transfer_mean_trajectories.pickle', 'rb') as f:
        mean_trajectories[combo_ids] = pickle.load(f)
#%%
combo_ids = []
for i, genes in enumerate(all_combos):
    combo_ids.append(tuple(genes))
# %%
# Sort the genes by the distance between the wildtype and mutant cell type distributions
# First get the distance between the wildtype and mutant cell type distributions
proportion_distance = {}
for ids in combo_ids:
    perturb_proportions, perturb_errors, baseline_proportions, baseline_errors = cell_type_proportions[ids]
    proportion_distance[ids] = np.sum(np.abs(perturb_proportions - baseline_proportions))
proportion_distance = sorted(proportion_distance.items(), key=lambda x: abs(x[1]), reverse=False)
# %%
graph = pickle.load(open(f'{tgt_outdir}/optimal_{target_genotype}_graph.pickle', 'rb'))
#%%
# Calculate a ranking of the genes by their mean expression
mean_expression = np.array(tgt_data.X.mean(axis=0)).flatten()
mean_expression_sort = np.argsort(mean_expression)[::-1]
mean_expression_rank = {idx_to_node[node_idx]:rank for rank,node_idx in enumerate(mean_expression_sort)}
# Calculate a ranking of the genes by their variance
var_expression = np.var(tgt_data.X.toarray(), axis=0).flatten()
var_expression_rank = np.argsort(var_expression)[::-1]
#%%
headers = ['Gene', 'Cell Proportion Dist', 'WT Num Targets Rank', 'WT Mean Rank',  'WT Var Rank', 'Jaccard Index of Targets']
rows = []
diffs = []
for i, (transfer_genes, distance) in enumerate(proportion_distance):
    # Get the outgoing edges from the transfer gene from the networkx graph
    out = set(graph.out_edges(transfer_genes))
    diffs.append(distance)

    transfer_names = [protein_id_name[gene] for gene in transfer_genes]

    perturb_proportions, perturb_error, baseline, baseline_error = cell_type_proportions[transfer_genes]
    dist = np.sum(np.abs(perturb_proportions - baseline))

    rows.append((f'{transfer_names}',
                 f'{distance:.3f}',
    ))
                #  f'{var_expression_rank[transfer_idx]:5d}'))
print(tabulate(rows, headers=headers))

#%%
# Make a heatmap of how the cell types are affected by the transfers
cell_types = sorted(tgt_data.obs['cell_type'].cat.categories)
all_diffs = np.zeros((len(all_genes), len(cell_types)))
n_genes = 20
for i, (transfer_gene, distance) in enumerate(proportion_distance):
    perturb_proportions, baseline = cell_type_proportions[transfer_gene]
    all_diffs[i] = perturb_proportions - baseline

plt.figure(figsize=(5, 7))
plt.imshow(all_diffs[:n_genes], interpolation='none', aspect='auto', cmap='bwr')
plt.xticks(np.arange(len(cell_types)), cell_types, rotation=90)
plt.yticks(np.arange(n_genes), [protein_id_name[transfer_gene] for transfer_gene, distance in proportion_distance[:n_genes]])
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
# Calculate the jaccard index of the targets of the WT and mutant models
jaccards = []
tgt_graph = pickle.load(open(f'{tgt_outdir}/optimal_{target_genotype}_graph.pickle', 'rb'))
src_graph = pickle.load(open(f'{src_outdir}/optimal_{source_genotype}_graph.pickle', 'rb'))

for transfer_gene in all_genes:
    tgt_targets = set(tgt_graph.out_edges(transfer_gene))
    src_targets = set(src_graph.out_edges(transfer_gene))
    if len(tgt_targets | src_targets) == 0:
        jaccards.append(0)
    else:
        targets_jaccard_idx = len(tgt_targets & src_targets)/len(tgt_targets | src_targets)
        jaccards.append(targets_jaccard_idx)
plt.scatter(jaccards, np.abs(diffs), marker='.', s=.8)
plt.xlabel('Jaccard Index of Targets')
plt.ylabel('Cell Type Proportion Distance')

# %%
# Get the gene expression distribution corresponding to each transfer
transfer_gene_expression = {}
for transfer_gene in all_genes:
    transfer_gene_expression[transfer_gene] = util.tonp(mean_trajectories[transfer_gene].mean(axis=0))
#%%
# Get the most differentially expressed genes corresponding to each transfer
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{tgt_outdir}/baseline_trajectories_{target_genotype}.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{tgt_outdir}/baseline_nearest_cell_idxs_{target_genotype}.pickle', 'rb'))
baseline_gene_expression_traj = baseline_trajectories.mean(axis=1)
baseline_gene_expression_total = baseline_trajectories.mean(axis=0).mean(axis=0)

#%%
print('Transfer gene differential expression')
for i, (transfer_gene, distance) in enumerate(proportion_distance[:30]):
    diff = transfer_gene_expression[transfer_gene] - baseline_gene_expression_total
    rows = []
    for idx in np.argsort(np.abs(diff))[::-1][:10]:
        rows.append((f'{protein_id_name[idx_to_node[idx]]:10s}',
                    f'{diff[idx]:.3f}'))
    title = f'Transfer: {protein_id_name[transfer_gene]}'
    print('-'*len(title))
    print(title)
    print('-'*len(title))
    print(tabulate(rows, headers=['Gene', 'Diff']))

#%%
#%%
print('Transfer gene differential expression heatmap')
from collections import Counter
all_top10_diff_genes = Counter()
for i, (transfer_gene, distance) in enumerate(proportion_distance[:n_genes]):
    # Use the pearson residual to find the most differentially expressed genes
    nonzero_genes = np.where(baseline_gene_expression_total > 0)[0]
    # diff = np.abs(transfer_gene_expression[transfer_gene] - baseline_gene_expression_total)/np.sqrt(baseline_gene_expression_total)
    diff = transfer_gene_expression[transfer_gene] - baseline_gene_expression_total
    for idx in np.argsort(np.abs(diff))[::-1][:10]:
        all_top10_diff_genes[idx_to_node[idx]] += 1

all_top10_diff_genes = all_top10_diff_genes.most_common(20)

gene_diffs = np.zeros((len(all_genes), len(all_top10_diff_genes)))
for i, (transfer_gene, distance) in enumerate(proportion_distance):
    diff = transfer_gene_expression[transfer_gene] - baseline_gene_expression_total
    for j, (gene,_) in enumerate(all_top10_diff_genes):
        gene_diffs[i, j] = diff[node_to_idx[gene]]

# Plot the heatmap
plt.figure(figsize=(12, 10))
# Center the colormap at 0
vmax = np.max(np.abs(gene_diffs))
vmin = -vmax
plt.imshow(gene_diffs[:n_genes], interpolation='none', aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
plt.xticks(np.arange(len(all_top10_diff_genes)), [protein_id_name[gene] for gene, count in all_top10_diff_genes], rotation=90)
plt.xlabel('Differentially Expressed Genes after Transfer', fontsize=15)
plt.yticks(np.arange(n_genes), [protein_id_name[transfer_gene] for transfer_gene, distance in proportion_distance[:n_genes]])
plt.ylabel('Transferred Gene', fontsize=15)
# Put xticks at the top too
ax = plt.gca()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.colorbar(shrink=.5)

# %%
# Find the single gene knockouts that have the largest effect on the cell type proportions
combined_proportion_distance = {}
for transfer_gene in all_genes:
    perturb_proportions, baseline = cell_type_proportions[transfer_gene]
combined_proportion_distance = sorted(combined_proportion_distance.items(), key=lambda x: abs(x[1]), reverse=True)
#%%
headers = ['Gene', 'Distance', 'WT Diff', 'WT Num Targets Rank']
rows = []
for i, (transfer_gene, distance) in enumerate(combined_proportion_distance):
    if transfer_gene in targets_rank:
        n_targets_rank = targets_rank[transfer_gene]
    else:
        n_targets_rank = len(all_genes)

    perturb_proportions, baseline = cell_type_proportions[transfer_gene]
    dist = np.sum(np.abs(perturb_proportions - baseline))

    rows.append((f'{protein_id_name[transfer_gene]:10s}',
                 f'{distance:.3f}',
                 f'{dist:.3f}',
                 f'{n_targets_rank:5d}'))
print(tabulate(rows, headers=headers))
#%%
# Genes that are most disregulated by the transfer across both wildtype and mutant
print('Transfer gene combined fold change differential expression')
for i, (transfer_gene, distance) in enumerate(combined_proportion_distance[:30]):
    diff = np.log((transfer_gene_expression[transfer_gene]+1)/(baseline_gene_expression_total+1))
    rows = []
    for idx in np.argsort(np.abs(diff))[::-1][:30]:
        rows.append((f'{protein_id_name[idx_to_node[idx]]:10s}',
                     f'{diff[idx]:.3f}',
                     f'{diff[idx]:.3f}'))
    title = f'Transfer: {protein_id_name[transfer_gene]}'
    print('-'*len(title))
    print(title)
    print('-'*len(title))
    print(tabulate(rows, headers=['Gene', 'WT Diff']))
#%%
print('Transfer gene combined differential expression')
for i, (transfer_gene, distance) in enumerate(combined_proportion_distance[:30]):
    diff = (transfer_gene_expression[transfer_gene])-(baseline_gene_expression_total)
    rows = []
    for idx in np.argsort(np.abs(diff))[::-1][:30]:
        rows.append((f'{protein_id_name[idx_to_node[idx]]:10s}',
                     f'{diff[idx]:.3f}',
                     f'{diff[idx]:.3f}'))
    title = f'Transfer: {protein_id_name[transfer_gene]}'
    print('-'*len(title))
    print(title)
    print('-'*len(title))
    print(tabulate(rows, headers=['Gene', 'WT Diff']))

# %%
