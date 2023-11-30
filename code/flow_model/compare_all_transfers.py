#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import util
from tabulate import tabulate
import scanpy as sc

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
transfer_dir = f'{tgt_outdir}/{transfer}_transfer_simulations'
pltdir = f'{tgt_outdir}/{transfer}_transfer_simulations/figures'
datadir = f'{tgt_outdir}/{transfer}_transfer_simulations/data'

node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}

#%%
cell_type_trajectories = {}
cell_type_proportions = {}
mean_trajectories = {}

for i, transfer_gene in enumerate(all_genes):
    transfer_gene_name = protein_id_name[transfer_gene]
    with open(f'{transfer_dir}/data/{transfer_gene_name}_{transfer}_transfer_cell_type_proportions.pickle', 'rb') as f:
        cell_type_proportions[transfer_gene] = pickle.load(f)

    with open(f'{transfer_dir}/data/{transfer_gene_name}_{transfer}_transfer_cell_type_trajectories.pickle', 'rb') as f:
        cell_type_trajectories[transfer_gene] = pickle.load(f)

    with open(f'{transfer_dir}/data/{transfer_gene_name}_{transfer}_transfer_mean_trajectories.pickle', 'rb') as f:
        mean_trajectories[transfer_gene] = pickle.load(f)
# %%
# Sort the genes by the distance between the wildtype and mutant cell type distributions
# First get the distance between the wildtype and mutant cell type distributions
proportion_distance = {}
for transfer_gene in all_genes:
    perturb_proportions, baseline = cell_type_proportions[transfer_gene]
    proportion_distance[transfer_gene] = np.sum(np.abs(perturb_proportions - baseline))
proportion_distance = sorted(proportion_distance.items(), key=lambda x: abs(x[1]), reverse=True)
pickle.dump(proportion_distance, open(f'{transfer_dir}/data/proportion_distance.pickle', 'wb'))
# %%
graph = pickle.load(open(f'{tgt_outdir}/optimal_{target_genotype}_graph.pickle', 'rb'))
targets = {}
for target_gene in all_genes:
    if target_gene in graph:
        targets[target_gene] = graph.out_degree[target_gene]
#%%
targets_sorted = sorted(targets.items(), key=lambda x: x[1], reverse=True)
targets_rank = {gene: i for i, (gene, num_targets) in enumerate(targets_sorted)}

#%%
# Calculate a ranking of the genes by their mean expression
tgt_mean_expression = np.array(tgt_data.X.mean(axis=0)).flatten()
tgt_mean_expression_sort = np.argsort(tgt_mean_expression)[::-1]
tgt_mean_expression_rank = {idx_to_node[node_idx]:rank for rank,node_idx in enumerate(tgt_mean_expression_sort)}
# Calculate a ranking of the genes by their variance
tgt_var_expression = np.var(tgt_data.X.toarray(), axis=0).flatten()
tgt_var_expression_rank = np.argsort(tgt_var_expression)[::-1]

src_mean_expression = np.array(src_data.X.mean(axis=0)).flatten()
src_mean_expression_sort = np.argsort(src_mean_expression)[::-1]
src_mean_expression_rank = {idx_to_node[node_idx]:rank for rank,node_idx in enumerate(src_mean_expression_sort)}
# Calculate a ranking of the genes by their variance
src_var_expression = np.var(src_data.X.toarray(), axis=0).flatten()
src_var_expression_rank = np.argsort(src_var_expression)[::-1]
#%%
headers = ['Gene', 'Cell Proportion Dist', 'WT Num Targets Rank', 'WT Mean Rank',  
           'Mut Mean Rank', 'WT Var Rank', 'Mut Var Rank']
rows = []
diffs = []
for i, (transfer_gene, distance) in enumerate(proportion_distance):
    if transfer_gene in targets_rank:
        n_targets_rank = targets_rank[transfer_gene]
    else:
        n_targets_rank = len(all_genes)

    # Get the outgoing edges from the transfer gene from the networkx graph
    out = set(graph.out_edges(transfer_gene))
    diffs.append(distance)

    perturb_proportions, baseline = cell_type_proportions[transfer_gene]
    dist = np.sum(np.abs(perturb_proportions - baseline))

    tgt_variance_rank = int(np.argwhere(tgt_var_expression_rank == node_to_idx[transfer_gene]))
    src_variance_rank = int(np.argwhere(src_var_expression_rank == node_to_idx[transfer_gene]))

    rows.append((f'{protein_id_name[transfer_gene]:10s}',
                 f'{distance:.3f}',
                 f'{n_targets_rank:5d}', 
                 f'{tgt_mean_expression_rank[transfer_gene]:5d}',
                 f'{src_mean_expression_rank[transfer_gene]:5d}',
                 f'{tgt_variance_rank:5d}',
                 f'{src_variance_rank:5d}'))
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
# Transfer gene regulator changes
# Which inputs are present in the wildtype and mutant models?
for gene,_ in proportion_distance:
    print(f'{protein_id_name[gene]:10s}')
    tgt_inputs = set(tgt_graph.in_edges(gene))
    src_inputs = set(src_graph.in_edges(gene))
    print('Common:', [protein_id_name[g[0]] for g in tgt_inputs & src_inputs])
    print('WT only:', [protein_id_name[g[0]] for g in tgt_inputs - src_inputs])
    print('Mut only:', [protein_id_name[g[0]] for g in src_inputs - tgt_inputs])
# %%
# Looking at Pou5f1 specifically, are its regulators differentially expressed in the mutant?
# Get the regulators of Pou5f1 in the wildtype and mutant models
pou5f1 = protein_name_id['POU5F1']
tgt_pou5f1_regulators = set([g[0] for g in tgt_graph.in_edges(pou5f1)])
src_pou5f1_regulators = set([g[0] for g in src_graph.in_edges(pou5f1)])
# Get the expression of the regulators in the wildtype and mutant models
tgt_pou5f1_regulators_expression = np.array([tgt_mean_expression[node_to_idx[regulator]] for regulator in tgt_pou5f1_regulators])
src_pou5f1_regulators_expression = np.array([src_mean_expression[node_to_idx[regulator]] for regulator in src_pou5f1_regulators])
# Plot the expression of common regulators
common_regulators = list(tgt_pou5f1_regulators & src_pou5f1_regulators)
tgt_common_regulators_expression = np.array([tgt_mean_expression[node_to_idx[regulator]] for regulator in common_regulators])
src_common_regulators_expression = np.array([src_mean_expression[node_to_idx[regulator]] for regulator in common_regulators])
# Sort the common regulators by total expression
sorted_common_regulators = np.argsort(tgt_common_regulators_expression + src_common_regulators_expression) 
common_regulators_expression = np.vstack((tgt_common_regulators_expression, src_common_regulators_expression))
sorted_common_regulators_expression = common_regulators_expression[:,sorted_common_regulators]
plt.plot(sorted_common_regulators_expression.T)
# Label the x axis with the gene names
plt.xticks(np.arange(len(common_regulators)), [protein_id_name[regulator] for regulator in np.array(common_regulators)[sorted_common_regulators]], rotation=90)
# %%
