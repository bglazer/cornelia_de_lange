#%%
%load_ext autoreload
%autoreload 2
#%%
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import util
from tabulate import tabulate
import scanpy as sc
import plotting
from sklearn.decomposition import PCA

#%%
mut_tmstp = '20230608_093734'
wt_tmstp = '20230607_165324'

wt_outdir = f'../../output/{wt_tmstp}'
mut_outdir = f'../../output/{mut_tmstp}'
wt_ko_dir = f'{wt_outdir}/knockout_simulations'
mut_ko_dir = f'{mut_outdir}/knockout_simulations'

node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}
#%%
wt_data = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut_data = sc.read_h5ad(f'../../data/mutant_net.h5ad')
#%%
pca = PCA()
# Set the PC mean and components
pca.mean_ = wt_data.uns['pca_mean']
pca.components_ = wt_data.uns['PCs']
#%%
mut_cell_type_trajectories = {}
mut_cell_type_proportions = {}
mut_mean_trajectories = {}

for i,ko_gene in enumerate(all_genes):
    ko_gene_name = protein_id_name[ko_gene]
    with open(f'{mut_ko_dir}/data/{ko_gene_name}_mutant_knockout_cell_type_proportions.pickle', 'rb') as f:
        mut_cell_type_proportions[ko_gene] = pickle.load(f)

    with open(f'{mut_ko_dir}/data/{ko_gene_name}_mutant_knockout_cell_type_trajectories.pickle', 'rb') as f:
        mut_cell_type_trajectories[ko_gene] = pickle.load(f)

    with open(f'{mut_ko_dir}/data/{ko_gene_name}_mutant_knockout_mean_trajectories.pickle', 'rb') as f:
        mut_mean_trajectories[ko_gene] = pickle.load(f)

#%%
wt_cell_type_trajectories = {}
wt_cell_type_proportions = {}
wt_mean_trajectories = {}

for i,ko_gene in enumerate(all_genes):
    ko_gene_name = protein_id_name[ko_gene]
    with open(f'{wt_ko_dir}/data/{ko_gene_name}_wildtype_knockout_cell_type_proportions.pickle', 'rb') as f:
        wt_cell_type_proportions[ko_gene] = pickle.load(f)

    with open(f'{wt_ko_dir}/data/{ko_gene_name}_wildtype_knockout_cell_type_trajectories.pickle', 'rb') as f:
        wt_cell_type_trajectories[ko_gene] = pickle.load(f)

    with open(f'{wt_ko_dir}/data/{ko_gene_name}_wildtype_knockout_mean_trajectories.pickle', 'rb') as f:
        wt_mean_trajectories[ko_gene] = pickle.load(f)

#%%
wt_simulation_trajectories = pickle.load(open(f'{wt_outdir}/baseline_trajectories_wildtype.pickle', 'rb'))
mut_simulation_trajectories = pickle.load(open(f'{mut_outdir}/baseline_trajectories_mutant.pickle', 'rb'))
wt_simulation_nearest_cell_idxs = pickle.load(open(f'{wt_outdir}/baseline_nearest_cell_idxs_wildtype.pickle', 'rb'))
mut_simulation_nearest_cell_idxs = pickle.load(open(f'{mut_outdir}/baseline_nearest_cell_idxs_mutant.pickle', 'rb'))
#%%
# Compare cell type proportion change relative to data proportions with 
unique_cell_types = set(wt_data.obs['cell_type']) | set(mut_data.obs['cell_type'])
cell_types = {c:i for i,c in enumerate(unique_cell_types)}
sorted_cell_types = sorted(cell_types.keys())
cell_type_to_idx = {k:i for i,k in enumerate(sorted_cell_types)}
device = 'cuda:0'

# TODO this only computes the cell type proportion if the 
# cells in the baseline simulation just have the one gene knocked out
# It's maybe better to remove any cell that has any expression of the knocked out gene
# but that introduces some difficulty in computing the nearest cell in the data
# because we end up with a 1-d array of all the non-expressing cells, which could be 
# huge or very small depending on how many cells express the knockout gene
# We also can't compute the std of the cell type proportions because 
# the number of cells differs across each repeat, which numpy doesn't like
def ko_cell_type_proportion(ko_gene, baseline, data, cell_type_to_idx, pltdir):
    ko_X_np = baseline.copy()
    # ko_expressing_cells = ko_X_np[:,:,node_to_idx[ko_gene]] > 0
    # print(ko_expressing_cells.sum()/ko_expressing_cells.size)
    ko_X_np[:,:,node_to_idx[ko_gene]] = 0
    data_X = data.X.toarray()

    # Convert ko_X_np and data_X to tensors on the GPU
    ko_X = torch.from_numpy(ko_X_np).to(device)
    data_X = torch.from_numpy(data_X).to(device)

    # Find the nearest cell in the data for each cell in the simulation
    ko_nearest_idxs = torch.zeros((ko_X.shape[0],ko_X.shape[1]), dtype=torch.long, device=device)
    for i in range(ko_X.shape[0]):
        dist, idxs = torch.sort(torch.cdist(ko_X[i], data_X), dim=1)
        ko_nearest_idxs[i] = idxs[:,1]
    ko_nearest_idxs_np = ko_nearest_idxs.detach().cpu().numpy()

    # Replot the baseline simulation with the knockout gene knocked out and new nearest cells
    baseline = baseline.reshape((-1, baseline.shape[-1]))
    ko_gene_name = protein_id_name[ko_gene]
    print(ko_gene_name.upper())
    plotting.cell_type_distribution(ko_X_np, ko_nearest_idxs_np, data, cell_type_to_idx, pca, 
                                    label=f'Baseline with {ko_gene_name.upper()} knocked out', 
                                    baseline=baseline, s=1)
    plt.savefig(f'{pltdir}/baseline_ko_{ko_gene_name.upper()}.png', dpi=300)
    plt.close()
    
    ko_nearest_idxs = ko_nearest_idxs.detach().cpu().numpy()
    # Get the cell type of the nearest cell in the data for each cell in the simulation
    idx_to_cell_type = np.array(data.obs['cell_type'].values)
    ko_nearest_cell_types = idx_to_cell_type[ko_nearest_idxs.flatten()]
    # Convert cell type strings to indices
    ko_nearest_cell_type_idxs = np.array([cell_type_to_idx[c] for c in ko_nearest_cell_types], dtype=np.int32)
    # cell_type_means = np.bincount(ko_nearest_cell_type_idxs, minlength=len(cell_type_to_idx))
    # cell_type_means = cell_type_means/cell_type_means.sum()
    # Then compute the standard deviation of the cell type proportions
    # TODO 10 is the number of simulation repeats, should be a variable
    cell_type_means_repeats = np.zeros((10, len(cell_type_to_idx)))
    for i in range(10):
        cell_type_means_repeats[i] = np.bincount(ko_nearest_cell_type_idxs.reshape((-1, 10))[:,i], 
                                                 minlength=len(cell_type_to_idx))
    cell_type_means = cell_type_means_repeats.sum(axis=0)
    cell_type_means /= cell_type_means.sum()
    # cell_type_stds = (cell_type_means_repeats/cell_type_means_repeats.sum(axis=1)[:,None]).std(axis=0)
    return cell_type_means #, cell_type_stds

# ko_cell_type_proportion('ENSMUSP00000134654', wt_simulation_trajectories, 
#                         wt_data, cell_type_to_idx, pltdir=f'{wt_outdir}/baseline_knockouts')
#%%
# Calculate the baseline cell type proportions after knocking out each gene
wt_ko_baselines = {}
mut_ko_baselines = {}

# Make sure the pltdir exists
import os
os.makedirs(f'{wt_outdir}/baseline_knockouts', exist_ok=True)
os.makedirs(f'{mut_outdir}/baseline_knockouts', exist_ok=True)

for i,ko_gene in enumerate(all_genes):
    print(i)
    wt_ko_baseline = ko_cell_type_proportion(ko_gene, wt_simulation_trajectories, 
                                             wt_data, cell_type_to_idx, 
                                             pltdir=f'{wt_outdir}/baseline_knockouts')
    mut_ko_baseline = ko_cell_type_proportion(ko_gene, mut_simulation_trajectories,
                                              mut_data, cell_type_to_idx, 
                                              pltdir=f'{mut_outdir}/baseline_knockouts')
    wt_ko_baselines[ko_gene] = wt_ko_baseline
    mut_ko_baselines[ko_gene] = mut_ko_baseline
# Save the ko baselines
pickle.dump(wt_ko_baselines, open(f'{wt_outdir}/ko_baselines_wildtype.pickle', 'wb'))
pickle.dump(mut_ko_baselines, open(f'{mut_outdir}/ko_baselines_mutant.pickle', 'wb'))
#%%
wt_ko_baselines = pickle.load(open(f'{wt_outdir}/ko_baselines_wildtype.pickle', 'rb'))
mut_ko_baselines = pickle.load(open(f'{mut_outdir}/ko_baselines_mutant.pickle', 'rb'))
# %%
# Sort the genes by the distance between the wildtype and mutant cell type distributions
# First get the distance between the wildtype and mutant cell type distributions
proportion_distance = {}
for i,ko_gene in enumerate(all_genes):
    mut_perturb_proportions, mut_baseline = mut_cell_type_proportions[ko_gene]
    wt_perturb_proportions, wt_baseline = wt_cell_type_proportions[ko_gene]
    wt_ko_baseline = wt_ko_baselines[ko_gene]
    mut_ko_baseline = mut_ko_baselines[ko_gene]
    proportion_distance[ko_gene] = np.sum(np.abs(wt_perturb_proportions - wt_ko_baseline)) - np.sum(np.abs(mut_perturb_proportions - mut_ko_baseline))
proportion_distance = sorted(proportion_distance.items(), key=lambda x: abs(x[1]), reverse=True)

# %%
mut_graph = pickle.load(open(f'{mut_outdir}/optimal_mutant_graph.pickle', 'rb'))
wt_graph = pickle.load(open(f'{wt_outdir}/optimal_wildtype_graph.pickle', 'rb'))
mut_targets = {}
wt_targets = {}
for target_gene in all_genes:
    if target_gene in mut_graph:
        mut_targets[target_gene] = mut_graph.out_degree[target_gene] 
    if target_gene in wt_graph:
        wt_targets[target_gene] = wt_graph.out_degree[target_gene]
#%%
wt_targets_sorted = sorted(wt_targets.items(), key=lambda x: x[1], reverse=True)
wt_targets_rank = {gene: i for i, (gene, num_targets) in enumerate(wt_targets_sorted)}
mut_targets_sorted = sorted(mut_targets.items(), key=lambda x: x[1], reverse=True)
mut_targets_rank = {gene: i for i, (gene, num_targets) in enumerate(mut_targets_sorted)}

#%%
# Calculate a ranking of the genes by their mean expression
wt_mean_expression = np.array(wt_data.X.mean(axis=0)).flatten()
mut_mean_expression = np.array(mut_data.X.mean(axis=0)).flatten()
wt_mean_expression_rank = np.argsort(wt_mean_expression)[::-1]
mut_mean_expression_rank = np.argsort(mut_mean_expression)[::-1]
# Calculate a ranking of the genes by their variance
wt_var_expression = np.var(wt_data.X.toarray(), axis=0).flatten()
mut_var_expression = np.var(mut_data.X.toarray(), axis=0).flatten()
wt_var_expression_rank = np.argsort(wt_var_expression)[::-1]
mut_var_expression_rank = np.argsort(mut_var_expression)[::-1]
#%%
headers = ['Gene', 'Distance', 'WT Diff', 'Mut Diff', 'WT Num Targets Rank', 'Mut Num Targets Rank', 'WT Mean Rank', 'Mut Mean Rank', 'WT Var Rank', 'Mut Var Rank', 'Jaccard Index of Targets']
rows = []
jaccards = []
diffs = []
for i, (ko_gene, distance) in enumerate(proportion_distance.items()):
    if ko_gene in wt_targets_rank:
        wt_n_targets_rank = wt_targets_rank[ko_gene]
    else:
        wt_n_targets_rank = len(all_genes)
    if ko_gene in mut_targets_rank:
        mut_n_targets_rank = mut_targets_rank[ko_gene]
    else:
        mut_n_targets_rank = len(all_genes)

    # Get the outgoing edges from the knockout gene from the networkx graph
    wt_out = set(wt_graph.out_edges(ko_gene))
    mut_out = set(mut_graph.out_edges(ko_gene))
    if len(wt_out | mut_out) == 0:
        wt_mut_target_jaccard = 0
    else:
        wt_mut_target_jaccard = len(wt_out & mut_out) / len(wt_out | mut_out)
    jaccards.append(wt_mut_target_jaccard)
    diffs.append(distance)

    mut_perturb_proportions, mut_baseline = mut_cell_type_proportions[ko_gene]
    wt_perturb_proportions, wt_baseline = wt_cell_type_proportions[ko_gene]
    wt_ko_baseline = wt_ko_baselines[ko_gene]
    mut_ko_baseline = mut_ko_baselines[ko_gene]
    wt_dist = np.sum(np.abs(wt_perturb_proportions - wt_ko_baseline))
    mut_dist = np.sum(np.abs(mut_perturb_proportions - mut_ko_baseline))

    ko_idx = node_to_idx[ko_gene]
    rows.append((f'{protein_id_name[ko_gene]:10s}',
                 f'{distance:.3f}',
                 f'{wt_dist:.3f}',
                 f'{mut_dist:.3f}',
                 f'{wt_n_targets_rank:5d}', 
                 f'{mut_n_targets_rank:5d}',
                 f'{wt_mean_expression_rank[ko_idx]:5d}/{wt_mean_expression[ko_idx]:.3f}',
                 f'{mut_mean_expression_rank[ko_idx]:5d}/{mut_mean_expression[ko_idx]:.3f}',
                 f'{wt_var_expression_rank[ko_idx]:5d}',
                 f'{mut_var_expression_rank[ko_idx]:5d}',
                 f'{wt_mut_target_jaccard:.3f}'))
print(tabulate(rows, headers=headers))

#%%
plt.scatter(jaccards, np.abs(diffs), marker='.', s=.5)
plt.xlabel('Jaccard Index of Targets')
plt.ylabel('Distance')

# %%
# Get the gene expression distribution corresponding to each knockout
mut_ko_gene_expression = {}
wt_ko_gene_expression = {}
for ko_gene in all_genes:
    mut_ko_gene_expression[ko_gene] = util.tonp(mut_mean_trajectories[ko_gene].mean(axis=0))
    wt_ko_gene_expression[ko_gene] = util.tonp(wt_mean_trajectories[ko_gene].mean(axis=0))
#%%
# Get the most differentially expressed genes corresponding to each knockout
wt_baseline_trajectories = pickle.load(open(f'{wt_outdir}/baseline_trajectories_wildtype.pickle', 'rb'))
wt_baseline_gene_expression_traj = wt_baseline_trajectories.mean(axis=1)
wt_baseline_gene_expression_total = wt_baseline_trajectories.mean(axis=0).mean(axis=0)
mut_baseline_trajectories = pickle.load(open(f'{mut_outdir}/baseline_trajectories_mutant.pickle', 'rb'))
mut_baseline_gene_expression_traj = mut_baseline_trajectories.mean(axis=1)
mut_baseline_gene_expression_total = mut_baseline_trajectories.mean(axis=0).mean(axis=0)

#%%
print('Knockout gene differential expression')
for i, (ko_gene, distance) in enumerate(proportion_distance[:30]):
    wt_diff = wt_ko_gene_expression[ko_gene] - wt_baseline_gene_expression_total
    mut_diff = mut_ko_gene_expression[ko_gene] - mut_baseline_gene_expression_total
    diff = wt_diff - mut_diff
    rows = []
    for idx in np.argsort(np.abs(diff))[::-1][:10]:
        rows.append((f'{protein_id_name[idx_to_node[idx]]:10s}',
                    f'{wt_diff[idx]:.3f}',
                    f'{mut_diff[idx]:.3f}', 
                    f'{diff[idx]:.3f}'))
    title = f'Knockout: {protein_id_name[ko_gene]}'
    print('-'*len(title))
    print(title)
    print('-'*len(title))
    print(tabulate(rows, headers=['Gene', 'WT Diff', 'Mut Diff', 'Diff']))

# %%
# Find the single gene knockouts that have the largest effect on the cell type proportions
combined_proportion_distance = {}
for ko_gene in all_genes:
    mut_perturb_proportions, mut_baseline = mut_cell_type_proportions[ko_gene]
    wt_perturb_proportions, wt_baseline = wt_cell_type_proportions[ko_gene]
    combined_proportion_distance[ko_gene] = np.sum(np.abs(wt_perturb_proportions - wt_baseline)) + np.sum(np.abs(mut_perturb_proportions - mut_baseline))
combined_proportion_distance = sorted(combined_proportion_distance.items(), key=lambda x: abs(x[1]), reverse=True)
#%%
headers = ['Gene', 'Distance', 'WT Diff', 'Mut Diff', 'WT Num Targets Rank', 'Mut Num Targets Rank']
rows = []
for i, (ko_gene, distance) in enumerate(combined_proportion_distance):
    if ko_gene in wt_targets_rank:
        wt_n_targets_rank = wt_targets_rank[ko_gene]
    else:
        wt_n_targets_rank = len(all_genes)
    if ko_gene in mut_targets_rank:
        mut_n_targets_rank = mut_targets_rank[ko_gene]
    else:
        mut_n_targets_rank = len(all_genes)

    mut_perturb_proportions, mut_baseline = mut_cell_type_proportions[ko_gene]
    wt_perturb_proportions, wt_baseline = wt_cell_type_proportions[ko_gene]
    wt_dist = np.sum(np.abs(wt_perturb_proportions - wt_baseline))
    mut_dist = np.sum(np.abs(mut_perturb_proportions - mut_baseline))

    rows.append((f'{protein_id_name[ko_gene]:10s}',
                 f'{distance:.3f}',
                 f'{wt_dist:.3f}',
                 f'{mut_dist:.3f}',
                 f'{wt_n_targets_rank:5d}', 
                 f'{mut_n_targets_rank:5d}'))
print(tabulate(rows, headers=headers))
#%%
# Genes that are most disregulated by the knockout across both wildtype and mutant
print('Knockout gene combined fold change differential expression')
for i, (ko_gene, distance) in enumerate(combined_proportion_distance[:30]):
    wt_diff = np.log((wt_ko_gene_expression[ko_gene]+1)/(wt_baseline_gene_expression_total+1))
    mut_diff = np.log((mut_ko_gene_expression[ko_gene]+1)/(mut_baseline_gene_expression_total+1))
    diff = mut_diff+wt_diff
    rows = []
    for idx in np.argsort(np.abs(diff))[::-1][:30]:
        rows.append((f'{protein_id_name[idx_to_node[idx]]:10s}',
                     f'{wt_diff[idx]:.3f}',
                     f'{mut_diff[idx]:.3f}', 
                     f'{diff[idx]:.3f}'))
    title = f'Knockout: {protein_id_name[ko_gene]}'
    print('-'*len(title))
    print(title)
    print('-'*len(title))
    print(tabulate(rows, headers=['Gene', 'WT Diff', 'Mut Diff', 'Mut-WT Diff']))
#%%
print('Knockout gene combined differential expression')
for i, (ko_gene, distance) in enumerate(combined_proportion_distance[:30]):
    wt_diff = (wt_ko_gene_expression[ko_gene])-(wt_baseline_gene_expression_total)
    mut_diff = (mut_ko_gene_expression[ko_gene])-(mut_baseline_gene_expression_total)
    diff = mut_diff+wt_diff
    rows = []
    for idx in np.argsort(np.abs(diff))[::-1][:30]:
        rows.append((f'{protein_id_name[idx_to_node[idx]]:10s}',
                     f'{wt_diff[idx]:.3f}',
                     f'{mut_diff[idx]:.3f}', 
                     f'{diff[idx]:.3f}'))
    title = f'Knockout: {protein_id_name[ko_gene]}'
    print('-'*len(title))
    print(title)
    print('-'*len(title))
    print(tabulate(rows, headers=['Gene', 'WT Diff', 'Mut Diff', 'Mut-WT Diff']))

# %%
