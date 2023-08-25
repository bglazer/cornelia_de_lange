#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import util
from tabulate import tabulate

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

# TODO! Need to rerun WT knockout simulations because they're named wrong
# That's suspicious, makes it seem like they may be incorrect
for i,ko_gene in enumerate(all_genes):
    ko_gene_name = protein_id_name[ko_gene]
    with open(f'{wt_ko_dir}/data/{ko_gene_name}_wildtype_knockout_cell_type_proportions.pickle', 'rb') as f:
        wt_cell_type_proportions[ko_gene] = pickle.load(f)

    with open(f'{wt_ko_dir}/data/{ko_gene_name}_wildtype_knockout_cell_type_trajectories.pickle', 'rb') as f:
        wt_cell_type_trajectories[ko_gene] = pickle.load(f)

    with open(f'{wt_ko_dir}/data/{ko_gene_name}_wildtype_knockout_mean_trajectories.pickle', 'rb') as f:
        wt_mean_trajectories[ko_gene] = pickle.load(f)
# %%
# Sort the genes by the distance between the wildtype and mutant cell type distributions
# First get the distance between the wildtype and mutant cell type distributions
proportion_distance = {}
for ko_gene in all_genes:
    mut_perturb_proportions, mut_baseline = mut_cell_type_proportions[ko_gene]
    wt_perturb_proportions, wt_baseline = wt_cell_type_proportions[ko_gene]
    proportion_distance[ko_gene] = np.sum(np.abs(wt_perturb_proportions - wt_baseline)) - np.sum(np.abs(mut_perturb_proportions - mut_baseline))
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
headers = ['Gene', 'Distance', 'WT Diff', 'Mut Diff', 'WT Num Targets Rank', 'Mut Num Targets Rank']
rows = []
for i, (ko_gene, distance) in enumerate(proportion_distance):
    if ko_gene in wt_targets_rank:
        wt_rank = wt_targets_rank[ko_gene]
    else:
        wt_rank = len(all_genes)
    if ko_gene in mut_targets_rank:
        mut_rank = mut_targets_rank[ko_gene]
    else:
        mut_rank = len(all_genes)

    mut_perturb_proportions, mut_baseline = mut_cell_type_proportions[ko_gene]
    wt_perturb_proportions, wt_baseline = wt_cell_type_proportions[ko_gene]
    wt_dist = np.sum(np.abs(wt_perturb_proportions - wt_baseline))
    mut_dist = np.sum(np.abs(mut_perturb_proportions - mut_baseline))

    rows.append((f'{protein_id_name[ko_gene]:10s}',
                 f'{distance:.3f}',
                 f'{wt_dist:.3f}',
                 f'{mut_dist:.3f}',
                 f'{wt_rank:5d}', 
                 f'{mut_rank:5d}'))
print(tabulate(rows, headers=headers))

# %%
# Get the gene expression distribution corresponding to each knockout
mut_ko_gene_expression = {}
wt_ko_gene_expression = {}
for ko_gene in all_genes:
    mut_ko_gene_expression[ko_gene] = util.tonp(mut_mean_trajectories[ko_gene].mean(axis=0))
    wt_ko_gene_expression[ko_gene] = util.tonp(wt_mean_trajectories[ko_gene].mean(axis=0))
#%%
# Get the most differentially expressed genes corresponding to each knockout
wt_baseline_trajectories = pickle.load(open(f'{mut_outdir}/baseline_trajectories_mutant_all.pickle', 'rb'))
wt_baseline_gene_expression_traj = wt_baseline_trajectories.mean(axis=1)
wt_baseline_gene_expression_total = wt_baseline_trajectories.mean(axis=0).mean(axis=0)
mut_baseline_trajectories = pickle.load(open(f'{mut_outdir}/baseline_trajectories_mutant_all.pickle', 'rb'))
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
# Plot a heatmap of the gene expression differences over time
# ko_gene = protein_name_id['ID3']
# for i, (ko_gene, distance) in enumerate(proportion_distance[:10]):
#     print(f'{protein_id_name[ko_gene]:10s}')
#     diff = util.tonp(mut_mean_trajectories[ko_gene]) - baseline_gene_expression_total
#     idxs = np.argsort(np.abs(diff.mean(axis=0)))[::-1][:10]
#     plt.figure()
#     # Use a diverging colormap centered at 0
#     plt.imshow(diff[:,idxs].T, aspect='auto', interpolation='none', cmap='coolwarm')
#     # Add the gene names as row labels
#     plt.yticks(range(len(idxs)), [protein_id_name[idx_to_node[idx]] for idx in idxs])
#     # Divide the x-axis tick labels by 4 since we take 4 steps per unit
#     plt.xticks(np.arange(0, diff.shape[0], 20),  np.arange(0, diff.shape[0]//4, 20//4))
#     plt.colorbar()
#     # TODO save the figure
#     plt.savefig() 
# %%
