#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import util

#%%
mut_tmstp = '20230608_093734'

genotype = 'mutant'
outdir = f'../../output/{mut_tmstp}'
ko_dir = f'{outdir}/knockout_simulations'
# Load the 
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
protein_name_id = {v:k for k,v in protein_id_name.items()}
#%%
all_genes = set(node_to_idx.keys())
cell_type_trajectories = {}
cell_type_proportions = {}
mean_trajectories = {}

for i,ko_gene in enumerate(all_genes):
    ko_gene_name = protein_id_name[ko_gene]
    with open(f'{ko_dir}/{ko_gene_name}_knockout_cell_type_proportions_mutant.pickle', 'rb') as f:
        cell_type_proportions[ko_gene] = pickle.load(f)

    with open(f'{ko_dir}/{ko_gene_name}_knockout_cell_type_trajectories_mutant.pickle', 'rb') as f:
        cell_type_trajectories[ko_gene] = pickle.load(f)

    with open(f'{ko_dir}/{ko_gene_name}_knockout_mean_trajectories_mutant.pickle', 'rb') as f:
        mean_trajectories[ko_gene] = pickle.load(f)
# %%
# Sort the genes by the distance between the wildtype and mutant cell type distributions
# First get the distance between the wildtype and mutant cell type distributions
proportion_distance = {}
for ko_gene in all_genes:
    perturb_proportions, baseline_proportions = cell_type_proportions[ko_gene]
    proportion_distance[ko_gene] = np.linalg.norm(perturb_proportions - baseline_proportions, ord=2)
proportion_distance = sorted(proportion_distance.items(), key=lambda x: x[1], reverse=True)
# %%
mutant_inputs = pickle.load(open(f'{outdir}/input_selection_pvalues_mutant.pickle','rb'))
mut_selected = {}
for target_gene in mutant_inputs:
    mut_selected[target_gene] = set([idx for idx, pval in mutant_inputs[target_gene].items() if pval < .01])
mut_targets = {}
for target_gene in mut_selected:
    for input_gene in mut_selected[target_gene]:
        if input_gene not in mut_targets:
            mut_targets[input_gene] = []
        mut_targets[input_gene].append(target_gene)
#%%
num_targets_sorted = sorted(mut_targets, key=lambda x: len(mut_targets[x]), reverse=True)
num_targets_rank = {gene: i for i, gene in enumerate(num_targets_sorted)}

#%%
for i, (ko_gene, distance) in enumerate(proportion_distance[:30]):
    print(f'{protein_id_name[ko_gene]:10s}: {distance:.3f}'\
          f' {i:3d} {num_targets_rank[ko_gene]:5d} {i-num_targets_rank[ko_gene]}')

# %%
# Get the gene expression distribution corresponding to each knockout
ko_gene_expression = {}
for ko_gene in all_genes:
    ko_gene_expression[ko_gene] = util.tonp(mean_trajectories[ko_gene].mean(axis=0))
#%%
# Get the most differentially expressed genes corresponding to each knockout
baseline_trajectories = pickle.load(open(f'{outdir}/baseline_trajectories_mutant_all.pickle', 'rb'))
baseline_gene_expression_traj = baseline_trajectories.mean(axis=1)
baseline_gene_expression_total = baseline_trajectories.mean(axis=0).mean(axis=0)
#%%
for i, (ko_gene, distance) in enumerate(proportion_distance[:30]):
    print(f'{protein_id_name[ko_gene]:10s}')
    diff = ko_gene_expression[ko_gene] - baseline_genigephe_expression_total
    for idx in np.argsort(np.abs(diff))[::-1][:10]:
        print(f'{protein_id_name[idx_to_node[idx]]:10s}: {diff[idx]:.3f}')
    print('-'*40)
    # ko_diff_gene_expression_sorted = sorted(enumerate(ko_gene_expression[ko_gene]), key=lambda x: x[1], reverse=True)
    # for idx, expr in ko_gene_expression_sorted[:10]:
    #     print(f'{adata.var_names[idx]:10s}: {expr:.3f}')
    # print('-'*40)
# %%
# Plot a heatmap of the gene expression differences over time
ko_gene = protein_name_id['ID3']
for i, (ko_gene, distance) in enumerate(proportion_distance[:10]):
    print(f'{protein_id_name[ko_gene]:10s}')
    diff = util.tonp(mean_trajectories[ko_gene]) - baseline_gene_expression_total
    idxs = np.argsort(np.abs(diff.mean(axis=0)))[::-1][:10]
    plt.figure()
    # Use a diverging colormap centered at 0
    plt.imshow(diff[:,idxs].T, aspect='auto', interpolation='none', cmap='coolwarm')
    # Add the gene names as row labels
    plt.yticks(range(len(idxs)), [protein_id_name[idx_to_node[idx]] for idx in idxs])
    # Divide the x-axis tick labels by 4 since we take 4 steps per unit
    plt.xticks(np.arange(0, diff.shape[0], 20),  np.arange(0, diff.shape[0]//4, 20//4))
    plt.colorbar()
    # TODO save the figure
    plt.savefig() 
# %%
