#%%
# %load_ext autoreload
# %autoreload 2
#%%
import pickle
import scanpy as sc
import numpy as np
from scipy.stats import hypergeom
import scanpy as sc
# import numpy as np
import pickle
from flow_model import GroupL1FlowModel
import torch
import sys
sys.path.append('..')
import util
import numpy as np
from sklearn.decomposition import PCA
from simulator import Simulator
from matplotlib import pyplot as plt
import plotting
from tabulate import tabulate
from collections import Counter

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)
#%%
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'
# %%
# genotype = 'wildtype'
# tmstp = '20230607_165324'
genotype = 'mutant'
tmstp = '20230608_093734'
data = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')
cell_types = {c:i for i,c in enumerate(set(data.obs['cell_type']))}
outdir = f'../../output/{tmstp}'
ko_dir = f'{outdir}/knockout_simulations'
pltdir = f'{outdir}/knockout_simulations/figures'
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
all_genes = set(node_to_idx.keys())
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
name_protein_id = {name: id for id, name in protein_id_name.items()}

# %%
# Load the shortest paths
optimal_model_shortest_paths_graph = pickle.load(open(f'../../output/{tmstp}/optimal_{genotype}_shortest_paths_graph.pickle', 'rb'))
shortest_paths_to_target = pickle.load(open(f'../../output/{tmstp}/optimal_{genotype}_shortest_paths.pickle', 'rb'))
#%%
shortest_paths_from_source = {}
for target, paths in shortest_paths_to_target.items():
    for path in paths:
        source = path[0]
        if source not in shortest_paths_from_source:
            shortest_paths_from_source[source] = []
        shortest_paths_from_source[source].append(path)

# Reverse the order of the shortest paths to target
shortest_paths_to_target_ = {}
for target, paths in shortest_paths_to_target.items():
    for path in paths:
        source = path[0]
        if target not in shortest_paths_to_target_:
            shortest_paths_to_target_[target] = []
        shortest_paths_to_target_[target].append(path[::-1])

shortest_paths_to_target = shortest_paths_to_target_


#%%
all_shortest_paths = pickle.load(open(f'../../output/{tmstp}/all_shortest_paths.pickle', 'rb'))

all_shortest_paths_from_source = {}
for target, paths in all_shortest_paths.items():
    for path in paths:
        source = path[0]
        if source not in all_shortest_paths_from_source:
            all_shortest_paths_from_source[source] = []
        all_shortest_paths_from_source[source].append(path)

all_shortest_paths_to_target = {}
for source, paths in all_shortest_paths.items():
    for path in paths:
        target = path[-1]
        if target not in all_shortest_paths_to_target:
            all_shortest_paths_to_target[target] = []
        all_shortest_paths_to_target[target].append(path[::-1])
        
#%%
# Calculate the percentage of shortest paths that a mediator appears in for each knockout gene
def count_mediators(all_paths):
    mediators = {}
    target_counts = {}
    for source, paths in all_paths.items():
        for path in paths:
            if len(path) > 2:
                if source not in mediators:
                    mediators[source] = {}
                target = path[-1]
                if target not in mediators[source]:
                    mediators[source][target] = set()
                for mediator in path[1:-1]:
                    mediators[source][target].add(mediator)
        if source in mediators:
            target_counts[source] = len(mediators[source])

    mediator_counts = {}
    for source, target_mediators in mediators.items():
        mediator_counts[source] = {}
        for target, _mediators in target_mediators.items():
            for mediator in _mediators:
                if mediator not in mediator_counts[source]:
                    mediator_counts[source][mediator] = 0
                mediator_counts[source][mediator] += 1

    return mediator_counts, target_counts

#%%
def mediator_probability(shortest_path_set, all_shortest_paths):
    # k - number of matches in chosen set, i.e. number of shortest paths that a mediator appears in
    # M - total number of items, 
    # n - number of matches in population
    # N - size of set chosen at random
    mediator_counts, target_counts = count_mediators(shortest_path_set)
    total_mediator_counts, _ = count_mediators(all_shortest_paths)
    mediator_probs = {}
    for source, mediators in mediator_counts.items():
        for mediator, count in mediators.items():
            # number of times we see this mediator in any shortest path from this source to the chosen targets
            k = mediator_counts[source][mediator]
            # Total possible number of shortest paths
            M = len(node_to_idx)
            # Number of times we see this mediator in any shortest path from the source to any target
            n = total_mediator_counts[source][mediator]
            # Number of targets of this source
            N = target_counts[source]
            # Probability of observing k or more matches 
            p = 1-hypergeom.cdf(k-1, M, n, N)
            mediator_probs[(source, mediator)] = p
            print(protein_id_name[source], protein_id_name[mediator], p)
            print('k=',k)
            print('M=',M)
            print('n=',n)
            print('N=',N)
    mediator_probs = {k: v for k, v in sorted(mediator_probs.items(), key=lambda item: item[1])}
    return mediator_probs

mediator_probs_source = mediator_probability(shortest_paths_from_source, all_shortest_paths_from_source)
mediator_probs_target = mediator_probability(shortest_paths_to_target, all_shortest_paths_to_target)

#%%
print("Significant mediators from source:")
mediated_sources = {}
num_significant = 0
for source_mediator, prob in mediator_probs_source.items():
    source, mediator = source_mediator
    if prob < 0.01:
        if mediator not in mediated_sources:
            mediated_sources[mediator] = []
        mediated_sources[mediator].append(source)
        num_significant += 1
        print(protein_id_name[source], protein_id_name[mediator], f'p={prob:.2e}')
print("Significant mediators to targets:")
mediated_targets = {}
for target_mediator, prob in mediator_probs_target.items():
    target, mediator = target_mediator
    if prob < 0.01:
        if mediator not in mediated_targets:
            mediated_targets[mediator] = []
        mediated_targets[mediator].append(target)
        num_significant += 1
        print(protein_id_name[target], protein_id_name[mediator], f'p={prob:.2e}')
#%%
print(f'Number of significant mediators: {num_significant} out of {len(mediator_probs_source)+len(mediator_probs_target)}')


#%%
mediated_interactions = {}
for mediator, sources in mediated_sources.items():
    for source in sources:
        # print(protein_id_name[mediator], protein_id_name[source])
        for path in shortest_paths_from_source[source]:
            if mediator in path:
                target = path[-1]
                if mediator not in mediated_interactions:
                    mediated_interactions[mediator] = set()
                mediated_interactions[mediator].add((source, target))
for mediator, targets in mediated_targets.items():
    for target in targets:
        # print(protein_id_name[mediator], protein_id_name[target])
        for path in shortest_paths_to_target[target]:
            # This is reversed from the normal direction because we inverted the 
            # paths above to make the mediator_count code work for both sources and targets
            source = path[-1]
            target = path[0]
            if mediator in path:
                if mediator not in mediated_interactions:
                    mediated_interactions[mediator] = set()
                mediated_interactions[mediator].add((source, target))
for mediator, interactions in mediated_interactions.items():
    print(protein_id_name[mediator])
    for interaction in sorted(interactions):
        source, target = interaction
        print(f'    {protein_id_name[source]} -> {protein_id_name[target]}')
#%%
regulatory_graph = pickle.load(open('../../data/filtered_graph.pickle','rb'))
degree_rank = {x[1][0]: x[0] 
               for x in enumerate(sorted(regulatory_graph.degree, key=lambda x:x[1], reverse=True))}    

mediated_counts = []

for mediator in mediated_interactions:
    mediated_counts.append((len(mediated_interactions[mediator]), mediator))
headers = ['Mediator',
           'Number of Mediated Interactions',
           'In Degree',
           'Out Degree',
           'Degree Rank']
rows = []
for count,mediator in sorted(mediated_counts, reverse=True):
    rows.append((protein_id_name[mediator], 
                 count,
                 regulatory_graph.in_degree(mediator), 
                 regulatory_graph.out_degree(mediator),
                 degree_rank[mediator]))
print(tabulate(rows, headers=headers))


#%%
# Check if ko_dir exists. This is where we will save the knockout simulations and figures
import os
if not os.path.exists(ko_dir):
    os.makedirs(ko_dir)
    os.makedirs(pltdir)

state_dict = torch.load(f'{outdir}/models/optimal_{genotype}.torch')

# %%
X = torch.tensor(data.X.toarray()).float()
cell_types = {c:i for i,c in enumerate(sorted(set(data.obs['cell_type'])))}
proj = np.array(data.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = data.uns['pca_mean']
pca.components_ = data.uns['PCs']
proj = np.array(pca.transform(X))[:,0:2]
T = data.obsm['transition_matrix']

V = util.velocity_vectors(T, X)
V_emb = util.embed_velocity(X, V, lambda x: np.array(pca.transform(x)[:,0:2]))

# %%
torch.set_num_threads(24)
start_idxs = data.uns['initial_points_via']
num_nodes = X.shape[1]
hidden_dim = 64
num_layers = 3

#%%
device='cuda:0'
n_repeats = 10
start_idxs = data.uns['initial_points_via']
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
len_trajectory = 98
n_steps = len_trajectory*4
t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{outdir}/baseline_trajectories_{genotype}.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{outdir}/baseline_nearest_cell_idxs_{genotype}.pickle', 'rb'))
# baseline_velo, _ = plotting.compute_velo(model=model, X=X, numpy=True)

#%%
def knockout(mediator, X, repeats, i):
    device = f'cuda:{i%4}'
    model = GroupL1FlowModel(input_dim=num_nodes, 
                             hidden_dim=hidden_dim, 
                             num_layers=num_layers,
                             predict_var=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Set the input weight multiplier of the interaction knockout target to a negative value
    # This effectively removes that specific interaction from the model
    # The input will be multiplied by ReLU(-1) = 0, so the interaction will be removed
    with torch.no_grad():
        for source, target in mediated_interactions[mediator]:
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            group_l1 = model.models[target_idx].group_l1
            group_l1[source_idx] = -1
    
    X = X.to(device)
    repeats = repeats.to(device)
    Xnp = util.tonp(X)
    simulator = Simulator(model, X, device=device)
    print(f'Knockout Mediator {protein_id_name[mediator]:10s}: ({i+1}/{len(mediated_interactions)})', flush=True)
    
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats, t_span, 
                                                                    boundary=False,
                                                                    show_progress=False)
    
    mediator_gene_name = protein_id_name[mediator]
    perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    
    # Aggregate the individual cell trajectories by mean
    mean_trajectories = perturb_trajectories.mean(dim=1)
    # Save the mean trajectories
    with open(f'{ko_dir}/{mediator_gene_name}_mediator_knockout_mean_trajectories_{genotype}.pickle', 'wb') as f:
        pickle.dump(mean_trajectories, f)

    plotting.distribution(perturb_trajectories_np[:,::10], pca,
                          label=f'{mediator_gene_name} Mediator Knockout',
                          baseline=Xnp)
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_distribution.png')
    plt.close()

    plotting.sample_trajectories(perturb_trajectories_np, Xnp, pca, f'{mediator_gene_name} Mediator Knockout')
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_trajectories.png')
    plt.close()
    trajectories = plotting.compare_cell_type_trajectories([perturb_idxs_np, baseline_idxs],
                                                           [data, data], 
                                                           cell_types,
                                                           [mediator_gene_name, 'baseline'])
    mut_cell_type_traj, perturb_cell_type_traj = trajectories
    # Save the trajectories
    with open(f'{ko_dir}/{mediator_gene_name}_mediator_knockout_cell_type_trajectories_{genotype}.pickle', 'wb') as f:
        pickle.dump(trajectories, f)
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_cell_type_trajectories.png')
    plt.close()
    mut_ct = data.obs['cell_type'].value_counts()/data.shape[0]
    mut_ct = mut_ct[sorted(cell_types)]
    mut_total_cell_types = mut_cell_type_traj.sum(axis=1)/mut_cell_type_traj.sum()
    perturb_total_cell_types = perturb_cell_type_traj.sum(axis=1)/perturb_cell_type_traj.sum()
    # Plot side by side bar charts of the cell type proportions
    n_cells = (len(start_idxs) * n_repeats * n_steps)
    perturb_cell_type_traj = trajectories[0]
    baseline_cell_type_traj = trajectories[1]
    perturb_cell_proportions = perturb_cell_type_traj.sum(axis=1) / n_cells
    baseline_cell_proportions = baseline_cell_type_traj.sum(axis=1) / n_cells
    # Save the cell type proportions
    plotting.cell_type_proportions(proportions=(perturb_cell_proportions, 
                                                baseline_cell_proportions), 
                                   cell_types=cell_types, 
                                   labels=[f'{mediator_gene_name} Mediator Knockout', 'Mutant baseline'])
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_cell_type_proportions.png',
                bbox_inches='tight');
    plt.close();
    with open(f'{ko_dir}/{mediator_gene_name}_mediator_knockout_cell_type_proportions_{genotype}.pickle', 'wb') as f:
        pickle.dump((perturb_cell_proportions, baseline_cell_proportions), f)

#%%
from joblib import Parallel, delayed
_=Parallel(n_jobs=8)(delayed(knockout)(mediator, X, repeats, i) for i,mediator in enumerate(mediated_interactions))

# %%
# Save the mediated interactions
with open(f'{outdir}/mediated_interactions_{genotype}.pickle', 'wb') as f:
    pickle.dump(mediated_interactions, f)
#%%
cell_type_ko_proportions = {}

for i, mediator in enumerate(mediated_interactions):
    mediator_gene_name = protein_id_name[mediator]
    # Load the knockout results
    with open(f'{ko_dir}/{mediator_gene_name}_mediator_knockout_cell_type_proportions_{genotype}.pickle', 'rb') as f:
        ko_cell_type_proportions = pickle.load(f)
    perturb_cell_proportions, baseline_cell_proportions = ko_cell_type_proportions

    with open(f'{ko_dir}/{mediator_gene_name}_mediator_knockout_cell_type_proportions_{genotype}.pickle', 'rb') as f:
        cell_type_ko_proportions[mediator] = {}
        for i,cell_type in enumerate(cell_types):
            cell_type_ko_proportions[mediator][cell_type] = perturb_cell_proportions[i]

# %%
# Sort by the difference in means
cell_type_changes = []
for mediator, proportions in cell_type_ko_proportions.items():
    cell_type_array = np.array([proportions[cell_type] for cell_type in cell_types])
    cell_type_diffs = cell_type_array - baseline_cell_proportions
    cell_type_diff_sum = np.sum(np.abs(cell_type_diffs))
    cell_type_changes.append((cell_type_diff_sum, mediator, cell_type_diffs))
sorted_cell_type_changes = sorted(cell_type_changes, reverse=True)
#%%
mediated_counts = {v:k for k,v in mediated_counts}
#%%
# Genes with highest variance in the data
variance = np.var(data.X.toarray(), axis=0)
variance = np.array(variance).flatten()
sorted_var_idxs = np.argsort(variance)[::-1]
# Genes with highest mean in the data
mean = np.mean(data.X.toarray(), axis=0)
mean = np.array(mean).flatten()
sorted_mean_idxs = np.argsort(mean)[::-1]
#%%
mean_sorted_genes = {idx_to_node[idx]: i for i,idx in enumerate(sorted_mean_idxs)}
var_sorted_genes = {idx_to_node[idx]: i for i,idx in enumerate(sorted_var_idxs)}
#%%
# Write a table
# Print the table
rows = []
headers = ["Mediator", "Cell Type Change", "Mediated Count", 
           "In Degree", "Out Degree", "Degree Rank",
           "Mean Rank", "Variance Rank"]
for cell_type_diff_sum, mediator, cell_type_diffs in sorted_cell_type_changes:
    # Get the list of genes that are significantly mediated by this mediator
    mediator_gene = protein_id_name[mediator]
    count = mediated_counts[mediator]

    if mediator not in mean_sorted_genes:
        mean_sorted_genes[mediator] = 9999
    if mediator not in var_sorted_genes:
        var_sorted_genes[mediator] = 9999

    rows.append([
        f'{protein_id_name[mediator]:10s}',
        f'{cell_type_diff_sum:4f}',
        f'{count:5d}',
        f'{regulatory_graph.in_degree(mediator):8d}',
        f'{regulatory_graph.out_degree(mediator):8d}',
        f'{degree_rank[mediator]:4d}',
        f'{mean_sorted_genes[mediator]:4d}',
        f'{var_sorted_genes[mediator]:4d}',
    ])

print(tabulate(rows, headers=headers))


# %%
# Print the mediated interactions for each mediator
for cell_type_diff_sum, mediator, cell_type_diffs in sorted_cell_type_changes:
    mediator_gene = protein_id_name[mediator]
    print(f'{mediator_gene:10s} {cell_type_diff_sum:4f}')
    for source, target in mediated_interactions[mediator]:
        print(f'    {protein_id_name[source]:10s} -> {protein_id_name[target]:10s}')
    print('-'*80)
# %%
