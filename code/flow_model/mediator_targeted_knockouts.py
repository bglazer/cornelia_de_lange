#%%
%load_ext autoreload
%autoreload 2
#%%
import pickle
import scanpy as sc
import numpy as np
import scanpy as sc
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
from mediators import find_mediators, find_bridges

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)
#%%
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'
# %%
genotype = 'wildtype'
tmstp = '20230607_165324'
# genotype = 'mutant'
# tmstp = '20230608_093734'
data = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')
outdir = f'../../output/{tmstp}'
ko_dir = f'{outdir}/knockout_simulations'
datadir = f'{outdir}/knockout_simulations/data'
pltdir = f'{outdir}/knockout_simulations/figures'
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
all_genes = set(node_to_idx.keys())
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
name_protein_id = {name: id for id, name in protein_id_name.items()}
graph = pickle.load(open(f'../../data/filtered_graph.pickle', 'rb'))

#%%
# Load the target input list
with open(f'{outdir}/optimal_{genotype}_active_inputs.pickle', 'rb') as f:
    target_active_genes = pickle.load(f)
#%%
results  = find_bridges(target_active_genes,
                        knowledge_graph=graph,
                        #all_shortest_paths=None,
                        verbose=False,
                        threshold=0.01)
mediator_probs, mediated_interactions = results #, all_shortest_paths = results


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
num_nodes = X.shape[1]
hidden_dim = 64
num_layers = 3

#%%
device='cuda:0'
n_repeats = 10
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
def knockout(mediator, X, i):
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
    # Choose a random sample of NMP cells as initial points for the simulation
    nmp_cells = np.where(data.obs['cell_type'] == 'NMP')[0]
    pct_initial_cells = .20
    n_initial_cells = int(pct_initial_cells * len(nmp_cells))
    start_idxs = np.random.choice(nmp_cells, size=n_initial_cells, replace=False)
    repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
    repeats = repeats.to(device)
    Xnp = util.tonp(X)
    simulator = Simulator(model, X, boundary=False, show_progress=False, device=device)
    print(f'Knockout Mediator {protein_id_name[mediator]:10s}: ({i+1}/{len(mediated_interactions)})', flush=True)
    
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats, t_span)
    
    mediator_gene_name = protein_id_name[mediator]
    perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    
    # Aggregate the individual cell trajectories by mean
    mean_trajectories = perturb_trajectories.mean(dim=1)
    # Save the mean trajectories
    with open(f'{datadir}/{mediator_gene_name}_{genotype}_mediator_knockout_mean_trajectories.pickle', 'wb') as f:
        pickle.dump(mean_trajectories, f)

    plotting.time_distribution(perturb_trajectories_np[:,:], pca,
                          label=f'{mediator_gene_name} Mediator Knockout',
                          baseline=Xnp)
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_time_distribution.png')
    plt.close()

    plotting.cell_type_distribution(perturb_trajectories_np[:,:], perturb_idxs_np[:,:], data,
                                    cell_types, pca, f'{mediator_gene_name} Mediator Knockout', 
                                    baseline_trajectories_np.reshape(-1, baseline_trajectories_np.shape[-1]))
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_cell_type_distribution.png')
    plt.close()

    plotting.sample_trajectories(perturb_trajectories_np, Xnp, pca, f'{mediator_gene_name} Mediator Knockout')
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_trajectories.png')
    plt.close()
    trajectories = plotting.compare_cell_type_trajectories([perturb_idxs_np, baseline_idxs],
                                                           [data, data], 
                                                           cell_types,
                                                           [mediator_gene_name, 'baseline'])
    # Save the trajectories
    with open(f'{datadir}/{mediator_gene_name}_{genotype}_mediator_knockout_cell_type_trajectories.pickle', 'wb') as f:
        pickle.dump(trajectories, f)
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_cell_type_trajectories.png')
    plt.close()
    # Plot side by side bar charts of the cell type proportions
    perturb_cell_proportions, perturb_cell_proportion_errors = plotting.calculate_cell_type_proportion(perturb_idxs_np, data, cell_types, n_repeats=n_repeats, error=True)
    # TODO manually entering the n_repeats for baseline as 10, need to figure this out dynamically
    baseline_cell_proportions, baseline_cell_proportion_errors = plotting.calculate_cell_type_proportion(baseline_idxs, data, cell_types, n_repeats=10, error=True)
    
    # Save the cell type proportions
    plotting.cell_type_proportions(proportions=(perturb_cell_proportions, 
                                                baseline_cell_proportions), 
                                   proportion_errors=(perturb_cell_proportion_errors,
                                                      baseline_cell_proportion_errors),
                                   cell_types=cell_types, 
                                   labels=[f'{mediator_gene_name} Mediator Knockout', f'{genotype.capitalize()} baseline'])
    plt.savefig(f'{pltdir}/{mediator_gene_name}_mediator_knockout_cell_type_proportions.png',
                bbox_inches='tight');
    plt.close();
    with open(f'{datadir}/{mediator_gene_name}_{genotype}_mediator_knockout_cell_type_proportions.pickle', 'wb') as f:
        pickle.dump((perturb_cell_proportions, baseline_cell_proportions), f)
#%%
# mediator = 'ENSMUSP00000031650'
# knockout(mediator, X, 0)
#%%
from joblib import Parallel, delayed
_=Parallel(n_jobs=8)(delayed(knockout)(mediator, X, i) for i,mediator in enumerate(mediated_interactions))

# %%
# Save the mediated interactions
with open(f'{outdir}/mediated_interactions_{genotype}.pickle', 'wb') as f:
    pickle.dump(mediated_interactions, f)
#%%
cell_type_ko_proportions = {}

for i, mediator in enumerate(mediated_interactions):
    mediator_gene_name = protein_id_name[mediator]
    # Load the knockout results
    with open(f'{datadir}/{mediator_gene_name}_{genotype}_mediator_knockout_cell_type_proportions.pickle', 'rb') as f:
        ko_cell_type_proportions = pickle.load(f)
    perturb_cell_proportions, baseline_cell_proportions = ko_cell_type_proportions

    with open(f'{datadir}/{mediator_gene_name}_{genotype}_mediator_knockout_cell_type_proportions.pickle', 'rb') as f:
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
