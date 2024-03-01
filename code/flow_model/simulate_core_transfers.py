#%%
# %load_ext autoreload
# %autoreload 2
#%%
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
import plotting
from joblib import Parallel, delayed
import os
from tabulate import tabulate
import random

#%%
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# %%
# Load the models
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


#%%
# Check if the output directories exists. This is where we will save the transfer simulations and figures
import os
if not os.path.exists(transfer_dir):
    os.makedirs(transfer_dir)
    os.makedirs(pltdir)
    os.makedirs(datadir)

source_state_dict = torch.load(f'{src_outdir}/models/optimal_{source_genotype}.torch')
target_state_dict = torch.load(f'{tgt_outdir}/models/optimal_{target_genotype}.torch')

# %%
tgt_X = torch.tensor(tgt_data.X.toarray()).float()
tgt_Xnp = util.tonp(tgt_X)
src_X = torch.tensor(src_data.X.toarray()).float()
src_Xnp = util.tonp(src_X)
proj = np.array(tgt_data.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = tgt_data.uns['pca_mean']
pca.components_ = tgt_data.uns['PCs']
proj = np.array(pca.transform(tgt_X))[:,0:2]

# %%
torch.set_num_threads(40)
device='cpu'
num_nodes = tgt_X.shape[1]
hidden_dim = 64
num_layers = 3

#%%
cell_types = {c:i for i,c in enumerate(sorted(set(tgt_data.obs['cell_type'])))}
idx_to_cell_type = {v:k for k,v in cell_types.items()}

#%%
# Convert from ids to gene names
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}

#%%
n_repeats = 10
start_idxs = src_data.uns['initial_points_nmp']
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
len_trajectory = tgt_data.uns['best_trajectory_length']
n_steps = tgt_data.uns['best_step_size']*len_trajectory

t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{src_outdir}/baseline_trajectories_{source_genotype}.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{src_outdir}/baseline_nearest_cell_idxs_{source_genotype}.pickle', 'rb'))
# baseline_velo,_ = plotting.compute_velo(model=model, X=src_X, numpy=True)
# baseline_X = baseline_trajectories_np.reshape(-1, num_nodes)
baseline_cell_proportions, baseline_cell_errors = plotting.calculate_cell_type_proportion(baseline_idxs, src_data, cell_types, n_repeats, error=True)
#%%
# src_baseline_trajectories = pickle.load(open(f'{src_outdir}/baseline_trajectories_{source_genotype}.pickle', 'rb'))
# src_baseline_trajectories_np = src_baseline_trajectories
# src_baseline_idxs = pickle.load(open(f'{src_outdir}/baseline_nearest_cell_idxs_{source_genotype}.pickle', 'rb'))
# src_baseline_cell_proportions, src_baseline_cell_errors = plotting.calculate_cell_type_proportion(src_baseline_idxs, src_data, cell_types, n_repeats, error=True)

# tgt_baseline_trajectories = pickle.load(open(f'{tgt_outdir}/baseline_trajectories_{target_genotype}.pickle', 'rb'))
# tgt_baseline_trajectories_np = tgt_baseline_trajectories
# tgt_baseline_idxs = pickle.load(open(f'{tgt_outdir}/baseline_nearest_cell_idxs_{target_genotype}.pickle', 'rb'))
# tgt_baseline_cell_proportions, tgt_baseline_cell_errors = plotting.calculate_cell_type_proportion(tgt_baseline_idxs, tgt_data, cell_types, n_repeats, error=True)

#%%
num_gpus = 4
def simulate_transfer(transfer_genes, idx, label):
    # For testing purposes return a random cell type proportion of the same shape as the baseline
    # return transfer_genes, np.random.rand(*baseline_cell_proportions.shape), np.random.rand(*baseline_cell_proportions.shape)
    gpu = idx % num_gpus
    device = f'cuda:{gpu}'

    # Re-initialize the model and simulator at each iteration
    tgt_model = GroupL1FlowModel(input_dim=num_nodes, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=num_layers,
                                 predict_var=True)
    tgt_model.load_state_dict(target_state_dict)
    tgt_model = tgt_model.to(device)
    src_model = GroupL1FlowModel(input_dim=num_nodes, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=num_layers,
                                 predict_var=True)
    src_model.load_state_dict(source_state_dict)
    src_model = src_model.to(device)
    for transfer_gene in transfer_genes:
        model_idx = node_to_idx[transfer_gene]
        src_gene_model = src_model.models[model_idx]
        tgt_gene_model = tgt_model.models[model_idx]
        tgt_gene_model.load_state_dict(src_gene_model.state_dict())
    simulator = Simulator(tgt_model, src_X.to(device), device=device, boundary=False, show_progress=False)
    repeats_gpu = repeats.to(device)
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats_gpu, t_span)
    # perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    # Delete full trajectories so that we can free the GPU memory
    del perturb_trajectories
    del perturb_nearest_idxs
    # Tell the garbage collector to free the GPU memory
    torch.cuda.empty_cache()

    perturb_cell_proportions, perturb_cell_errors = plotting.calculate_cell_type_proportion(perturb_idxs_np, src_data, cell_types, n_repeats, error=True)
    
    proportion_dict = {'transfer_genes': transfer_genes,
                       'perturb_proportions':perturb_cell_proportions, 
                       'perturb_errors': perturb_cell_errors}
    with open(f'{datadir}/{label}_combination_{idx}_{transfer}_transfer_cell_type_proportions.pickle', 'wb') as f:
        pickle.dump(proportion_dict, f)
    d = np.abs(perturb_cell_proportions - baseline_cell_proportions).sum()
    print(f'{d}, {" ".join([protein_id_name[g] for g in transfer_genes])}', flush=True)
    return transfer_genes, perturb_cell_proportions, perturb_cell_errors

#%%
def individual_transfers(core_genes):
    idx=0
    parallel = Parallel(n_jobs=12)
    gene_sets = []
    for out_gene in core_genes:
        leave_one_out = list((core_genes - set([out_gene])))
        gene_sets.append((idx, leave_one_out, out_gene))
        idx+=1
    label = 'core_leave_one_out'
    results = parallel(delayed(simulate_transfer)(transfer_genes, i, label) for i, transfer_genes,_ in gene_sets)

    table = []
    distances = []
    headers = ['Distance', 'Num Transfers', 'Cell Types Outside 2 Std Dev', '']
    i=0
    for transfer_genes, perturb_cell_proportions, perturb_cell_errors in results:
        d = np.abs(perturb_cell_proportions - baseline_cell_proportions)
        # Check if the perturb cell proportions are within 2 standard deviations of the baseline
        misses = (d > 2*perturb_cell_errors + 2*baseline_cell_errors)
        left_out_gene = gene_sets[i][2]
        table.append((protein_id_name[left_out_gene], d.sum(), len(transfer_genes), misses.sum(),
                      ' '.join([idx_to_cell_type[miss_idx] for miss_idx in np.where(misses)[0]])))
        distances.append((d, left_out_gene, transfer_genes))
        i+=1
    print(tabulate(table, headers=headers), flush=True)
    return distances

#%%
core_gene_graph = pickle.load(open(f'{datadir}/core_circuit.pickle', 'rb'))
core_genes = set(core_gene_graph.nodes)
#%%
# Transfer every gene in the core set together to get a baseline distance
core_transfer_distances = simulate_transfer(core_genes, 0, 'core')
pickle.dump(core_transfer_distances, open(f'{datadir}/core_transfer_distances.pickle', 'wb'))

#%%
# Transfer every gene in the core set individually 
core_individual_transfer_distances = individual_transfers(core_genes)
#%%
pickle.dump(core_individual_transfer_distances, open(f'{datadir}/core_individual_transfer_distances.pickle', 'wb'))
#%%
# Greedy strategy for finding the combination of genes that minimizes the distance to 
# the baseline cell type proportions
start_idx = 0
n_repeats = 10
for repeat in range(start_idx, start_idx+n_repeats):
    best_combo = tuple(core_genes)
    remaining_genes = core_genes
    best_distance = float('inf')
    num_misses = 1
    idx = 0
    while num_misses > 0 and len(remaining_genes) > 0 and best_distance > 0.05:
        all_combos = []
        for gene in remaining_genes:
            combo = tuple(set(best_combo) - set([gene]))
            all_combos.append((idx,combo))
            idx += 1 
        label = 'core_greedy'
        parallel = Parallel(n_jobs=12)
        results = parallel(delayed(simulate_transfer)(transfer_genes, i, label) for i, transfer_genes in all_combos)
        # results = [simulate_transfer(transfer_genes, i) for i,transfer_genes in all_combos]
        # Calculate the distance to the baseline cell type proportions
        combo_distances = []
        for combo, perturb_cell_proportions, perturb_cell_errors in results:
            d = np.abs(perturb_cell_proportions - baseline_cell_proportions)
            # Check if the perturb cell proportions are within 2 standard deviations of the baseline
            misses = (d > 2*perturb_cell_errors + 2*baseline_cell_errors)
            combo_distances.append((d.sum(), combo, misses))
        best_distance, best_combo, best_misses = min(combo_distances)
        num_misses = best_misses.sum()

        print(f'Remaining genes: {len(remaining_genes)}')
        print(f'Best distance: {best_distance:.3f}')
        print(f'Best combo: {",".join([protein_id_name[gene] for gene in best_combo])}')
        print(f'Cell types outside 2 std dev: {",".join([idx_to_cell_type[miss_idx] for miss_idx in np.where(best_misses)[0]])}')
        print('-'*80)
        remaining_genes = set(best_combo)
        pickle.dump(best_combo, open(f'{datadir}/core_combination_{repeat}.pickle', 'wb'))
# %%