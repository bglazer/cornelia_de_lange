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
from matplotlib import pyplot as plt
import plotting
from textwrap import fill
from joblib import Parallel, delayed
import os
import matplotlib
from itertools import combinations
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
start_idxs = tgt_data.uns['initial_points_nmp']
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
len_trajectory = tgt_data.uns['best_trajectory_length']
n_steps = tgt_data.uns['best_step_size']*len_trajectory

t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the model
model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers,
                         predict_var=True)
# Here the baseline is the source model because we're looking for transfers that 
# recapitulate the source model's behavior, using only a few genes from the source model
model.load_state_dict(source_state_dict)
model = model.to(device)
tgt_Xgpu = tgt_X.to(device)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{src_outdir}/baseline_trajectories_{source_genotype}.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{src_outdir}/baseline_nearest_cell_idxs_{source_genotype}.pickle', 'rb'))
# baseline_velo,_ = plotting.compute_velo(model=model, X=src_X, numpy=True)
# baseline_X = baseline_trajectories_np.reshape(-1, num_nodes)
baseline_cell_proportions, baseline_cell_errors = plotting.calculate_cell_type_proportion(baseline_idxs, src_data, cell_types, n_repeats, error=True)

#%%
num_gpus = 4
def simulate_transfer(transfer_genes, idx):
    # For testing purposes return a random cell type proportion of the same shape as the baseline
    # return transfer_genes, np.random.rand(*baseline_cell_proportions.shape), np.random.rand(*baseline_cell_proportions.shape)
    transfer_gene_names = [protein_id_name[gene] for gene in transfer_genes]
    # print(f'Transfer Genes {",".join(transfer_gene_names):10s}: '
    #       f'({idx+1})', flush=True)
    
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
    Xgpu = tgt_X.to(device)
    simulator = Simulator(tgt_model, Xgpu, device=device, boundary=False, show_progress=False)
    repeats_gpu = repeats.to(device)
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats_gpu, t_span)
    perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    # Delete full trajectories so that we can free the GPU memory
    del perturb_trajectories
    del perturb_nearest_idxs
    # Tell the garbage collector to free the GPU memory
    torch.cuda.empty_cache()
    # Aggregate the individual cell trajectories by mean
    mean_trajectories = perturb_trajectories_np.mean(axis=1)
    # Save the mean trajectories
    with open(f'{datadir}/combination_{idx}_{transfer}_transfer_mean_trajectories.pickle', 'wb') as f:
        mean_trajectory_dict = {'transfer_genes': transfer_genes,
                                'mean_trajectories': mean_trajectories,}
        pickle.dump(mean_trajectory_dict, f)

    perturb_cell_proportions, perturb_cell_errors = plotting.calculate_cell_type_proportion(perturb_idxs_np, tgt_data, cell_types, n_repeats, error=True)
    
    proportion_dict = {'transfer_genes': transfer_genes,
                       'perturb_proportions':perturb_cell_proportions, 
                       'perturb_errors': perturb_cell_errors}
    pickle.dump(proportion_dict,
                open(f'{datadir}/combination_{idx}_{transfer}_transfer_cell_type_proportions.pickle', 'wb'))
    d = np.abs(perturb_cell_proportions - baseline_cell_proportions).sum()
    print(f'{d}, {" ".join([protein_id_name[g] for g in transfer_genes])}', flush=True)
    return transfer_genes, perturb_cell_proportions, perturb_cell_errors

#%%
simulate_transfer(transfer_genes=['ENSMUSP00000134654'], idx=0)

#%%
proportion_distance_individual_genes = pickle.load(open(f'{tgt_outdir}/{transfer}_transfer_simulations/data/proportion_distance.pickle', 'rb'))
top_combo = (proportion_distance_individual_genes[0][0],)

remaining_genes = [gene for gene,distance in proportion_distance_individual_genes[1:50]]

best_distance = float('inf')
parallel = Parallel(n_jobs=12)
# Greedy strategy for finding the combination of genes that minimizes the distance to 
# the baseline cell type proportions
idx = 0
while best_distance > .05 and len(remaining_genes) > 0:
    all_combos = []
    for gene in remaining_genes:
        combo = top_combo + (gene,)
        idx += 1
        all_combos.append((idx,combo))

    results = parallel(delayed(simulate_transfer)(transfer_genes, i) for i, transfer_genes in all_combos)
    # results = [simulate_transfer(transfer_genes, i) for i,transfer_genes in all_combos]
    # Calculate the distance to the baseline cell type proportions
    combo_distances = []
    for combo, perturb_cell_proportions, perturb_cell_errors in results:
        d = np.abs(perturb_cell_proportions - baseline_cell_proportions).sum()
        combo_distances.append((combo, d))
    top_combo, best_distance = min(combo_distances, key=lambda x: x[1])

    print(f'Remaining genes: {len(remaining_genes)}')
    print(f'Best distance: {best_distance:.3f}')
    print(f'Best combo: {",".join([protein_id_name[gene] for gene in top_combo])}')
    print('-'*80)
    remaining_genes = [gene for gene in remaining_genes if gene not in top_combo]
