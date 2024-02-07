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
import glob

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
label = ''
# label = 'VIM_first_'
transfer_dir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations'
pltdir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/figures'
datadir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/data'

#%%
best_gene_combinations = []
for file in glob.glob(f'{datadir}/top_{label}{transfer}_combination*.pickle'):
    combo = pickle.load(open(file, 'rb'))
    best_gene_combinations.append(combo)

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
# Load utility data structures for gene names and indexes
cell_types = {c:i for i,c in enumerate(sorted(set(tgt_data.obs['cell_type'])))}
idx_to_cell_type = {v:k for k,v in cell_types.items()}
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
step_size = tgt_data.uns['best_step_size']
n_steps = step_size*len_trajectory

t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{tgt_outdir}/baseline_trajectories_{target_genotype}.pickle', 'rb'))
baseline_idxs = pickle.load(open(f'{tgt_outdir}/baseline_nearest_cell_idxs_{target_genotype}.pickle', 'rb'))
# baseline_velo,_ = plotting.compute_velo(model=model, X=src_X, numpy=True)
# baseline_X = baseline_trajectories_np.reshape(-1, num_nodes)
baseline_cell_proportions, baseline_cell_errors = plotting.calculate_cell_type_proportion(baseline_idxs, tgt_data, cell_types, n_repeats, error=True)
#%%
def transfer_genes_to_model(transfer_genes, device):
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
    return tgt_model

def reset_model(device):
    tgt_model = GroupL1FlowModel(input_dim=num_nodes, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=num_layers,
                                 predict_var=True)
    tgt_model.load_state_dict(target_state_dict)
    tgt_model = tgt_model.to(device)
    return tgt_model

def simulate(tgt_model, idxs_0, t_span, device):
    simulator = Simulator(tgt_model, tgt_X.to(device), device=device, boundary=False, show_progress=False)
    idxs0_gpu = idxs_0.to(device)
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(idxs0_gpu, t_span)
    # Delete full trajectories so that we can free the GPU memory
    del perturb_trajectories
    # Tell the memory manager to free the GPU memory
    torch.cuda.empty_cache()

    return perturb_nearest_idxs

#%%
num_gpus = 4
def simulate_temporal_rescue(t, transfer_genes, idx, label, save=True):
    # For testing purposes return a random cell type proportion of the same shape as the baseline
    # return transfer_genes, np.random.rand(*baseline_cell_proportions.shape), np.random.rand(*baseline_cell_proportions.shape)
    gpu = idx % num_gpus
    device = f'cuda:{gpu}'
    tgt_model = transfer_genes_to_model(transfer_genes, device)

    steps_0 = int(t*step_size)
    steps_1 = n_steps - steps_0

    t_span_0 = torch.linspace(0, t, steps_0, device=device)
    t_span_1 = torch.linspace(t, len_trajectory, steps_1, device=device)
    
    idxs_0 = simulate(tgt_model, repeats, t_span_0, device)
    tgt_model = reset_model(device)
    start_1 = idxs_0[-1]
    idxs_1 = simulate(tgt_model, start_1, t_span_1, device)

    idxs_0_np = util.tonp(idxs_0)
    idxs_1_np = util.tonp(idxs_1)
    del idxs_0
    del idxs_1
    torch.cuda.empty_cache()

    perturb_idxs_np = np.concatenate([idxs_0_np, idxs_1_np], axis=0)

    perturb_cell_proportions, perturb_cell_errors = plotting.calculate_cell_type_proportion(perturb_idxs_np, tgt_data, cell_types, n_repeats, error=True)
    
    proportion_dict = {'transfer_genes': transfer_genes,
                       'perturb_proportions':perturb_cell_proportions, 
                       'perturb_errors': perturb_cell_errors}
    if save:
        with open(f'{datadir}/{label}_combination_{idx}_{transfer}_transfer_cell_type_proportions.pickle', 'wb') as f:
            pickle.dump(proportion_dict, f)
    d = np.abs(perturb_cell_proportions - baseline_cell_proportions).sum()
    print(f'{d}, {" ".join([protein_id_name[g] for g in transfer_genes])}', flush=True)
    return t, transfer_genes, perturb_cell_proportions, perturb_cell_errors

#%%
# vim = protein_name_id['VIM']
# simulate_temporal_rescue(t=1, transfer_genes=[vim], idx=0, label='test', save=False)
#%%
results = {}
for combo in best_gene_combinations:
    print(len(combo), ','.join([protein_id_name[g] for g in combo]))
    parallel = Parallel(n_jobs=12, verbose=11)

    results[combo] = parallel(delayed(simulate_temporal_rescue)(t, combo, idx, label) 
                              for idx, t in enumerate(t_span[1::10]))
#%%
print('Done')
# %%
# Save the results
with open(f'{datadir}/{label}{transfer}_critical_period_proportions.pickle', 'wb') as f:
    pickle.dump(results, f)


# %%
