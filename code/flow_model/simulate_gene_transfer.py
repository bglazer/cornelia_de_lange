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
transfer_dir = f'{tgt_outdir}/{transfer}_transfer_simulations'
pltdir = f'{tgt_outdir}/{transfer}_transfer_simulations/figures'
datadir = f'{tgt_outdir}/{transfer}_transfer_simulations/data'

#%%
# Check if transfer_dir exists. This is where we will save the transfer simulations and figures
import os
if not os.path.exists(transfer_dir):
    os.makedirs(transfer_dir)
    os.makedirs(pltdir)
    os.makedirs(datadir)

source_state_dict = torch.load(f'{src_outdir}/models/optimal_{source_genotype}.torch')
target_state_dict = torch.load(f'{tgt_outdir}/models/optimal_{target_genotype}.torch')

# %%
X = torch.tensor(tgt_data.X.toarray()).float()
Xnp = util.tonp(X)
proj = np.array(tgt_data.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = tgt_data.uns['pca_mean']
pca.components_ = tgt_data.uns['PCs']
proj = np.array(pca.transform(X))[:,0:2]
T = tgt_data.obsm['transition_matrix']

V = util.velocity_vectors(T, X)
V_emb = util.embed_velocity(X, V, lambda x: np.array(pca.transform(x)[:,0:2]))
# %%
torch.set_num_threads(24)
device='cpu'
num_nodes = X.shape[1]
hidden_dim = 64
num_layers = 3

#%%
cell_types = {c:i for i,c in enumerate(sorted(set(tgt_data.obs['cell_type'])))}

#%%
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}

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
model.load_state_dict(target_state_dict)
model = model.to(device)
Xgpu = X.to(device)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{tgt_outdir}/baseline_trajectories_{target_genotype}.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{tgt_outdir}/baseline_nearest_cell_idxs_{target_genotype}.pickle', 'rb'))
baseline_velo,_ = plotting.compute_velo(model=model, X=X, numpy=True)
#%%
plotting.time_distribution(baseline_trajectories_np[:,:], pca,
                           label=f'Baseline {target_genotype} - no transfer',
                           baseline=Xnp)
# plt.show()
plt.savefig(f'{pltdir}/baseline_{target_genotype}_time_distribution.png', bbox_inches='tight')
plt.close()
#%%
plotting.cell_type_distribution(baseline_trajectories_np[:,:], 
                                baseline_idxs[:,:],
                                tgt_data,
                                cell_types,
                                pca,
                                label=f'Baseline {target_genotype} - no transfer',
                                baseline=Xnp)
# plt.show()
plt.savefig(f'{pltdir}/baseline_{target_genotype}_cell_type_distribution.png', bbox_inches='tight')
plt.close()
#%%
plotting.cell_type_distribution(np.expand_dims(Xnp, axis=1), 
                                np.expand_dims(np.arange(Xnp.shape[0]), axis=1),
                                tgt_data,
                                cell_types,
                                pca,
                                label=f'Cell Type Distribution - Data',
                                baseline=None,
                                s=20)
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
all_genes = set(node_to_idx.keys())

num_gpus = 4

#%%
def simulate_transfer(transfer_gene, i, repeats):
    print(f'Transfer Gene {protein_id_name[transfer_gene]:10s}: ({i+1}/{len(all_genes)})', flush=True)
    
    gpu = i % num_gpus
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
    model_idx = node_to_idx[transfer_gene]
    src_gene_model = src_model.models[model_idx]
    tgt_gene_model = tgt_model.models[model_idx]
    tgt_gene_model.load_state_dict(src_gene_model.state_dict())
    Xgpu = X.to(device)
    simulator = Simulator(tgt_model, Xgpu, device=device, boundary=False, show_progress=False)

    repeats = repeats.to(device)
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats, t_span)
    transfer_gene_name = protein_id_name[transfer_gene]
    perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    # Aggregate the individual cell trajectories by mean
    mean_trajectories = perturb_trajectories.mean(dim=1)
    # Save the mean trajectories
    with open(f'{datadir}/{transfer_gene_name}_{transfer}_transfer_mean_trajectories.pickle', 'wb') as f:
        pickle.dump(mean_trajectories, f)

    plotting.time_distribution(perturb_trajectories_np[:,:], pca,
                               label=f'{transfer_gene_name} Transfer',
                               baseline=Xnp)
    plt.savefig(f'{pltdir}/{transfer_gene_name}_{transfer}_transfer_time_distribution.png',
                bbox_inches='tight')
    plt.close()
    plotting.cell_type_distribution(perturb_trajectories_np[:,:], 
                                    perturb_idxs_np[:,:],
                                    tgt_data,
                                    cell_types,
                                    pca,
                                    label=f'{transfer_gene_name} Transfer',
                                    baseline=Xnp)
    plt.savefig(f'{pltdir}/{transfer_gene_name}_{transfer}_transfer_cell_type_distribution.png',
                bbox_inches='tight')
    plt.close()
    velo,_ = plotting.compute_velo(model=simulator.model, X=Xgpu, perturbation=None, numpy=True)
    plotting.arrow_grid(velos=[velo, baseline_velo], 
                        data=[tgt_data, tgt_data], 
                        pca=pca, 
                        labels=[f'{transfer_gene_name} Transfer', f'{target_genotype.capitalize()} baseline'])
    plt.savefig(f'{pltdir}/{transfer_gene_name}_{transfer}_transfer_arrow_grid.png')
    plt.close()
    plotting.sample_trajectories(perturb_trajectories_np, Xnp, pca, f'{transfer_gene_name} Transfer')
    plt.savefig(f'{pltdir}/{transfer_gene_name}_{transfer}_transfer_trajectories.png')
    plt.close()
    trajectories = plotting.compare_cell_type_trajectories([perturb_idxs_np, baseline_idxs],
                                                           [tgt_data, tgt_data], 
                                                           cell_types,
                                                           [transfer_gene_name, 'baseline'])
    # Save the trajectories
    with open(f'{datadir}/{transfer_gene_name}_{transfer}_transfer_cell_type_trajectories.pickle', 'wb') as f:
        pickle.dump(trajectories, f)
    plt.savefig(f'{pltdir}/{transfer_gene_name}_{transfer}_transfer_cell_type_trajectories.png')
    plt.close()
    # Plot side by side bar charts of the cell type proportions
    perturb_cell_proportions, perturb_cell_errors = plotting.calculate_cell_type_proportion(perturb_idxs_np, tgt_data, cell_types, n_repeats, error=True)
    baseline_cell_proportions, baseline_cell_errors = plotting.calculate_cell_type_proportion(baseline_idxs, tgt_data, cell_types, n_repeats, error=True)
    # Save the cell type proportions
    plotting.cell_type_proportions(proportions=(perturb_cell_proportions, 
                                                baseline_cell_proportions), 
                                   proportion_errors=(perturb_cell_errors,
                                           baseline_cell_errors),
                                   cell_types=list(cell_types), 
                                   labels=[f'{transfer_gene_name} Transfer', f'{target_genotype.capitalize()} baseline'])
    plt.savefig(f'{pltdir}/{transfer_gene_name}_{transfer}_transfer_cell_type_proportions.png',
                bbox_inches='tight');
    plt.close();
    with open(f'{datadir}/{transfer_gene_name}_{transfer}_transfer_cell_type_proportions.pickle', 'wb') as f:
        pickle.dump((perturb_cell_proportions, perturb_cell_errors, baseline_cell_proportions), f)

#%%
# simulate_transfer(transfer_gene='ENSMUSP00000134654', i=0, repeats=repeats)

# %%
_ = Parallel(n_jobs=8)(delayed(simulate_transfer)(transfer_gene, i, repeats) for i,transfer_gene in enumerate(all_genes))

# %%
