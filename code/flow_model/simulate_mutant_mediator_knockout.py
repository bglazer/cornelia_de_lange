#%%
%load_ext autoreload
%autoreload 2
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

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# %%
# wt = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut = sc.read_h5ad(f'../../data/mutant_net.h5ad')
# %%
# Load the models
# tmstp = '20230607_165324'  
# genotype = 'wildtype'
mut_tmstp = '20230608_093734'

genotype = 'mutant'
outdir = f'../../output/{mut_tmstp}'
state_dict = torch.load(f'{outdir}/models/optimal_{genotype}.torch')

# %%
X = torch.tensor(mut.X.toarray()).float()
# %%
cell_types = {c:i for i,c in enumerate(set(mut.obs['cell_type']))}
proj = np.array(mut.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = mut.uns['pca_mean']
pca.components_ = mut.uns['PCs']
proj = np.array(pca.transform(X))[:,0:2]
T = mut.obsm['transition_matrix']

V = util.velocity_vectors(T, X)
V_emb = util.embed_velocity(X, V, lambda x: np.array(pca.transform(x)[:,0:2]))

# %%
torch.set_num_threads(24)
start_idxs = mut.uns['initial_points_via']
device='cpu'
num_nodes = X.shape[1]
hidden_dim = 64
num_layers = 3

model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers,
                         predict_var=True)
model.load_state_dict(state_dict)

#%%
device='cuda:0'
model = model.to(device)
X = X.to(device)
Xnp = util.tonp(X)
simulator = Simulator(model, X, device=device)

#%%
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}


#%%
n_repeats = 10
start_idxs = mut.uns['initial_points_via']
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
len_trajectory = 98
n_steps = len_trajectory*4
t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{outdir}/baseline_trajectories_mutant.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{outdir}/baseline_nearest_cell_idxs_mutant.pickle', 'rb'))
baseline_velo,_ = plotting.compute_velo(model=model, X=X, numpy=True)
#%%
pltdir = '../../figures/perturbation_simulations/mutant'

#%%
mediator_targets = pickle.load(open(f'{outdir}/mediator_targets_mutant.pickle', 'rb'))
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
 #%%
for mediator, targets in mediator_targets.items():
    targets.append(mediator)
    print('Mediator', protein_id_name[mediator])
    target_str = ', '.join([protein_id_name[t] for t in targets])
    print('Targets:', fill(target_str, break_long_words=False))
    target_idxs = [node_to_idx[t] for t in targets]
    perturbation = (target_idxs, 
                    torch.zeros(len(targets), device=device))

    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats, t_span, 
                                                                    perturbation=perturbation, 
                                                                    boundary=False)

    mediator_name = protein_id_name[mediator]
    perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    plotting.distribution(perturb_trajectories_np[:,::10], pca,
                          label=f'{mediator_name} Targets Knockout',
                          baseline=Xnp)
    plt.savefig(f'{pltdir}/{mediator_name}_perturbed_distribution.png')
    plt.close()
    velo,_ = plotting.compute_velo(model=model, X=X, perturbation=perturbation, numpy=True)
    plotting.arrow_grid(velos=[velo, baseline_velo], 
                        data=[mut, mut], 
                        pca=pca, 
                        labels=[f'{mediator_name} Targets Knockout', 'Mutant baseline'])
    plt.savefig(f'{pltdir}/{mediator_name}_perturbed_arrow_grid.png')
    plt.close()
    plotting.sample_trajectories(perturb_trajectories_np, Xnp, pca, f'{mediator_name} Targets Knockout')
    plt.savefig(f'{pltdir}/{mediator_name}_perturbed_trajectories.png')
    plt.close()
    trajectories = plotting.compare_cell_type_trajectories([perturb_idxs_np, baseline_idxs],
                                                           [mut, mut], 
                                                           cell_types,
                                                           [mediator_name, 'baseline'])
    mut_cell_type_traj, perturb_cell_type_traj = trajectories
    plt.savefig(f'{pltdir}/{mediator_name}_perturbed_cell_type_trajectories.png')
    plt.close()
    mut_ct = mut.obs['cell_type'].value_counts()/mut.shape[0]
    mut_ct = mut_ct[sorted(cell_types)]
    mut_total_cell_types = mut_cell_type_traj.sum(axis=1)/mut_cell_type_traj.sum()
    perturb_total_cell_types = perturb_cell_type_traj.sum(axis=1)/perturb_cell_type_traj.sum()
    # Plot side by side bar charts of the cell type proportions
    n_cells = (len(start_idxs) * n_repeats * n_steps)
    perturb_cell_type_traj = trajectories[0]
    baseline_cell_type_traj = trajectories[1]
    perturb_cell_proportions = perturb_cell_type_traj.sum(axis=1) / n_cells
    baseline_cell_proportions = baseline_cell_type_traj.sum(axis=1) / n_cells
    plotting.cell_type_proportions(proportions=(perturb_cell_proportions, 
                                                baseline_cell_proportions), 
                                   cell_types=cell_types, 
                                   labels=[f'{mediator_name} Knockout', 'Mutant baseline'])
    plt.savefig(f'{pltdir}/{mediator_name}_perturbed_cell_type_proportions.png',
                bbox_inches='tight');
    plt.close();

# %%
