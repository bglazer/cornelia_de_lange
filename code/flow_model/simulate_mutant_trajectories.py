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
#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# %%
genotype = 'mutant'
data = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')
# %%
# Load the models
tmstp = '20230608_093734'
outdir = f'../../output/{tmstp}'
state_dict = torch.load(f'{outdir}/models/optimal_mutant.torch')

# %%
X = torch.tensor(data.X.toarray()).float()

# %%
cell_types = {c:i for i,c in enumerate(set(data.obs['cell_type']))}
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
simulator = Simulator(model, X, device=device)
#%%
n_repeats = 10
len_trajectory = 250
n_steps = 250*4
t_span = torch.linspace(0, len_trajectory, n_steps, device=device)
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
mut_trajectories, mut_nearest_idxs = simulator.simulate(repeats, t_span)
#%%
mut_trajectories_np = util.tonp(mut_trajectories)
mut_idxs_np = util.tonp(mut_nearest_idxs)
Xnp = util.tonp(X)
#%%
plotting.distribution(mut_trajectories_np, pca, 'Mutant')
#%%
plotting.arrow_grid(data, pca, model, 'Mutant', perturbation=None, device=device)
# %%
plotting.sample_trajectories(mut_trajectories_np, Xnp, pca, 'Mutant')
#%%
mut_cell_type_traj = plotting.cell_type_trajectories(mut_idxs_np, data, 'Mutant')
# %%
def overall_cell_type_proportions():
    mut_total_cell_types = mut_cell_type_traj.sum(axis=1)/mut_cell_type_traj.sum()
    # Plot side by side bar charts of the cell type proportions
    plt.bar(np.arange(len(mut_total_cell_types)), mut_total_cell_types, label='Mutant', width=.4)
    plt.xticks(np.arange(len(mut_total_cell_types)), data.obs['cell_type'].cat.categories, rotation=90);
    plt.ylabel('Proportion of cells')
    plt.legend()

overall_cell_type_proportions()
# %%
