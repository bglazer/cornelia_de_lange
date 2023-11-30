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
wt_data = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut_data = sc.read_h5ad(f'../../data/mutant_net.h5ad')

# %%
# Load the models
wt_tmstp = '20230607_165324'
wt_outdir = f'../../output/{wt_tmstp}'
mut_tmstp = '20230608_093734'
mut_outdir = f'../../output/{mut_tmstp}'
wt_state_dict = torch.load(f'{wt_outdir}/models/optimal_wildtype.torch')
mut_state_dict = torch.load(f'{mut_outdir}/models/optimal_mutant.torch')

# %%
wt_X = torch.tensor(wt_data.X.toarray()).float()
mut_X = torch.tensor(mut_data.X.toarray()).float()
# %%
unique_cell_types = set(wt_data.obs['cell_type']) | set(mut_data.obs['cell_type'])
cell_types = {c:i for i,c in enumerate(unique_cell_types)}
wt_proj = np.array(wt_data.obsm['X_pca'])
mut_proj = np.array(mut_data.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = wt_data.uns['pca_mean']
pca.components_ = wt_data.uns['PCs']
wt_proj = np.array(pca.transform(wt_X))[:,0:2]
mut_proj = np.array(pca.transform(mut_X))[:,0:2]
wt_T = wt_data.obsm['transition_matrix']
mut_T = mut_data.obsm['transition_matrix']

wt_V = util.velocity_vectors(wt_T, wt_X)
mut_V = util.velocity_vectors(mut_T, mut_X)
wt_V_emb = util.embed_velocity(wt_X, wt_V, lambda x: np.array(pca.transform(x)[:,0:2]))
mut_V_emb = util.embed_velocity(mut_X, mut_V, lambda x: np.array(pca.transform(x)[:,0:2]))

# %%
torch.set_num_threads(24)
# 
num_nodes = wt_X.shape[1]
hidden_dim = 64
num_layers = 3

wt_model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers,
                         predict_var=True)
wt_model.load_state_dict(wt_state_dict)
mut_model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers,
                         predict_var=True)
mut_model.load_state_dict(mut_state_dict)

#%%
device='cuda:0'
wt_model = wt_model.to(device).eval()
wt_X = wt_X.to(device)
mut_model = mut_model.to(device).eval()
mut_X = mut_X.to(device)
wt_Xnp = util.tonp(wt_X)
mut_Xnp = util.tonp(mut_X)
#%%
sorted_cell_types = sorted(cell_types.keys())
cell_type_to_idx = {k:i for i,k in enumerate(sorted_cell_types)}
cell_type_list = list(cell_type_to_idx.keys())
wt_data_cell_proportions = wt_data.obs['cell_type'].value_counts()/wt_data.shape[0]
wt_data_cell_proportions = wt_data_cell_proportions[cell_type_list].to_numpy()
mut_data_cell_proportions = mut_data.obs['cell_type'].value_counts()/mut_data.shape[0]
mut_data_cell_proportions = mut_data_cell_proportions[cell_type_list].to_numpy()
#%%
n_repeats = 10
len_trajectory = 107
step_size = 2
n_steps = len_trajectory * step_size
#%%
t_span = torch.linspace(0, len_trajectory, n_steps, device=device)
pct_initial_cells = .10
wt_nmp_cell_population = np.where(wt_data.obs['cell_type'] == 'NMP')[0]
wt_n_initial_nmp = int(wt_nmp_cell_population.shape[0] * pct_initial_cells)
wt_nmp_initial_idxs = np.random.choice(wt_nmp_cell_population, size=wt_n_initial_nmp, replace=False)
wt_initial = wt_nmp_initial_idxs
# wt_initial = wt_data.uns['initial_points_via']

mut_nmp_cell_population = np.where(mut_data.obs['cell_type'] == 'NMP')[0]
mut_n_initial_nmp = int(mut_nmp_cell_population.shape[0] * pct_initial_cells)
mut_nmp_initial_idxs = np.random.choice(mut_nmp_cell_population, size=mut_n_initial_nmp, replace=False)
mut_initial = mut_nmp_initial_idxs
# mut_initial = mut_data.uns['initial_points_via']

wt_repeats = torch.tensor(wt_initial.repeat(n_repeats)).to(device)
mut_repeats = torch.tensor(mut_initial.repeat(n_repeats)).to(device)
#%%
wt_simulator = Simulator(wt_model, wt_X, device=device, boundary=True, pearson=False)
mut_simulator = Simulator(mut_model, mut_X, device=device, boundary=True, pearson=False)
wt_trajectories, wt_nearest_idxs = wt_simulator.simulate(wt_repeats, t_span) 
mut_trajectories, mut_nearest_idxs = mut_simulator.simulate(mut_repeats, t_span)

wt_pearson_simulator = Simulator(wt_model, wt_X, device=device, boundary=True, pearson=True)
mut_pearson_simulator = Simulator(mut_model, mut_X, device=device, boundary=True, pearson=True)
wt_pearson_trajectories, wt_pearson_nearest_idxs = wt_simulator.simulate(wt_repeats, t_span) 
mut_pearson_trajectories, mut_pearson_nearest_idxs = mut_simulator.simulate(mut_repeats, t_span)


#%%
all_trajectories = []
all_idxs = []
all_diffs = []

wt_trajectories_np = util.tonp(wt_trajectories)
wt_idxs_np = util.tonp(wt_nearest_idxs)
mut_trajectories_np = util.tonp(mut_trajectories)
mut_idxs_np = util.tonp(mut_nearest_idxs)
wt_pearson_trajectories_np = util.tonp(wt_pearson_trajectories)
wt_pearson_idxs_np = util.tonp(wt_pearson_nearest_idxs)
mut_pearson_trajectories_np = util.tonp(mut_pearson_trajectories)
mut_pearson_idxs_np = util.tonp(mut_pearson_nearest_idxs)
trajectories = plotting.compare_cell_type_trajectories([wt_idxs_np, mut_idxs_np, wt_pearson_idxs_np, mut_pearson_idxs_np],
                                                        data=[wt_data, mut_data, wt_data, mut_data],
                                                        cell_type_to_idx=cell_type_to_idx, 
                                                        labels=['WT Euclidean', 'Mut Euclidean', 'WT Pearson', 'Mut Pearson'])
all_trajectories.append(trajectories)
plt.show()

num_cells_in_trajectories = np.array([len(wt_initial) * n_repeats * n_steps,
                                        len(mut_initial) * n_repeats * n_steps])

data_proportions = np.stack((wt_data_cell_proportions, 
                                mut_data_cell_proportions))
labels = ['Wildtype Euclidean', 'Mutant Euclidean', 'Wildtype Pearson', 'Mutant Pearson']
wt_cell_proportions, wt_cell_proportions_error = plotting.calculate_cell_type_proportion(wt_idxs_np, wt_data, cell_type_to_idx, n_repeats=n_repeats, error=True)
mut_cell_proportions, mut_cell_proportions_error = plotting.calculate_cell_type_proportion(mut_idxs_np, mut_data, cell_type_to_idx, n_repeats=n_repeats, error=True)
wt_pearson_cell_proportions, wt_pearson_cell_proportions_error = plotting.calculate_cell_type_proportion(wt_pearson_idxs_np, wt_data, cell_type_to_idx, n_repeats=n_repeats, error=True)
mut_pearson_cell_proportions, mut_pearson_cell_proportions_error = plotting.calculate_cell_type_proportion(mut_pearson_idxs_np, mut_data, cell_type_to_idx, n_repeats=n_repeats, error=True)

proportions = np.stack((wt_cell_proportions, mut_cell_proportions, 
                            wt_pearson_cell_proportions, mut_pearson_cell_proportions))
proportion_errors = np.stack((wt_cell_proportions_error, mut_cell_proportions_error,
                                  wt_pearson_cell_proportions_error, mut_pearson_cell_proportions_error))

plotting.cell_type_proportions(proportions, proportion_errors, cell_type_to_idx, labels)
plt.show()
#%%
plotting.cell_type_distribution(wt_trajectories_np, wt_idxs_np, wt_data, cell_type_to_idx, pca, label='WT Euclidean')

# %%
plotting.cell_type_distribution(wt_pearson_trajectories_np, wt_pearson_idxs_np, wt_data, cell_type_to_idx, pca, label='WT Pearson')
# %%
