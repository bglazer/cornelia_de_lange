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
tmstp = '20230608_093734'
wt_outdir = f'../../output/{tmstp}'
mut_outdir = f'../../output/{tmstp}'
wt_state_dict = torch.load(f'{wt_outdir}/models/optimal_mutant.torch')
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
wt_start_idxs = wt_data.uns['initial_points_via']
mut_start_idxs = mut_data.uns['initial_points_via']

device='cpu'
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
wt_simulator = Simulator(wt_model, wt_X, device=device)
mut_model = mut_model.to(device).eval()
mut_X = mut_X.to(device)
mut_simulator = Simulator(mut_model, mut_X, device=device)
wt_Xnp = util.tonp(wt_X)
mut_Xnp = util.tonp(mut_X)
#%%
n_repeats = 10
len_trajectory = np.linspace(start=85, stop=125, num=10, dtype=int)
# TODO change back to 250*4
n_steps = [l*4 for l in len_trajectory]
#%%
t_span = torch.linspace(0, len_trajectory[-1], n_steps[-1], device=device)
wt_repeats = torch.tensor(wt_start_idxs.repeat(n_repeats)).to(device)
mut_repeats = torch.tensor(mut_start_idxs.repeat(n_repeats)).to(device)
wt_trajectories, wt_nearest_idxs = wt_simulator.simulate(wt_repeats, t_span)
mut_trajectories, mut_nearest_idxs = mut_simulator.simulate(mut_repeats, t_span)
#%%
sorted_cell_types = sorted(cell_types.keys())
wt_data_cell_proportions = wt_data.obs['cell_type'].value_counts()/wt_data.shape[0]
wt_data_cell_proportions = wt_data_cell_proportions[sorted_cell_types].to_numpy()
mut_data_cell_proportions = mut_data.obs['cell_type'].value_counts()/mut_data.shape[0]
mut_data_cell_proportions = mut_data_cell_proportions[sorted_cell_types].to_numpy()
#%%
all_trajectories = []
all_idxs = []
all_diffs = []
for i in range(len(len_trajectory)):
    print('-'*40)
    print('Trajectory length:', len_trajectory[i])
    print('-'*40)

    wt_trajectories_np = util.tonp(wt_trajectories[:n_steps[i]])
    wt_idxs_np = util.tonp(wt_nearest_idxs[:n_steps[i]])
    mut_trajectories_np = util.tonp(mut_trajectories[:n_steps[i]])
    mut_idxs_np = util.tonp(mut_nearest_idxs[:n_steps[i]])
    trajectories = plotting.compare_cell_type_trajectories([wt_idxs_np, mut_idxs_np], 
                                                           data=[wt_data, mut_data],
                                                           cell_type_to_idx=cell_types, 
                                                           labels=['Wildtype', 'Mutant'])
    all_trajectories.append(trajectories)
    plt.show()
    
    num_cells_in_trajectories = np.array([len(wt_start_idxs) * n_repeats * n_steps[i],
                                          len(mut_start_idxs) * n_repeats * n_steps[i]])
    sim_proportions = (trajectories.sum(axis=-1).T/num_cells_in_trajectories).T

    sorted_cell_type_idxs = np.argsort(list(cell_types.keys()))
    data_proportions = np.stack((wt_data_cell_proportions, 
                                 mut_data_cell_proportions))
    sim_proportions = sim_proportions[:, sorted_cell_type_idxs]
    proportions = np.concatenate((sim_proportions, data_proportions))
    labels = ['Wildtype Sim.', 'Mutant Sim.', 'Wildtype Data', 'Mutant Data']
    plotting.cell_type_proportions(proportions, sorted_cell_types, labels)
    plt.show()
    diff = np.abs(sim_proportions - data_proportions).sum()
    all_diffs.append(diff)
    print('Overall difference in proportions:', diff)
    print('Mutant and WT differences:        ', np.abs(sim_proportions - data_proportions).sum(axis=1))


print('-'*40)
#%%
# Find the best trajectory length, i.e. the one with the mimimum difference in proportions
best_idx = np.argmin(all_diffs)
best_len = len_trajectory[best_idx]
print('Best trajectory length:', best_len)
#%%
plotting.distribution(wt_trajectories_np[:,::10], pca, 'Wildtype', baseline=wt_Xnp)
#%%
plotting.distribution(mut_trajectories_np[:,::10], pca, 'Mutant', baseline=mut_Xnp)
#%%
wt_velos = plotting.compute_velo(wt_model, wt_X)[0].detach().cpu().numpy()
mut_velos = plotting.compute_velo(mut_model, mut_X)[0].detach().cpu().numpy()
velos = (wt_velos, mut_velos)
#%%
plotting.arrow_grid(velos, 
                    data=(wt_data, mut_data),
                    pca=pca, 
                    labels=['Wildtype', 'Mutant'])
# %%
plotting.sample_trajectories(wt_trajectories_np, wt_Xnp, pca, 'Wildtype')
# %%
plotting.sample_trajectories(mut_trajectories_np, mut_Xnp, pca, 'Mutant')

# %%
# Save the best trajectories
wt_best_trajectories, mut_best_trajectories = all_trajectories[best_idx]
pickle.dump(wt_best_trajectories, open(f'{wt_outdir}/baseline_cell_type_trajectories_wildtype.pickle', 'wb'))
pickle.dump(mut_best_trajectories, open(f'{mut_outdir}/baseline_cell_type_trajectories_mutant.pickle', 'wb'))
pickle.dump(wt_idxs_np[:n_steps[best_idx]], open(f'{wt_outdir}/baseline_nearest_cell_idxs_wildtype.pickle', 'wb'))
pickle.dump(mut_idxs_np[:n_steps[best_idx]], open(f'{mut_outdir}/baseline_nearest_cell_idxs_mutant.pickle', 'wb'))
# %%
pickle.dump(wt_trajectories_np[:n_steps[best_idx]], open(f'{wt_outdir}/baseline_trajectories_wildtype_all.pickle', 'wb'))
pickle.dump(mut_trajectories_np[:n_steps[best_idx]], open(f'{mut_outdir}/baseline_trajectories_mutant_all.pickle', 'wb'))
# %%
