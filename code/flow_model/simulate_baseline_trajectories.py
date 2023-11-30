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
# Group observations by cell line and calculate cell proportion 
# Then calculate stdev across the cell lines
cell_lines = wt_data.obs['cell_line'].unique()
wt_sim_cell_proportions = np.zeros((len(cell_lines), len(cell_type_list)))
for i,cell_line in enumerate(cell_lines):
    wt_cell_line = wt_data[wt_data.obs['cell_line'] == cell_line]
    wt_sim_cell_proportions[i] = wt_cell_line.obs['cell_type'].value_counts()/wt_cell_line.shape[0]
wt_data_cell_proportions_error = wt_sim_cell_proportions.std(axis=0)
cell_lines = mut_data.obs['cell_line'].unique()
mut_sim_cell_proportions = np.zeros((len(cell_lines), len(cell_type_list)))
for i,cell_line in enumerate(cell_lines):
    mut_cell_line = mut_data[mut_data.obs['cell_line'] == cell_line]
    mut_sim_cell_proportions[i] = mut_cell_line.obs['cell_type'].value_counts()/mut_cell_line.shape[0]
mut_data_cell_proportions_error = mut_sim_cell_proportions.std(axis=0)
data_proportions_error = np.stack((wt_data_cell_proportions_error, mut_data_cell_proportions_error))
#%%
n_repeats = 10
len_trajectory = np.linspace(start=85, stop=125, num=10, dtype=int)
step_size = 4
n_steps = [l*step_size for l in len_trajectory]
#%%
t_span = torch.linspace(0, len_trajectory[-1], n_steps[-1], device=device)
pct_initial_cells = .20
wt_nmp_cell_population = np.where(wt_data.obs['cell_type'] == 'NMP')[0]
wt_mmp_cell_population = np.where(wt_data.obs['cell_type'] == 'MMP')[0]
wt_n_initial_nmp = int(wt_nmp_cell_population.shape[0] * pct_initial_cells)
wt_n_initial_mmp = int(wt_mmp_cell_population.shape[0] * pct_initial_cells)
wt_nmp_initial_idxs = np.random.choice(wt_nmp_cell_population, size=wt_n_initial_nmp, replace=False)
wt_mmp_initial_idxs = np.random.choice(wt_mmp_cell_population, size=wt_n_initial_mmp, replace=False)
# wt_initial = np.concatenate((wt_nmp_initial_idxs, wt_mmp_initial_idxs))
wt_initial = wt_nmp_initial_idxs
wt_data.uns['initial_points_nmp'] = wt_initial

mut_nmp_cell_population = np.where(mut_data.obs['cell_type'] == 'NMP')[0]
mut_mmp_cell_population = np.where(mut_data.obs['cell_type'] == 'MMP')[0]
mut_n_initial_nmp = int(mut_nmp_cell_population.shape[0] * pct_initial_cells)
mut_n_initial_mmp = int(mut_mmp_cell_population.shape[0] * pct_initial_cells)
mut_nmp_initial_idxs = np.random.choice(mut_nmp_cell_population, size=mut_n_initial_nmp, replace=False)
mut_mmp_initial_idxs = np.random.choice(mut_mmp_cell_population, size=mut_n_initial_mmp, replace=False)
# mut_initial = np.concatenate((mut_nmp_initial_idxs, mut_mmp_initial_idxs))
mut_initial = mut_nmp_initial_idxs
mut_data.uns['initial_points_nmp'] = mut_initial
#%%
# Save the scanpy object with the initial points as an h5ad file
wt_data.write(f'../../data/wildtype_net.h5ad')
mut_data.write(f'../../data/mutant_net.h5ad')
#%%
wt_repeats = torch.tensor(wt_initial.repeat(n_repeats)).to(device)
mut_repeats = torch.tensor(mut_initial.repeat(n_repeats)).to(device)
#%%
wt_simulator = Simulator(wt_model, wt_X, device=device, boundary=True, pearson=False)
mut_simulator = Simulator(mut_model, mut_X, device=device, boundary=True, pearson=False)
#%%
wt_trajectories, wt_nearest_idxs = wt_simulator.simulate(wt_repeats, t_span) 
mut_trajectories, mut_nearest_idxs = mut_simulator.simulate(mut_repeats, t_span)

#%%
all_trajectories = []
all_idxs = []
all_diffs = []

bar_colors = ['#004d80', '#b16e02', '#0099ff', '#fda61c']
for i in range(len(len_trajectory)):
    print('-'*40)
    print('Trajectory length:', len_trajectory[i])
    print('-'*40)

    wt_trajectories_np = util.tonp(wt_trajectories[:n_steps[i]])#:step_size])
    wt_idxs_np = util.tonp(wt_nearest_idxs[:n_steps[i]])#:step_size])
    mut_trajectories_np = util.tonp(mut_trajectories[:n_steps[i]])#:step_size])
    mut_idxs_np = util.tonp(mut_nearest_idxs[:n_steps[i]])#:step_size])
    trajectories = plotting.compare_cell_type_trajectories([wt_idxs_np, mut_idxs_np], 
                                                           data=[wt_data, mut_data],
                                                           cell_type_to_idx=cell_type_to_idx, 
                                                           labels=['Wildtype', 'Mutant'])
    plt.close()
    all_trajectories.append(trajectories)
    
    num_cells_in_trajectories = np.array([len(wt_initial) * n_repeats * n_steps[i],
                                          len(mut_initial) * n_repeats * n_steps[i]])

    data_proportions = np.stack((wt_data_cell_proportions, 
                                 mut_data_cell_proportions))
    labels = ['Wildtype Sim.', 'Mutant Sim.', 'Wildtype Data',  'Mutant Data']
    wt_sim_cell_proportions, wt_sim_cell_proportions_error = plotting.calculate_cell_type_proportion(wt_idxs_np, wt_data, cell_type_to_idx, n_repeats=n_repeats, error=True)
    mut_sim_cell_proportions, mut_sim_cell_proportions_error = plotting.calculate_cell_type_proportion(mut_idxs_np, mut_data, cell_type_to_idx, n_repeats=n_repeats, error=True)
    sim_proportions = np.stack((wt_sim_cell_proportions, mut_sim_cell_proportions))
    proportions = np.stack((wt_sim_cell_proportions, mut_sim_cell_proportions,
                            wt_data_cell_proportions, mut_data_cell_proportions))
    sim_proportion_errors = np.stack((wt_sim_cell_proportions_error, mut_sim_cell_proportions_error))
    proportion_errors = np.stack((wt_sim_cell_proportions_error, mut_sim_cell_proportions_error,
                                  wt_data_cell_proportions_error, mut_data_cell_proportions_error))
    plotting.cell_type_proportions(proportions, proportion_errors, cell_type_to_idx, labels, colors=bar_colors)
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
wt_data.uns['best_trajectory_length'] = best_len
mut_data.uns['best_trajectory_length'] = best_len
wt_data.uns['best_step_size'] = step_size
mut_data.uns['best_step_size'] = step_size
#%%
plotting.time_distribution(wt_trajectories_np[:,::10], pca, 'Wildtype', baseline=wt_Xnp)
#%%
plotting.time_distribution(mut_trajectories_np[:,::10], pca, 'Mutant', baseline=mut_Xnp)
#%%
plotting.cell_type_distribution(wt_trajectories_np, wt_idxs_np, wt_data, cell_type_to_idx, pca, 'Wildtype', baseline=wt_Xnp)
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
pickle.dump(wt_trajectories_np[:n_steps[best_idx]], open(f'{wt_outdir}/baseline_trajectories_wildtype.pickle', 'wb'))
pickle.dump(mut_trajectories_np[:n_steps[best_idx]], open(f'{mut_outdir}/baseline_trajectories_mutant.pickle', 'wb'))
# %%
