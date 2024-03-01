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
device='cpu'
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
wt_cell_proportions = np.zeros((len(cell_lines), len(cell_type_list)))
for i,cell_line in enumerate(cell_lines):
    wt_cell_line = wt_data[wt_data.obs['cell_line'] == cell_line]
    wt_cell_proportions[i] = wt_cell_line.obs['cell_type'].value_counts()/wt_cell_line.shape[0]
wt_cell_proportions_error = wt_cell_proportions.std(axis=0)
cell_lines = mut_data.obs['cell_line'].unique()
mut_cell_proportions = np.zeros((len(cell_lines), len(cell_type_list)))
for i,cell_line in enumerate(cell_lines):
    mut_cell_line = mut_data[mut_data.obs['cell_line'] == cell_line]
    mut_cell_proportions[i] = mut_cell_line.obs['cell_type'].value_counts()/mut_cell_line.shape[0]
mut_cell_proportions_error = mut_cell_proportions.std(axis=0)
data_proportions_error = np.stack((wt_cell_proportions_error, mut_cell_proportions_error))
#%%
n_repeats = 10
len_trajectory = wt_data.uns['best_trajectory_length']
step_size = wt_data.uns['best_step_size']
n_steps = len_trajectory*step_size 
#%%
t_span = torch.linspace(0, len_trajectory, n_steps, device=device)

wt_initial = wt_data.uns['initial_points_nmp']
mut_initial = mut_data.uns['initial_points_nmp']
#%%
wt_repeats = torch.tensor(wt_initial.repeat(n_repeats)).to(device)
mut_repeats = torch.tensor(mut_initial.repeat(n_repeats)).to(device)
#%%
wt_simulator = Simulator(wt_model, wt_X, device=device, boundary=False, pearson=False)
mut_simulator = Simulator(mut_model, mut_X, device=device, boundary=False, pearson=False)
#%%
wt_trajectories, wt_nearest_idxs = wt_simulator.simulate(wt_repeats, t_span) 
mut_trajectories, mut_nearest_idxs = mut_simulator.simulate(mut_repeats, t_span)
wt_trajectories_np = util.tonp(wt_trajectories)
wt_idxs_np = util.tonp(wt_nearest_idxs)
mut_trajectories_np = util.tonp(mut_trajectories)
mut_idxs_np = util.tonp(mut_nearest_idxs)
#%%
trajectories = plotting.compare_cell_type_trajectories([wt_idxs_np, mut_idxs_np], 
                                                        data=[wt_data, mut_data],
                                                        cell_type_to_idx=cell_type_to_idx, 
                                                        labels=['Wildtype', 'Mutant'])
#%%
diff = trajectories[0] - trajectories[1]
max_diff = max(np.abs(diff).max(), np.abs(diff).min())

plt.imshow(diff, 
           vmin=-max_diff, vmax=max_diff,
           aspect='auto', cmap='bwr',
           interpolation='none')
# Label the y-axis with the cell types
plt.yticks(np.arange(len(cell_type_list)), cell_type_list)
plt.xticks(np.arange(0, n_steps, n_steps//10), 
           np.arange(0, len_trajectory, len_trajectory//10))
plt.colorbar()
plt.title('Difference in cell type proportions across time')
#%%
num_cells_in_trajectories = np.array([len(wt_initial) * n_repeats * n_steps,
                                        len(mut_initial) * n_repeats * n_steps])

data_proportions = np.stack((wt_data_cell_proportions, 
                                mut_data_cell_proportions))
labels = ['Wildtype Sim.', 'Mutant Sim.', 'Wildtype Data', 'Mutant Data']
wt_cell_proportions, wt_cell_proportions_error = plotting.calculate_cell_type_proportion(wt_idxs_np, wt_data, cell_type_to_idx, n_repeats=n_repeats, error=True)
mut_cell_proportions, mut_cell_proportions_error = plotting.calculate_cell_type_proportion(mut_idxs_np, mut_data, cell_type_to_idx, n_repeats=n_repeats, error=True)
sim_proportions = np.stack((wt_cell_proportions, mut_cell_proportions))
proportions = np.concatenate((sim_proportions, data_proportions))
sim_proportion_errors = np.stack((wt_cell_proportions_error, mut_cell_proportions_error))
proportion_errors = np.concatenate((sim_proportion_errors, data_proportions_error))
# plotting.cell_type_proportions(proportions, proportion_errors, cell_type_to_idx, labels)
# plt.show()
diff = np.abs(sim_proportions - data_proportions).sum()
print('Overall difference in proportions:', diff)
print('Mutant and WT differences:        ', np.abs(sim_proportions - data_proportions).sum(axis=1))


#%%
plotting.time_distribution(wt_trajectories_np, pca, 'Wildtype', baseline=wt_Xnp)
#%%
plotting.time_distribution(mut_trajectories_np, pca, 'Mutant', baseline=mut_Xnp)
#%%
plotting.cell_type_distribution(wt_trajectories_np, wt_idxs_np, wt_data, cell_type_to_idx, pca, 'Wildtype', baseline=None)
#%%
wt_gene_trajectories = wt_trajectories_np.mean(axis=1)
mut_gene_trajectories = mut_trajectories_np.mean(axis=1)
wt_gene_std = np.std(wt_gene_trajectories, axis=0)
mut_gene_std = np.std(mut_gene_trajectories, axis=0)

#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}
#%%
# Get the expression of HOXB1
wt_hoxb1_mean = wt_gene_trajectories[:,node_to_idx[protein_name_id['HOXB1']]]
mut_hoxb1_mean = mut_gene_trajectories[:,node_to_idx[protein_name_id['HOXB1']]]
wt_hoxb1_std = wt_gene_std[node_to_idx[protein_name_id['HOXB1']]]
mut_hoxb1_std = mut_gene_std[node_to_idx[protein_name_id['HOXB1']]]

# %%
# Plot the mean HOXB1 expression at each time point
plt.plot(wt_hoxb1_mean, label='Wildtype')
plt.plot(mut_hoxb1_mean, label='Mutant')
plt.legend()
# # Plot the variance
# plt.fill_between(np.arange(len(wt_hoxb1_mean)), wt_hoxb1_mean - wt_hoxb1_std*2, wt_hoxb1_mean + wt_hoxb1_std*2, alpha=.5)
# plt.fill_between(np.arange(len(mut_hoxb1_mean)), mut_hoxb1_mean - mut_hoxb1_std*2, mut_hoxb1_mean + mut_hoxb1_std*2, alpha=.5)
#%%
hoxbs = ['HOXB1', 'HOXB2', 'HOXB4', 
         'HOXB5', 'HOXB6', 'HOXB7']

# %%
# Get the mean expression of each of the HOXB genes
wt_hoxb_mean = wt_gene_trajectories[:,[node_to_idx[protein_name_id[hoxb]] for hoxb in hoxbs]]
mut_hoxb_mean = mut_gene_trajectories[:,[node_to_idx[protein_name_id[hoxb]] for hoxb in hoxbs]]

# In side by side plots, show the mean expression of each of the HOXB genes across time
fig, axs = plt.subplots(len(hoxbs),1, figsize=(5,15))
for i, hoxb in enumerate(hoxbs):
    axs[i].plot(wt_hoxb_mean[:,i], label='Wildtype')
    axs[i].plot(mut_hoxb_mean[:,i], label='Mutant')
    axs[i].set_title(hoxb)
    # axs[1,i].set_title(hoxb)
    # axs[0,i].legend()
    # axs[1,i].plot(wt_hoxb_mean[:,i] - mut_hoxb_mean[:,i])
    # axs[1,i].set_title('Difference')
plt.tight_layout()
# %%
# Is POU5F1 differentially expressed in WT and mutant simulations?
pou5f1_idx = node_to_idx[protein_name_id['POU5F1']]
wt_pou5f1_mean = wt_gene_trajectories[:,pou5f1_idx].mean()
mut_pou5f1_mean = mut_gene_trajectories[:,pou5f1_idx].mean()
wt_pou5f1_std = wt_gene_trajectories[:,pou5f1_idx].std()
mut_pou5f1_std = mut_gene_trajectories[:,pou5f1_idx].std()
print('WT mean:', wt_pou5f1_mean, 'WT std:', wt_pou5f1_std)
print('Mut mean:', mut_pou5f1_mean, 'Mut std:', mut_pou5f1_std)
plt.plot(wt_gene_trajectories[:,pou5f1_idx], label='Wildtype')
plt.plot(mut_gene_trajectories[:,pou5f1_idx], label='Mutant')
plt.legend()
# %%
# Compare VIM, MESP1, ID2/3, which we hypothesize are driving 
# cardiogenic mesoderm progenitor (CMP) migration and differentiation
cmp_genes = ['VIM', 'MESP1', 'ID2', 'ID3', 'SNAI1', 'TWIST1']

wt_cmp_mean = wt_gene_trajectories[:,[node_to_idx[protein_name_id[cmp]] for cmp in cmp_genes]]
mut_cmp_mean = mut_gene_trajectories[:,[node_to_idx[protein_name_id[cmp]] for cmp in cmp_genes]]


# In side by side plots, show the mean expression of each of the cmp genes across time
fig, axs = plt.subplots(len(cmp_genes),1, figsize=(5,15))
for i, cmp in enumerate(cmp_genes):
    axs[i].plot(wt_cmp_mean[:,i], label='Wildtype')
    axs[i].plot(mut_cmp_mean[:,i], label='Mutant')
    axs[i].set_title(cmp)
    # axs[1,i].set_title(cmp)
    # axs[0,i].legend()
    # axs[1,i].plot(wt_cmp_mean[:,i] - mut_cmp_mean[:,i])
    # axs[1,i].set_title('Difference')
plt.tight_layout()

# %%
