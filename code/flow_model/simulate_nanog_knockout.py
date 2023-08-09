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
state_dict = torch.load(f'{outdir}/models/optimal_{genotype}.torch')

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
knockout_gene = 'NANOG'
knockout_idx = data.var.index.get_loc(knockout_gene)
n_repeats = 10
len_trajectory = 250
n_steps = 250*4
noise_scale = 1.0
t_span = torch.linspace(0, len_trajectory, n_steps, device=device)
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
ko_trajectories, ko_nearest_idxs = simulator.simulate(repeats, t_span, 
                                                      perturbation=(knockout_idx, 0.0), 
                                                      noise_scale=noise_scale)
wt_trajectories, wt_nearest_idxs = simulator.simulate(repeats, t_span, 
                                                      perturbation=None, 
                                                      noise_scale=noise_scale)
#%%
ko_trajectories_np = util.tonp(ko_trajectories)
wt_trajectories_np = util.tonp(wt_trajectories)
ko_idxs_np = util.tonp(ko_nearest_idxs)
wt_idxs_np = util.tonp(wt_nearest_idxs)
Xnp = util.tonp(X)
#%%
plotting.distribution(ko_trajectories_np, pca, 'MESP1 Knockout')
#%%
plotting.distribution(wt_trajectories_np, pca, 'Wildtype')
# %%
plotting.arrow_grid(data, pca, model, 'MESP1 Knockout', 
                    perturbation=(knockout_idx,0.0), device=device)
#%%
plotting.arrow_grid(data, pca, model, 'Wildtype', perturbation=None, device=device)
# %%
plotting.sample_trajectories(ko_trajectories_np, Xnp, pca, 'MESP1 Knockout')
#%%
plotting.sample_trajectories(wt_trajectories_np, Xnp, pca, 'Wildtype')

# %%
trajectories = plotting.compare_cell_type_trajectories(np.stack((wt_idxs_np,ko_idxs_np )), 
                                                       data, ('Wildtype','MESP1 Knockout'))
ko_cell_type_traj = trajectories[0]
wt_cell_type_traj = trajectories[1]
# %%
wt_total_cell_types = wt_cell_type_traj.sum(axis=1)/wt_cell_type_traj.sum()
ko_total_cell_types = ko_cell_type_traj.sum(axis=1)/ko_cell_type_traj.sum()
# Plot side by side bar charts of the cell type proportions
plt.bar(np.arange(len(wt_total_cell_types))+.2, wt_total_cell_types, label='Wildtype', width=.4)
plt.bar(np.arange(len(ko_total_cell_types))-.2, ko_total_cell_types, label='Knockout', width=.4)
plt.xticks(np.arange(len(wt_total_cell_types)), data.obs['cell_type'].cat.categories, rotation=90);
plt.ylabel('Proportion of cells')
plt.legend()
#%%
# Recreating Fig 2E from https://www.science.org/doi/10.1126/science.aao4174
genes = ['NANOG', 'SNAI1']
fig, axs = plt.subplots(nrows=len(genes), ncols=1, figsize=(5, len(genes)*5))
for i,gene in enumerate(genes):
    gene_idx = data.var.index.get_loc(gene)
    # plt.scatter(proj[:,0], proj[:,1], s=nanog*20+1, alpha=.5, c=nanog)
    wt_nanog = wt_trajectories_np[:][:,:,gene_idx].flatten()
    ko_nanog = ko_trajectories_np[:][:,:,gene_idx].flatten()
    axs[i].violinplot([wt_nanog, ko_nanog], showmeans=True)
    axs[i].set_xticks([1,2], ['Wildtype', 'Knockout'])
    axs[i].set_ylabel(gene)

# %%
# Plot WT vs Mutant Nanog expression
wt_data = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut_data = sc.read_h5ad(f'../../data/mutant_net.h5ad')
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
nanog_idx = data.var.index.get_loc('NANOG')
wt_nanog = wt_data.X[:,nanog_idx].toarray().flatten()
mut_nanog = mut_data.X[:,nanog_idx].toarray().flatten()
wt_nanog_gt_zero = wt_nanog[wt_nanog>0]
mut_nanog_gt_zero = mut_nanog[mut_nanog>0]
_, bins, _ = axs.hist(wt_nanog_gt_zero, bins=100, alpha=.2, label='Wildtype')
axs.hist(mut_nanog_gt_zero, bins=bins, alpha=.2, label='Mutant')
axs.set_xlabel('Nanog Expression')
axs.set_ylabel('Number of cells')
axs.legend()

#%%

# %%
