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
from util import tonp
import numpy as np
from sklearn.decomposition import PCA
from simulator import Simulator
from matplotlib import pyplot as plt
import plotting
from scipy import stats

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# %%
genotype = 'wildtype'
data = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')
# %%
# Load the models
tmstp = '20230607_165324'
outdir = f'../../output/{tmstp}'
state_dict = torch.load(f'{outdir}/models/optimal_wildtype.torch')

# tmstp = '20230602_112554'
# outdir = f'../../output/{tmstp}'
# models = pickle.load(open(f'{outdir}/models/group_l1_variance_model_mutant.pickle', 'rb'))

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
knockout_gene = 'MESP1'
knockout_idx = data.var.index.get_loc(knockout_gene)
perturbation = (knockout_idx, 0.0)
n_repeats = 10
len_trajectory = 98
n_steps = len_trajectory*4
noise_scale = 1.0
t_span = torch.linspace(0, len_trajectory, n_steps, device=device)
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
ko_trajectories, ko_nearest_idxs = simulator.simulate(repeats, t_span, 
                                                      perturbation=perturbation, 
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
wt_velos = tonp(plotting.compute_velo(model, X)[0])
mut_velos = tonp(plotting.compute_velo(model, X, perturbation=perturbation)[0])
velos = (wt_velos, mut_velos)
plotting.arrow_grid(velos, [data]*2, pca, ('Wildtype', 'MESP1 Knockout'))
# %%
plotting.sample_trajectories(ko_trajectories_np, Xnp, pca, 'MESP1 Knockout')
#%%
plotting.sample_trajectories(wt_trajectories_np, Xnp, pca, 'Wildtype')

#%%
trajectories = plotting.compare_cell_type_trajectories(nearest_idxs=(ko_idxs_np, wt_idxs_np), 
                                                       data=[data]*2, 
                                                       cell_type_to_idx=cell_types,
                                                       labels=('MESP1 Knockout', 'Wildtype'))
ko_cell_type_traj = trajectories[0]
wt_cell_type_traj = trajectories[1]
# %%
n_cells = (len(start_idxs) * n_repeats * n_steps)
wt_cell_proportions = wt_cell_type_traj.sum(axis=1) / n_cells
ko_cell_proportions = ko_cell_type_traj.sum(axis=1) / n_cells
plotting.cell_type_proportions(proportions=(wt_cell_proportions, ko_cell_proportions), 
                               cell_types=cell_types, 
                               labels=['Wildtype', 'MESP1 Knockout'])
#%%
# Recreating Fig 2E from https://www.science.org/doi/10.1126/science.aao4174
genes = ['NANOG', 'SNAI1']
fig, axs = plt.subplots(nrows=len(genes), ncols=1, figsize=(5, len(genes)*5))
for i,gene in enumerate(genes):
    gene_idx = data.var.index.get_loc(gene)
    wt_expr = wt_trajectories_np[:,:,gene_idx].flatten()
    ko_expr = ko_trajectories_np[:,:,gene_idx].flatten()
    axs[i].violinplot([wt_expr, ko_expr], showmeans=True)
    axs[i].set_xticks([1,2], ['Wildtype', 'Knockout'])
    axs[i].set_ylabel('Expression')

# %%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
gene_idx = data.var.index.get_loc('NANOG')
wt_expr = wt_trajectories_np[:,:,gene_idx].flatten()
ko_expr = ko_trajectories_np[:,:,gene_idx].flatten()
# Make a scatter plot of the trajectory with NANOG expression as the color
wt_traj_proj = pca.transform(wt_trajectories_np.reshape(-1, wt_trajectories_np.shape[2]))
ko_traj_proj = pca.transform(ko_trajectories_np.reshape(-1, ko_trajectories_np.shape[2]))
axs[0].scatter(wt_traj_proj[:,0], wt_traj_proj[:,1], c=wt_expr, cmap='viridis', s=wt_expr*20+.1)
axs[0].set_title('Wildtype')
axs[1].scatter(ko_traj_proj[:,0], ko_traj_proj[:,1], c=ko_expr, cmap='viridis', s=ko_expr*20+.1)
axs[1].set_title('MESP1 Knockout');