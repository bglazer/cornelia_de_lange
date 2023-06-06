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
genotype = 'wildtype'
data = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')
# %%
# Load the models
tmstp = '20230601_143356'
outdir = f'../../output/{tmstp}'
models = pickle.load(open(f'{outdir}/models/group_l1_variance_model_wildtype.pickle', 'rb'))
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
hidden_dim = 32
num_layers = 3

model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers,
                         predict_var=True)
for i,state_dict in enumerate(models):
    model.models[i].load_state_dict(state_dict[0])

#%%
device='cuda:0'
model = model.to(device)
X = X.to(device)
simulator = Simulator(model, X, device=device)
#%%
knockout_gene = 'MESP1'
knockout_idx = data.var.index.get_loc(knockout_gene)
n_repeats = 10
len_trajectory = 250
noise_scale = .5
t_span = torch.linspace(0, len_trajectory, len_trajectory, device=device)
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
ko_trajectories, ko_nearest_idxs = simulator.simulate(repeats, t_span, knockout_idx=knockout_idx, noise_scale=noise_scale)
wt_trajectories, wt_nearest_idxs = simulator.simulate(repeats, t_span, knockout_idx=None, noise_scale=noise_scale)
#%%
ko_trajectories_np = util.tonp(ko_trajectories)
wt_trajectories_np = util.tonp(wt_trajectories)
ko_idxs_np = util.tonp(ko_nearest_idxs)
wt_idxs_np = util.tonp(wt_nearest_idxs)
Xnp = util.tonp(X)
#%%
plotting.distribution(ko_trajectories_np, pca)
#%%
plotting.distribution(wt_trajectories_np, pca)
# %%
plotting.arrow_grid(data, pca, model, 'MESP1 Knockout', knockout_idx=knockout_idx, device=device)
#%%
plotting.arrow_grid(data, pca, model, 'Wildtype', knockout_idx=None, device=device)
# %%
plotting.sample_trajectories(ko_trajectories_np, Xnp, pca, 'MESP1 Knockout')
#%%
plotting.sample_trajectories(wt_trajectories_np, Xnp, pca, 'Wildtype')
# %%
ko_cell_type_traj = plotting.cell_type_proportions(ko_trajectories_np, data, ko_idxs_np, 'MESP1 Knockout')
#%%
wt_cell_type_traj = plotting.cell_type_proportions(wt_trajectories_np, data, wt_idxs_np, 'Wildtype')
# %%
wt_total_cell_types = wt_cell_type_traj.sum(axis=0)/wt_cell_type_traj.sum()
ko_total_cell_types = ko_cell_type_traj.sum(axis=0)/ko_cell_type_traj.sum()
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
    nanog = Xnp[:,gene_idx]
    # plt.scatter(proj[:,0], proj[:,1], s=nanog*20+1, alpha=.5, c=nanog)
    wt_nanog = wt_trajectories_np[:][:,:,gene_idx].flatten()
    ko_nanog = ko_trajectories_np[:][:,:,gene_idx].flatten()
    axs[i].violinplot([wt_nanog, ko_nanog], showmeans=True)
    axs[i].set_xticks([1,2], ['Wildtype', 'Knockout'])
    axs[i].set_ylabel(gene)
# %%
# Recreating Fig 2E from https://www.science.org/doi/10.1126/science.aao4174
gene_idx = data.var.index.get_loc('NANOG')
nanog = Xnp[:,gene_idx]
plt.scatter(proj[:,0], proj[:,1], s=nanog*20+1, alpha=.5, c=nanog)

# %%
wt_nanog = wt_trajectories_np[-10:][:,:,gene_idx].flatten()
ko_nanog = ko_trajectories_np[-10:][:,:,gene_idx].flatten()
plt.violinplot([wt_nanog, ko_nanog], showmeans=True)
plt.xticks([1,2], ['Wildtype', 'Knockout'])
plt.legend(['Wildtype', 'Knockout'])
