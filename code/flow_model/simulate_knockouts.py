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
import plotting
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
# models = pickle.load(open(f'{outdir}/models/group_l1_variance_model_{genotype}.pickle', 'rb'))

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
starts = X[start_idxs,:].to('cpu')
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
simulator = Simulator(model, X[:10])
#%%
# Run simulations from each starting point

# Get the index of the gene that we want to knock out
knockout_gene = 'MESP1'
knockout_idx = data.var.index.get_loc(knockout_gene)
len_trajectory = 100
n_repeats = 1
num_trajectories = starts.shape[0]
t_span = torch.linspace(0,len_trajectory,len_trajectory, device='cpu')
# simulator.simulate(starts, t_span, knockout_idx=knockout_idx)
trajectories = simulator.trajectory(starts, t_span, 
                                    knockout_idx=knockout_idx,
                                    n_repeats=n_repeats, 
                                    n_parallel=40)

#%%
plotting.distribution(trajectories, pca)
# %%
plotting.arrow_grid(data, pca, model, genotype)
# %%
plotting.sample_trajectories(trajectories, X, pca, genotype)
# %%
kdtree = simulator.tree
plotting.cell_type_proportions(trajectories, data, kdtree, genotype)