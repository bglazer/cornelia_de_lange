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
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from tqdm import tqdm

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
class Simulator():
    def __init__(self, model, X):
        self.model = model
        print('Building KDTree')
        self.tree = KDTree(X, leaf_size=100)
        print('Querying KDTree')
        self.nearest_neighb_dists = self.tree.query(X, k=2)[0][:,1]
        self.max_distance = np.percentile(self.nearest_neighb_dists, 99)

    def euler_step(self, x, dt):
        with torch.no_grad():
            dx, var = self.model(x)
            var[var < 0] = 0
            std = torch.sqrt(var)
            inside = np.zeros(x.shape[0], dtype=bool)
            noise = torch.zeros_like(x)
            while True:
                num_outside = (~inside).sum()
                noise[~inside] = torch.randn(num_outside, noise.shape[1]) * std[~inside]
                #*****************************
                # TODO why is this negative?
                #*****************************
                x1 = x - dt*(dx+noise)
                dist, _ = self.tree.query(x1[~inside], k=2)
                inside_query = dist[:,1] < self.max_distance
                inside[~inside] = inside_query
                if np.all(inside):
                    break
                
            return x1

    def simulate(self, x, t_span, knockout_idx=None):
        last_t = t_span[0]
        traj = torch.zeros(len(t_span), x.shape[0], x.shape[1])
        traj[0,:,:] = x.clone()
        if knockout_idx is not None:
            traj[0,:,knockout_idx] = 0.0
        for i,t in tqdm(enumerate(t_span[1:]), total=len(t_span)-1):
            dt = t - last_t
            traj[i+1] = self.euler_step(traj[i], dt)
            if knockout_idx is not None:
                traj[i+1,:,knockout_idx] = 0.0
            last_t = t
        return traj

    def trajectory(self, x, t_span, knockout_idx, n_repeats=1, n_parallel=1):
        with torch.no_grad():
            x = x.clone()
            parallel = Parallel(n_jobs=n_parallel, verbose=0)
            jobs = []
            for j in range(n_repeats):
                jobs.append(delayed(self.simulate)(x, t_span, knockout_idx))
            trajectories = parallel(jobs)
            traj = torch.concatenate(trajectories, dim=1)
            return traj

simulator = Simulator(model, X)
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
# %%
