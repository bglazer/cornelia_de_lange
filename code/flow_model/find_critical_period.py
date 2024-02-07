#%%
import scanpy as sc
# import numpy as np
import pickle
import torch
import sys
sys.path.append('..')
import util
import numpy as np
from sklearn.decomposition import PCA
import plotting
from matplotlib import pyplot as plt
# %%
# Load the models
source_genotype = 'mutant'
target_genotype = 'wildtype'

src_tmstp = '20230607_165324' if source_genotype == 'wildtype' else '20230608_093734'
tgt_tmstp = '20230607_165324' if target_genotype == 'wildtype' else '20230608_093734'
tgt_data = sc.read_h5ad(f'../../data/{target_genotype}_net.h5ad')
src_data = sc.read_h5ad(f'../../data/{source_genotype}_net.h5ad')
tgt_outdir = f'../../output/{tgt_tmstp}'
src_outdir = f'../../output/{src_tmstp}'
transfer = f'{source_genotype}_to_{target_genotype}'
label = ''
# label = 'VIM_first_'
transfer_dir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations'
pltdir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/figures'
datadir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/data'
# %%
torch.set_num_threads(40)
device='cpu'
# num_nodes = tgt_X.shape[1]
hidden_dim = 64
num_layers = 3

#%%
# Load utility data structures for gene names and indexes
cell_types = {c:i for i,c in enumerate(sorted(set(tgt_data.obs['cell_type'])))}
idx_to_cell_type = {v:k for k,v in cell_types.items()}
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}

#%%
n_repeats = 10
start_idxs = src_data.uns['initial_points_nmp']
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
len_trajectory = tgt_data.uns['best_trajectory_length']
step_size = tgt_data.uns['best_step_size']
n_steps = step_size*len_trajectory

t_span = torch.linspace(0, len_trajectory, n_steps)
#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{src_outdir}/baseline_trajectories_{source_genotype}.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{src_outdir}/baseline_nearest_cell_idxs_{source_genotype}.pickle', 'rb'))
# baseline_velo,_ = plotting.compute_velo(model=model, X=src_X, numpy=True)
# baseline_X = baseline_trajectories_np.reshape(-1, num_nodes)
baseline_cell_proportions, baseline_cell_errors = plotting.calculate_cell_type_proportion(baseline_idxs, src_data, cell_types, n_repeats, error=True)
#%%
with open(f'{datadir}/{label}{transfer}_critical_period_proportions.pickle', 'rb') as f:
    results = pickle.load(f)

# %%
distances = []
for combo, rs in results.items():
    ds=[]
    rs = sorted(rs, key=lambda x: x[0])
    for r in rs:
        t = r[1]
        d = np.abs(r[2] - baseline_cell_proportions).sum()
        ds.append(d)
        # print(r)
    distances.append(ds)

# %%
# Plot the distance from the baseline for each rescue timepoint
distances = np.array(distances)
for ds in distances:
    plt.plot(ds, c='blue', marker='.', lw=.5)
plt.plot(distances.mean(axis=0), c='orange')
mean_distance = distances.mean(axis=0)
# decline_point is the first time point where the distance is lesser than the first distance
decline_idx = np.where(mean_distance < mean_distance[0])[0][0]
# Get the t value for the decline point
decline_t = list(results.values())[0][decline_idx][0]
# Annotate the decline point
plt.axvline(decline_idx, c='green')
plt.text(decline_idx, mean_distance.max(), ' Critical Point', ha='left', va='bottom', color='green')
# Annotate the baseline distance
plt.axhline(mean_distance[0], c='grey', alpha=.2)
plt.text(0, mean_distance[0], '\nBaseline Distance', ha='left', va='top', color='grey', alpha=.5)
plt.xlabel('Rescue Timepoint')
plt.ylabel('Distance from Mutant Proportions')
# Set the x-ticks to be the timepoints
plt.xticks(np.linspace(0, len(ds), num=20), 
           [f'{t:.0f}' for t in np.linspace(0, len_trajectory, num=20)],
           rotation=90);
#%%
# Get the cell type of all the cells at the decline point in the baseline data
baseline_decline_nearest_idxs = baseline_idxs[decline_idx*10,None]#-10:decline_idx*10+10]
decline_cell_type_proportions,decline_cell_type_error = plotting.calculate_cell_type_proportion(baseline_decline_nearest_idxs, src_data, cell_types, n_repeats, error=True)

plt.bar(np.arange(len(cell_types)), decline_cell_type_proportions)
plt.xticks(np.arange(len(cell_types)), [idx_to_cell_type[i] for i in range(len(cell_types))])
plt.xlabel('Cell Type')
plt.ylabel('Proportion')
# Add error bars
cell_type_idxs = np.arange(len(cell_types))
y_high = decline_cell_type_error*2
y_low = decline_cell_type_error*2
lt0 = (decline_cell_type_proportions - decline_cell_type_error*2) < 0
y_low[lt0] = decline_cell_type_proportions[lt0]
plt.errorbar(x=cell_type_idxs,
                y=decline_cell_type_proportions,
                yerr=(y_low, y_high),
                c='grey',
                fmt='none')
#%%
tgt_X = torch.tensor(tgt_data.X.toarray()).float()
tgt_Xnp = util.tonp(tgt_X)
src_X = torch.tensor(src_data.X.toarray()).float()
src_Xnp = util.tonp(src_X)
proj = np.array(tgt_data.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = tgt_data.uns['pca_mean']
pca.components_ = tgt_data.uns['PCs']
X_proj = np.array(pca.transform(tgt_X))[:,0:2]
#%%
trajectory_flat = baseline_trajectories_np.reshape(-1, tgt_X.shape[1])
trajectory_proj = np.array(pca.transform(trajectory_flat))
#%%
# Get the indexes of the trajectory points corresponding to the decline point
decline_idxs = np.zeros_like(trajectory_proj[:,0], dtype=bool)
interval = baseline_trajectories.shape[1]
decline_idxs[decline_idx*10*interval:decline_idx*10*interval+interval] = True
# Get the cell types of the trajectory points
traj_cell_types = plotting.idx_to_cell_type(baseline_idxs, src_data, cell_types)

#%%
decline_points = baseline_trajectories_np[decline_idx*10]
decline_points_proj = np.array(pca.transform(decline_points))
plt.scatter(trajectory_proj[:,0], trajectory_proj[:,1], c='blue', s=1, alpha=.01)
plt.scatter(decline_points_proj[:,0], decline_points_proj[:,1], c='red', s=1, alpha=1, marker='.')
#%%
# Make a contour plot in PCA space of the decline points
from scipy.stats import gaussian_kde as kde
x = decline_points_proj[:,0]
y = decline_points_proj[:,1]
xmin = trajectory_proj[:,0].min()
xmax = trajectory_proj[:,0].max()
ymin = trajectory_proj[:,1].min()
ymax = trajectory_proj[:,1].max()

k = kde([x,y])
# 1j is the imaginary unit, which is used as a flag to mgrid to tell it 
# a number of points to make in the grid
xi, yi = np.mgrid[xmin:xmax:x.size**0.5*1j, 
                  ymin:ymax:y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

plotting.cell_type_distribution(baseline_trajectories, nearest_idxs=baseline_idxs, 
                                data=src_data, cell_type_to_idx=cell_types, pca=pca,
                                baseline=decline_points, label='Critical Period Cell Distribution',
                                scatter_alpha=1)

# %%
# Plot the velocity vectors with an overlay showing the decline point density
from flow_model import GroupL1FlowModel

source_state_dict = torch.load(f'{src_outdir}/models/optimal_{source_genotype}.torch')
target_state_dict = torch.load(f'{tgt_outdir}/models/optimal_{target_genotype}.torch')

src_model = GroupL1FlowModel(input_dim=tgt_X.shape[1], 
                             hidden_dim=hidden_dim, 
                             num_layers=num_layers,
                             predict_var=True)
src_model.load_state_dict(source_state_dict)
src_model = src_model.to(device)
tgt_model = GroupL1FlowModel(input_dim=tgt_X.shape[1], 
                             hidden_dim=hidden_dim, 
                             num_layers=num_layers,
                             predict_var=True)
tgt_model.load_state_dict(target_state_dict)
tgt_model = tgt_model.to(device)
#%%
src_velos,_ = plotting.compute_velo(model=src_model, X=src_X, numpy=True)
tgt_velos,_ = plotting.compute_velo(model=tgt_model, X=src_X, numpy=True)

plotting.arrow_grid(velos=[tgt_velos, src_velos],
                    data=[src_data, src_data], 
                    pca=pca, 
                    labels=[target_genotype, source_genotype],)
xi, yi = np.mgrid[xmin:xmax:x.size**0.5*1j, 
                  ymin:ymax:y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
# %%
