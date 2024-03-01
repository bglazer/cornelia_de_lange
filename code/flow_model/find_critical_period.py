#%%
%load_ext autoreload
%autoreload 2

#%%
import scanpy as sc
# import numpy as np
import pickle
import torch
import sys
sys.path.append('..')
import util
from util import tonp
import numpy as np
from sklearn.decomposition import PCA
import plotting
from matplotlib import pyplot as plt
from tqdm import tqdm
from glob import glob
import torch
from matplotlib.cm import viridis, plasma, ScalarMappable
from matplotlib.gridspec import GridSpec

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
device='cuda:0'
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
src_baseline_trajectories = pickle.load(open(f'{src_outdir}/baseline_trajectories_{source_genotype}.pickle', 'rb'))
tgt_baseline_trajectories = pickle.load(open(f'{tgt_outdir}/baseline_trajectories_{target_genotype}.pickle', 'rb'))
src_baseline_trajectories_np = src_baseline_trajectories
tgt_baseline_trajectories_np = tgt_baseline_trajectories
src_baseline_idxs = pickle.load(open(f'{src_outdir}/baseline_nearest_cell_idxs_{source_genotype}.pickle', 'rb'))
tgt_baseline_idxs = pickle.load(open(f'{tgt_outdir}/baseline_nearest_cell_idxs_{target_genotype}.pickle', 'rb'))
# baseline_velo,_ = plotting.compute_velo(model=model, X=src_X, numpy=True)
# baseline_X = baseline_trajectories_np.reshape(-1, num_nodes)
src_baseline_cell_proportions, src_baseline_cell_errors = plotting.calculate_cell_type_proportion(src_baseline_idxs, src_data, cell_types, n_repeats, error=True)
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
        d = np.abs(r[2] - src_baseline_cell_proportions).sum()
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
# Calculate the distance from baseline for each cell type over time
n_timepoints = distances.shape[1]
n_cell_types = len(cell_types)
cell_type_distances = np.zeros((n_timepoints, n_cell_types))
for combo, rs in results.items():
    rs = sorted(rs, key=lambda x: x[0])
    for i,r in enumerate(rs):
        t = r[1]
        rescue_proportions = r[2] 
        d = np.abs(rescue_proportions - src_baseline_cell_proportions)
        cell_type_distances[i] += d
cell_type_distances /= len(results)
#%%
# Plot the distance from the baseline for each cell type over time
for i,d in enumerate(cell_type_distances):
    plt.plot(d, marker='.', lw=.5, c=viridis(i/n_timepoints), alpha=.6)
    plt.xticks(np.arange(len(cell_types)), [idx_to_cell_type[i] for i in range(len(cell_types))], rotation=45)
    plt.xlabel('Cell Type')
    plt.ylabel(f'Distance from {source_genotype.capitalize()} Proportions')
#%%
# Plot SHF and TPM cell types specifically over time
shf_idx = cell_types['SHF']
tpm_idx = cell_types['TPM']

plt.plot(cell_type_distances[:,shf_idx], marker='.', lw=.5, label='SHF')
plt.plot(cell_type_distances[:,tpm_idx], marker='.', lw=.5, label='TPM')
plt.title('SHF/TPM Cell Type Distance from Baseline')
plt.xticks([])
plt.ylabel('Distance from Baseline')

#%%
# Plot the distance from the baseline for each cell type over time as a heatmap
plt.imshow(cell_type_distances.T, cmap='viridis', aspect='auto', origin='upper', interpolation='none')
plt.xlabel('Rescue Timepoint')
plt.ylabel('Cell Type')
plt.colorbar()
plt.xticks(np.linspace(0, len(ds)-1, num=20), 
           [f'{t:.0f}' for t in np.linspace(0, len_trajectory, num=20)],
           rotation=45);
plt.yticks(np.arange(len(cell_types)), 
           [idx_to_cell_type[i] for i in range(len(cell_types))]);
plt.axvline(decline_idx, c='white', alpha=1)

#%%
# Get the cell type of all the cells at the decline point in the baseline data
baseline_decline_nearest_idxs = src_baseline_idxs[decline_idx*10,None]#-10:decline_idx*10+10]
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
src_trajectory_flat = src_baseline_trajectories_np.reshape(-1, tgt_X.shape[1])
src_trajectory_proj = np.array(pca.transform(src_trajectory_flat))
tgt_trajectory_flat = tgt_baseline_trajectories_np.reshape(-1, tgt_X.shape[1])
tgt_trajectory_proj = np.array(pca.transform(tgt_trajectory_flat))
#%%
# Get the indexes of the trajectory points corresponding to the decline point
decline_idxs = np.zeros_like(src_trajectory_proj[:,0], dtype=bool)
interval = src_baseline_trajectories.shape[1]
decline_idxs[decline_idx*10*interval:decline_idx*10*interval+interval] = True
# Get the cell types of the trajectory points
traj_cell_types = plotting.idx_to_cell_type(src_baseline_idxs, src_data, cell_types)

#%%
decline_points = src_baseline_trajectories_np[decline_idx*10]
decline_points_proj = np.array(pca.transform(decline_points))
plt.scatter(src_trajectory_proj[:,0], src_trajectory_proj[:,1], c='blue', s=1, alpha=.01)
plt.scatter(decline_points_proj[:,0], decline_points_proj[:,1], c='red', s=1, alpha=1, marker='.')
#%%
# Make a contour plot in PCA space of the decline points
from scipy.stats import gaussian_kde as kde
x = decline_points_proj[:,0]
y = decline_points_proj[:,1]
xmin = src_trajectory_proj[:,0].min()
xmax = src_trajectory_proj[:,0].max()
ymin = src_trajectory_proj[:,1].min()
ymax = src_trajectory_proj[:,1].max()

k = kde([x,y])
# 1j is the imaginary unit, which is used as a flag to mgrid to tell it 
# a number of points to make in the grid
xi, yi = np.mgrid[xmin:xmax:x.size**0.5*1j, 
                  ymin:ymax:y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# plotting.cell_type_distribution(baseline_trajectories, nearest_idxs=baseline_idxs, 
#                                 data=src_data, cell_type_to_idx=cell_types, pca=pca,
#                                 baseline=decline_points, label='Critical Period Cell Distribution',
#                                 scatter_alpha=1)

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
src_velos,_ = plotting.compute_velo(model=src_model, X=src_X.to(device), numpy=True)
tgt_velos,_ = plotting.compute_velo(model=tgt_model, X=src_X.to(device), numpy=True)

plotting.arrow_grid(velos=[tgt_velos, src_velos],
                    data=[src_data, src_data], 
                    pca=pca, 
                    labels=[target_genotype, source_genotype],)
xi, yi = np.mgrid[xmin:xmax:x.size**0.5*1j, 
                  ymin:ymax:y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
# %%
# Get the velos at each step of the trajectory
device='cuda:0'
src_model = src_model.to(device)
tgt_model = tgt_model.to(device)
src_baseline_trajectories_tensor = torch.tensor(src_baseline_trajectories_np).float().to(device)
tgt_baseline_trajectories_tensor = torch.tensor(tgt_baseline_trajectories_np).float().to(device)
src_velocity_timeseries = torch.zeros_like(src_baseline_trajectories_tensor, device='cpu')
tgt_velocity_timeseries = torch.zeros_like(src_baseline_trajectories_tensor, device='cpu')
src_var_timeseries = torch.zeros_like(src_baseline_trajectories_tensor, device='cpu')
tgt_var_timeseries = torch.zeros_like(src_baseline_trajectories_tensor, device='cpu')

with torch.no_grad():
    for i in tqdm(range(len(src_baseline_trajectories_tensor))):
        src_velo, src_var = src_model(src_baseline_trajectories_tensor[i])
        tgt_velo, tgt_var = tgt_model(src_baseline_trajectories_tensor[i])
        src_velocity_timeseries[i] = torch.clamp(src_velo, min=0.0).cpu()
        tgt_velocity_timeseries[i] = torch.clamp(tgt_velo, min=0.0).cpu()
        src_var_timeseries[i] = src_var.detach().cpu()
        tgt_var_timeseries[i] = tgt_var.detach().cpu()
# %%
# Plot the average velocity magnitude at each timepoint
src_velo_mag = torch.norm(src_velocity_timeseries, dim=2)
tgt_velo_mag = torch.norm(tgt_velocity_timeseries, dim=2)

# Standard deviation of the velocity magnitude
src_velo_mag_std = src_velo_mag.std(dim=1)
tgt_velo_mag_std = tgt_velo_mag.std(dim=1)

plt.plot(src_velo_mag.mean(dim=1), label=source_genotype)
plt.plot(tgt_velo_mag.mean(dim=1), label=target_genotype)
# Add error bars
plt.fill_between(np.arange(len(src_velo_mag.mean(dim=1))),
                 src_velo_mag.mean(dim=1)-src_velo_mag_std,
                 src_velo_mag.mean(dim=1)+src_velo_mag_std,
                 alpha=.5)

plt.fill_between(np.arange(len(tgt_velo_mag.mean(dim=1))),
                    tgt_velo_mag.mean(dim=1)-tgt_velo_mag_std,
                    tgt_velo_mag.mean(dim=1)+tgt_velo_mag_std,
                    alpha=.5)

plt.xticks(np.linspace(0, len(t_span), num=20),
              [f'{t:.0f}' for t in np.linspace(0, t_span[-1], num=20)],
              rotation=90)
plt.axvline(t_span[decline_idx*10]*4, c='green', alpha=.3)
plt.legend()

# %%
# Plot a grid in PCA space of the velocity magnitudes
ngrid=60
xgrid = torch.linspace(xmin, xmax, ngrid)
ygrid = torch.linspace(ymin, ymax, ngrid)

# Convert numpy arrays to PyTorch tensors
trajectory_proj_tensor = torch.from_numpy(src_trajectory_proj)
# src_velo_mag = torch.from_numpy(src_velo_mag)
# tgt_velo_mag = torch.from_numpy(tgt_velo_mag)

# Get the points inside each grid cell
src_velo_mag_grid = torch.zeros((ngrid,ngrid))
tgt_velo_mag_grid = torch.zeros((ngrid,ngrid))
diff_grid = torch.zeros((ngrid,ngrid))
src_velo_mag_grid[:] = np.nan
tgt_velo_mag_grid[:] = np.nan
diff_grid[:] = np.nan

x = trajectory_proj_tensor[:,0]
y = trajectory_proj_tensor[:,1]

# Use torch.bucketize to find the bin indices
x_indices = torch.bucketize(x, xgrid)
y_indices = torch.bucketize(y, ygrid)

for i in range(ngrid-1):
    for j in range(ngrid-1):
        inside = (x_indices == i+1) & (y_indices == j+1)
        if inside.sum() >= 10:
            src_velo_mag_grid[i,j] = src_velo_mag.flatten()[inside].mean()
            tgt_velo_mag_grid[i,j] = tgt_velo_mag.flatten()[inside].mean()
diff_grid = src_velo_mag_grid - tgt_velo_mag_grid

# %%
plt.imshow(src_velo_mag_grid.T, cmap='viridis', origin='lower', 
           extent=(xmin, xmax, ymin, ymax))
plt.colorbar()
plt.title(f'{source_genotype.capitalize()} Velocity Magnitude')

# Plot the contours of the decline point density
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
#%%
plt.imshow(tgt_velo_mag_grid.T, cmap='viridis', origin='lower',
           extent=(xmin, xmax, ymin, ymax))
plt.colorbar()
plt.title(f'{target_genotype.capitalize()} Velocity Magnitude')
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
# %%
diffs = diff_grid.flatten()
diffs = diffs[~torch.isnan(diffs)]
maxdiff = max(torch.abs(diffs).max(), torch.abs(diffs).min())
plt.imshow(diff_grid.T, cmap='coolwarm', origin='lower',
           extent=(xmin, xmax, ymin, ymax), 
           vmin=-maxdiff, vmax=maxdiff)
plt.colorbar()
plt.title(f'{source_genotype.capitalize()} - {target_genotype.capitalize()} Velocity Magnitude Difference')
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
# %%
# Get the trajectory points inside the cells with the highest velocity magnitude difference
diffs = (src_velo_mag - tgt_velo_mag).flatten()
diff_pct_idxs = torch.sort(diffs)[1][:int(len(diffs)*.01)]

traj_cell_types = plotting.idx_to_cell_type(src_baseline_idxs, src_data, cell_types)
traj_cell_types = traj_cell_types.flatten()

diff_pct_cell_types = traj_cell_types[diff_pct_idxs]
diff_pct_cell_types = torch.from_numpy(diff_pct_cell_types)
# Count the number of cells of each type, using torch.unique
count_type_idxs, cell_type_counts = torch.unique(diff_pct_cell_types, return_counts=True)

# Bar chart of the cell types
plt.bar(np.arange(len(cell_type_counts)), cell_type_counts/cell_type_counts.sum())
plt.xticks(np.arange(len(cell_type_counts)), 
           [idx_to_cell_type[int(i)] for i in count_type_idxs]);

#%%
# Map cell types to colors from the palette
cell_type_colors = plotting.cell_colors(cell_types)
diff_pct_colors = [cell_type_colors[i] for i in diff_pct_cell_types]
plt.scatter(src_trajectory_proj[:,0], src_trajectory_proj[:,1], c='grey', s=1, alpha=.1)
plt.scatter(src_trajectory_proj[diff_pct_idxs,0], src_trajectory_proj[diff_pct_idxs,1], 
            s=1, alpha=1, c=diff_pct_colors)

# Rectangles to show color of cell types
from matplotlib.patches import Rectangle
handles = [Rectangle((0,0),1,1, color=cell_type_colors[c], ec='black') 
           for c in cell_types.values()]
labels = [idx_to_cell_type[int(i)] for i in count_type_idxs]
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')

# %%
# Plot the variance of the velocities
src_velo_var = torch.mean(src_var_timeseries, dim=(2,1))
tgt_velo_var = torch.mean(tgt_var_timeseries, dim=(2,1))

plt.plot(src_velo_var, label=source_genotype)
plt.plot(tgt_velo_var, label=target_genotype)
plt.fill_between(np.arange(len(src_velo_var)),
                 src_velo_var-src_velo_var.std(),
                 src_velo_var+src_velo_var.std(),
                 alpha=.5)
plt.fill_between(np.arange(len(tgt_velo_var)),
                 tgt_velo_var-tgt_velo_var.std(),
                 tgt_velo_var+tgt_velo_var.std(),
                 alpha=.5)
plt.legend()
# %%
# Make a grid in PCA space of the variance of the velocities
src_velo_var = torch.mean(src_var_timeseries, dim=(2))
tgt_velo_var = torch.mean(tgt_var_timeseries, dim=(2))

src_velo_var_grid = torch.zeros((ngrid,ngrid))
tgt_velo_var_grid = torch.zeros((ngrid,ngrid))
src_velo_var_grid[:] = np.nan
tgt_velo_var_grid[:] = np.nan

for i in range(ngrid-1):
    for j in range(ngrid-1):
        inside = (x_indices == i+1) & (y_indices == j+1)
        if inside.sum() >= 10:
            src_velo_var_grid[i,j] = src_velo_var.flatten()[inside].mean()
            tgt_velo_var_grid[i,j] = tgt_velo_var.flatten()[inside].mean()
# diff_grid = src_velo_var_grid - tgt_velo_var_grid
#%%
maxvar = max(src_velo_var_grid[~src_velo_var_grid.isnan()].max(), 
             tgt_velo_var_grid[~tgt_velo_var_grid.isnan()].max())
minvar = min(src_velo_var_grid[~src_velo_var_grid.isnan()].min(), 
             tgt_velo_var_grid[~tgt_velo_var_grid.isnan()].min())
#%%
plt.imshow(src_velo_var_grid.T, cmap='viridis', origin='lower', 
           extent=(xmin, xmax, ymin, ymax), vmin=minvar, vmax=maxvar)
plt.title(f'{source_genotype.capitalize()} Velocity Variance')
plt.colorbar()
# %%
plt.imshow(tgt_velo_var_grid.T, cmap='viridis', origin='lower',
              extent=(xmin, xmax, ymin, ymax), vmin=minvar, vmax=maxvar)
plt.colorbar()
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')

plt.title(f'{target_genotype.capitalize()} Velocity Variance')


#%%
def autocorr(x, lag=1):
    # Mean across time dimension for all genes and repeats
    u = x.mean(axis=0)[None,:,:]
    ac = ((x[lag:]-u)*(x[:-lag]-u))
    return ac

#%%
# Plot the lag-1 autocorrelation of the trajectories
n_genes = tgt_X.shape[1]
len_tspan = len(src_baseline_trajectories_tensor)
src_autocorr = torch.zeros((len_tspan, n_genes))
tgt_autocorr = torch.zeros((len_tspan, n_genes))

src_autocorr = autocorr(src_baseline_trajectories_tensor, lag=1)
tgt_autocorr = autocorr(tgt_baseline_trajectories_tensor, lag=1)

# Autocorrelation averaged across both the repeat and gene dimensions
src_autocorr_t = tonp(src_autocorr.mean(dim=(1,2)))
tgt_autocorr_t = tonp(tgt_autocorr.mean(dim=(1,2)))
# Autocorrelation for time and genes averaged across repeats
src_autocorr_genes = tonp(src_autocorr.mean(dim=(1)))
tgt_autocorr_genes = tonp(tgt_autocorr.mean(dim=(1)))

fig, ax1 = plt.subplots()
ax1.plot(src_autocorr_t,c='orange', label=source_genotype)
ax1.plot(tgt_autocorr_t,c='blue', label=target_genotype)
diff = src_autocorr_t - tgt_autocorr_t
# Make a second y axis
ax1.set_title('Lag-1 Autocorrelation of Trajectories Over Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Autocorrelation')
ax1.set_xticks(np.linspace(0, len_tspan, num=20),
           [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=20)],
           rotation=45)
ax2 = plt.twinx()
ax2.set_ylabel('AutoCorrelation Difference')
ax2.plot(diff, c='grey', alpha=.5, label='Difference')
ax2.axvline(decline_idx*10, c='green', alpha=.3)
# label the critical point
ax2.text(decline_idx*10, diff.max()*.95, ' Critical Point', 
         ha='right', va='top', color='green',
         rotation=90, alpha=.5)

artists = ax1.lines + ax2.lines
plt.legend(artists, [l.get_label() for l in artists])
# %%
# n_repeats = src_baseline_trajectories_tensor.shape[1]
# # Make the autocorrelation the same shape as the trajectory, then unroll it 
# plt_autocorr = src_autocorr.sum(dim=1).expand((n_repeats,len_tspan)).T.reshape((-1))
# plt.scatter(src_trajectory_proj[:,0], src_trajectory_proj[:,1], 
#             c=plt_autocorr[:],
#             s=1, alpha=.1, cmap='viridis')
# # Contour plot
# plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
# plt.title(f'{target_genotype.capitalize()}Lag-1 Autocorrelation of Trajectories')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.xticks([])
# plt.yticks([]);

#%%
n_repeats = tgt_baseline_trajectories_tensor.shape[1]
# plt_autocorr = tgt_autocorr.sum(dim=1).expand((n_repeats,len_tspan)).T.reshape((-1))
# plt.scatter(tgt_trajectory_proj[:,0], tgt_trajectory_proj[:,1], 
#             c=plt_autocorr[:],
#             s=1, alpha=.1, cmap='viridis')
# # Contour plot
# plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
# plt.title(f'{target_genotype.capitalize()} Lag-1 Autocorrelation of Trajectories')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.xticks([])
# plt.yticks([]);

# %%
# Make a grid in PCA space of the lag-1 autocorrelation
src_autocorr_grid = torch.zeros((ngrid,ngrid))
tgt_autocorr_grid = torch.zeros((ngrid,ngrid))
src_autocorr_grid[:] = np.nan
tgt_autocorr_grid[:] = np.nan

src_x_indices = np.digitize(src_trajectory_proj[:,0], xgrid)
src_y_indices = np.digitize(src_trajectory_proj[:,1], ygrid)

tgt_x_indices = np.digitize(tgt_trajectory_proj[:,0], xgrid)
tgt_y_indices = np.digitize(tgt_trajectory_proj[:,1], ygrid)

src_n_repeats = src_baseline_trajectories_tensor.shape[1]
tgt_n_repeats = tgt_baseline_trajectories_tensor.shape[1]

for i in range(ngrid-1):
    for j in range(ngrid-1):
        src_inside = (src_x_indices == i+1) & (src_y_indices == j+1)
        tgt_inside = (tgt_x_indices == i+1) & (tgt_y_indices == j+1)
        if src_inside.sum() >= 10:
            # Get the timepoints of the trajectory points inside the grid cell
            inside_idxs = src_inside.reshape(len_tspan, src_n_repeats)
            # Get the autocorrelation of the trajectory points inside the grid cell
            time_counts = (inside_idxs.sum(axis=1)/inside_idxs.sum())[1:]
            # Calculate the mean autocorrelation, weighted by the number of timepoints
            # inside the grid cell
            src_autocorr_grid[i,j] = (src_autocorr_t*time_counts).mean()
        if tgt_inside.sum() >= 10:
            # Get the timepoints of the trajectory points inside the grid cell
            inside_idxs = tgt_inside.reshape(len_tspan, tgt_n_repeats)
            # Get the autocorrelation of the trajectory points inside the grid cell
            time_counts = (inside_idxs.sum(axis=1)/inside_idxs.sum())[1:]
            # Calculate the mean autocorrelation, weighted by the number of timepoints
            # inside the grid cell
            tgt_autocorr_grid[i,j] = (tgt_autocorr_t*time_counts).mean()
diff_grid = src_velo_var_grid - tgt_velo_var_grid

#%%
maxautocorr = max(src_autocorr_grid[~torch.isnan(src_autocorr_grid)].max(),
                    tgt_autocorr_grid[~torch.isnan(tgt_autocorr_grid)].max())
minautocorr = min(src_autocorr_grid[~torch.isnan(src_autocorr_grid)].min(),
                    tgt_autocorr_grid[~torch.isnan(tgt_autocorr_grid)].min())
#%%
plt.imshow(src_autocorr_grid.T, cmap='viridis', origin='lower',
              extent=(xmin, xmax, ymin, ymax), vmin=minautocorr, vmax=maxautocorr)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
plt.title(f'{source_genotype.capitalize()} Lag-1 Autocorrelation')
#%%
plt.imshow(tgt_autocorr_grid.T, cmap='viridis', origin='lower',
                extent=(xmin, xmax, ymin, ymax), vmin=minautocorr, vmax=maxautocorr)
plt.colorbar()
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.xlabel('PC1')
plt.ylabel('PC2')

# plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
plt.title(f'{target_genotype.capitalize()} Lag-1 Autocorrelation')
#%%
maxdiff = max(torch.abs(diff_grid[~diff_grid.isnan()].max()), 
              torch.abs(diff_grid[~diff_grid.isnan()].min()))
plt.imshow(diff_grid.T, cmap='coolwarm', origin='lower',
                extent=(xmin, xmax, ymin, ymax),
                vmin=-maxdiff, vmax=maxdiff)
plt.colorbar()
plt.title(f'Difference in Lag-1 Autocorrelation '
          f'{source_genotype.capitalize()} - {target_genotype.capitalize()}')

#%%
# Clear the cache
torch.cuda.empty_cache()
#%%
# Calculate the spatial divergence of the velocity model
# def divergence_num(model, x, h=1e-5):
#     with torch.no_grad():
#         d = torch.zeros_like(x[:,0])
#         for i in range(x.shape[1]):
#             m_i = model.models[i]
#             dx = torch.zeros_like(x)
#             dx[:,i] = h
#             velo_xp = m_i(x + dx)[:,0]
#             velo_xm = m_i(x - dx)[:,0]
#             d += ((velo_xp - velo_xm)/(2*h))
#         return d

def divergence_torch(model, x):
    div = torch.zeros_like(x[:,0])
    for i in range(x.shape[1]):
        m_i = model.models[i]
        v_i = m_i(x)[:,0]
        d = torch.autograd.grad(v_i, x, 
                                torch.ones_like(v_i), 
                                retain_graph=False, 
                                create_graph=False)[0][:,i].detach()
        div += d
    return div
#%%
all_X = torch.cat((src_X, tgt_X), dim=0).to(device)
all_X_np = util.tonp(all_X)
all_X_proj = pca.transform(all_X_np)

#%%
# Calculate the divergence of the velocity at each timepoint
src_div = torch.zeros_like(src_baseline_trajectories_tensor[:,:,0], device='cpu')
tgt_div = torch.zeros_like(src_baseline_trajectories_tensor[:,:,0], device='cpu')

for i in tqdm(range(len_tspan)):
    points = src_baseline_trajectories_tensor[i]
    points.requires_grad_(True)

    # Compute the gradient of the velocity with respect to the trajectory
    src_div_batch = divergence_torch(src_model, points)
    tgt_div_batch = divergence_torch(tgt_model, points)
    src_div[i] = src_div_batch.detach().cpu()
    tgt_div[i] = tgt_div_batch.detach().cpu()
    points.requires_grad_(False)
    del src_div_batch
    del tgt_div_batch
    del points

#%%
src_div = src_div.detach()
tgt_div = tgt_div.detach()
torch.cuda.empty_cache()

# %%
src_div_sample = src_div.reshape(len_tspan, -1)[:,::10].mean(axis=1)
tgt_div_sample = tgt_div.reshape(len_tspan, -1)[:,::10].mean(axis=1)
plt.plot(src_div_sample, label=source_genotype)
plt.plot(tgt_div_sample, label=target_genotype)
plt.legend()

#%%
# Make a grid in PCA space of the divergence of the velocity
ngrid = 100

xmin = src_trajectory_proj[:,0].min()
xmax = src_trajectory_proj[:,0].max()
ymin = src_trajectory_proj[:,1].min()
ymax = src_trajectory_proj[:,1].max()

xgrid_X = np.linspace(xmin, xmax, ngrid)
ygrid_X = np.linspace(ymin, ymax, ngrid)

# Flatten the divergence arrays
src_div_flat = src_div.flatten()
tgt_div_flat = tgt_div.flatten()

# Use np.histogram2d to bin the data
src_histogram, xedges, yedges = np.histogram2d(src_trajectory_proj[:,0], src_trajectory_proj[:,1], bins=[xgrid_X, ygrid_X], weights=src_div_flat)
tgt_histogram, _, _ = np.histogram2d(src_trajectory_proj[:,0], src_trajectory_proj[:,1], bins=[xgrid_X, ygrid_X], weights=tgt_div_flat)

# Calculate the mean of the bins
src_div_grid = src_histogram / np.histogram2d(src_trajectory_proj[:,0], src_trajectory_proj[:,1], bins=[xgrid_X, ygrid_X])[0]
tgt_div_grid = tgt_histogram / np.histogram2d(src_trajectory_proj[:,0], src_trajectory_proj[:,1], bins=[xgrid_X, ygrid_X])[0]

#%%
# Calculate the 99th percentile of the divergence
src_div_grid_values = src_div_grid[~np.isnan(src_div_grid)]
tgt_div_grid_values = tgt_div_grid[~np.isnan(tgt_div_grid)]
maxdiv = max(np.quantile(src_div_grid_values, 0.99), np.quantile(tgt_div_grid_values, 0.99))
mindiv = min(np.quantile(src_div_grid_values, 0.01), np.quantile(tgt_div_grid_values, 0.01))

xi = np.linspace(xmin, xmax, ngrid)
yi = np.linspace(ymin, ymax, ngrid)
xi, yi = np.meshgrid(xi, yi)
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#%%
plt.imshow(src_div_grid.T, cmap='viridis', origin='lower', 
           extent=(xmin, xmax, ymin, ymax), vmin=mindiv, vmax=maxdiv)
plt.colorbar()
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
plt.title(f'{source_genotype.capitalize()} Velocity Divergence')
# plt.scatter(all_X_proj[:,0], all_X_proj[:,1], c='grey', s=1, alpha=.8)

# %%
plt.imshow(tgt_div_grid.T, cmap='viridis', origin='lower',
              extent=(xmin, xmax, ymin, ymax), vmin=mindiv, vmax=maxdiv)
plt.colorbar()
plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys')
plt.title(f'{target_genotype.capitalize()} Velocity Divergence')
# plt.scatter(all_X_proj[:,0], all_X_proj[:,1], c='grey', s=1, alpha=.8)

# %%
diff_grid = src_div_grid - tgt_div_grid
# Center the diff grid on zero, excluding the 99th percentile
valid = lambda x: x[~np.isnan(x)]
all_diffs = diff_grid[~np.isnan(diff_grid)]
maxdiff = max(np.abs(np.quantile(all_diffs, 0.99)), np.abs(np.quantile(all_diffs, 0.01)))
plt.imshow(diff_grid.T, cmap='coolwarm', origin='lower',
           extent=(xmin, xmax, ymin, ymax), vmin=-maxdiff, vmax=maxdiff)
plt.colorbar()
# plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Greys', alpha=.5)
plt.title(f'{source_genotype.capitalize()} - {target_genotype.capitalize()} '
          'Velocity Divergence Difference')
# %%
# Plot the distributions of the number of zeros in the cells of the trajectory
src_zeros = tonp(src_baseline_trajectories_tensor == 0).mean(axis=(1,2), dtype=float)
tgt_zeros = tonp(tgt_baseline_trajectories_tensor == 0).mean(axis=(1,2), dtype=float)

fig, ax1 = plt.subplots()
ax1.plot(src_zeros, label=source_genotype)
ax1.plot(tgt_zeros, label=target_genotype)
ax1.set_xticks(np.linspace(0, len_tspan, num=20),
              [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=20)],
              rotation=45)
ax2 = plt.twinx()
ax2.plot(src_zeros - tgt_zeros, label='Difference', c='grey', alpha=.5)
ax2.axvline(decline_idx*10, c='green', alpha=.3)

ax1.set_ylabel('Fraction of Zeros')
ax1.set_xlabel('Time')
ax1.set_title('Fraction of Zeros in Trajectories')
ax2.set_ylabel('\nDifference in Fraction of Zeros')
# Add a legend with all the lines
artists = ax1.lines + ax2.lines
plt.legend(artists, [l.get_label() for l in artists])

plt.tight_layout()

# %%
# Subset to only the rescue genes
best_gene_combinations = []
# TODO use the label when we want to select only the VIM first transfers
label = ''
for file in glob(f'{datadir}/top_{label}{transfer}_combination*.pickle'):
    combo = pickle.load(open(file, 'rb'))
    best_gene_combinations.append(combo)
# Get the indexes of the genes in each combination
combo_idxs = []
for combo in best_gene_combinations:
    combo_idxs.append([node_to_idx[g] for g in combo])
# %%
# Compute the number of zeros in the trajectory for each gene combination
n_zeros = []

for combo in combo_idxs:
    n_zeros.append((src_baseline_trajectories_np[:,combo] == 0).mean(axis=(1,2)))
# %%
# Plot the number of zeros for each gene combination
for n in n_zeros:
    plt.plot(n, c='blue', alpha=.5)
plt.plot(src_zeros, label=source_genotype)
plt.plot(tgt_zeros, label=target_genotype)
# %%
# Compute the autocorrelation of the rescue genes
fig, axs = plt.subplots(1,2, figsize=(10,5))
maxautocorr = 0
n_combos = len(combo_idxs)
tgt_combo_autocorr = np.zeros((n_combos, len_tspan-1))
src_combo_autocorr = np.zeros((n_combos, len_tspan-1))
for i,combo in enumerate(combo_idxs):
    tgt_autocorr_combo = autocorr(tgt_baseline_trajectories_tensor[:,:,combo], lag=1).abs()
    src_autocorr_combo = autocorr(src_baseline_trajectories_tensor[:,:,combo], lag=1).abs()
    src_autocorr_combo_time = src_autocorr_combo.mean(dim=(1,2))
    tgt_autocorr_combo_time = tgt_autocorr_combo.mean(dim=(1,2))
    src_combo_autocorr[i] = tonp(src_autocorr_combo_time)
    tgt_combo_autocorr[i] = tonp(tgt_autocorr_combo_time)
    # src_autocorr_combo = autocorr(src_baseline_trajectories_tensor[:,:,combo], lag=1).abs()
    # tgt_autocorr_combo = autocorr(tgt_baseline_trajectories_tensor[:,:,combo], lag=1).abs()
    maxautocorr = max(maxautocorr, src_autocorr_combo_time.max(), tgt_autocorr_combo_time.max())
    axs[0].plot(tonp(src_autocorr_combo_time), c='grey', alpha=.5, label=source_genotype)
    axs[1].plot(tonp(tgt_autocorr_combo_time), c='grey', alpha=.5, label=target_genotype)
    # axs[2].plot(tonp(src_autocorr_mean - tgt_autocorr_mean), c='grey', alpha=.5, label='Difference')
    axs[0].axvline(decline_idx*10, c='green', alpha=.3)
    axs[1].axvline(decline_idx*10, c='green', alpha=.3)
    # plt.legend()
# Make both plots have the same y axis range
maxautocorr = tonp(maxautocorr)
axs[0].set_ylim(0, maxautocorr) 
axs[1].set_ylim(0, maxautocorr)
axs[0].set_ylabel('Autocorrelation')
axs[0].set_xlabel('Time')
axs[0].set_title(f'{source_genotype.capitalize()}')
axs[1].set_title(f'{target_genotype.capitalize()}')
plt.suptitle('Autocorrelation of Rescue Gene Sets')

#%%
# Heatmap of the autocorrelation of the rescue genes
fig, axs = plt.subplots(1,2, figsize=(10,5))

axs[0].imshow(tgt_combo_autocorr, cmap='viridis', aspect='auto', interpolation='none')
axs[0].axvline(decline_idx*10, c='white', alpha=.8)
axs[1].imshow(src_combo_autocorr, cmap='viridis', aspect='auto', interpolation='none')
axs[1].axvline(decline_idx*10, c='white', alpha=.8)

#%%
# Plot the difference in autocorrelation of the highest mean genes
fig, ax = plt.subplots(1,2, figsize=(10,5))
combo_autocorr_diff = src_combo_autocorr - tgt_combo_autocorr
for i in range(len(combo_autocorr_diff)):
    ax[0].plot(combo_autocorr_diff[i], c='grey', alpha=.5, label='Difference')
    ax[0].axvline(decline_idx*10, c='green', alpha=.3)
ax[0].axhline(0, c='blue', alpha=.3)
ax[0].set_ylabel('Autocorrelation Difference')
ax[0].set_xlabel('Time')
plt.suptitle(f'Difference in Autocorrelation of Rescue Gene Sets '
             f' ({source_genotype.capitalize()} - {target_genotype.capitalize()})')
maxdiff = max(abs(combo_autocorr_diff.max()), abs(combo_autocorr_diff.min()))
ax[1].imshow(combo_autocorr_diff, cmap='coolwarm', aspect='auto', interpolation='none',
             vmin=-maxdiff, vmax=maxdiff)
ax[1].axvline(decline_idx*10, c='black', alpha=1.0)
ax[1].set_ylabel('Gene Sets')
ax[1].set_xlabel('Time')
# TODO fix the x labels to have the actual time values
plt.colorbar(ax[1].images[0], ax=ax[1])
#%%
# Get random gene sets the size of the rescue gene sets
random_combos = []
n_genes = src_baseline_trajectories_tensor.shape[2]
for combo in combo_idxs:
    random_combo = np.random.choice(np.arange(n_genes), size=len(combo))
    random_combos.append(random_combo)
# Compute the autocorrelation of the random gene sets
src_random_autocorr = np.zeros((len(combo_idxs), len_tspan-1))
tgt_random_autocorr = np.zeros((len(combo_idxs), len_tspan-1))
repeat = 100
for repeat in tqdm(range(repeat)):
    for i,random_combo in enumerate(random_combos):
        src_autocorr_combo = autocorr(src_baseline_trajectories_tensor[:,:,random_combo], lag=1).abs()
        tgt_autocorr_combo = autocorr(tgt_baseline_trajectories_tensor[:,:,random_combo], lag=1).abs()
        src_autocorr_random = src_autocorr_combo.mean(dim=(1,2))
        tgt_autocorr_random = tgt_autocorr_combo.mean(dim=(1,2))
        src_random_autocorr[i] += tonp(src_autocorr_random)
        tgt_random_autocorr[i] += tonp(tgt_autocorr_random)
#%%
# Plot the difference in autocorrelation of the random gene sets
plt.imshow((src_random_autocorr - tgt_random_autocorr)/repeat, cmap='coolwarm', aspect='auto', interpolation='none')
plt.axvline(decline_idx*10, c='black', alpha=.9)
plt.xticks(np.linspace(0, len_tspan, num=10),
              [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)])
plt.colorbar()
plt.title('Average Difference in Autocorrelation of Random Gene Sets')

# %%
# Get only the highest mean genes
n_genes = src_X.shape[1]
src_unrolled_trajectories = src_baseline_trajectories_tensor.reshape((-1, n_genes))
tgt_unrolled_trajectories = tgt_baseline_trajectories_tensor.reshape((-1, n_genes))
src_top_genes = src_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:10]
tgt_top_genes = tgt_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:10]

# Plot the autocorrelation of the highest mean genes
top_ns = np.arange(1, 150, 5)

top_n_autocorr = np.zeros((len(top_ns), len_tspan-1))
top_n_pct_zeros = np.zeros((len(top_ns), len_tspan))

fig = plt.figure(figsize=(7,12))

gs = GridSpec(3,2, width_ratios=[1,.05])
axs = np.zeros((gs.nrows, gs.ncols), dtype=object)
for i in range(gs.nrows):
    for j in range(gs.ncols):
        ax = fig.add_subplot(gs[i,j])
        axs[i,j] = ax

for i,top_n in enumerate(top_ns):
    # src_var_genes = src_unrolled_trajectories.var(dim=0).argsort(descending=True)[:10]
    tgt_top_genes = tonp(tgt_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:top_n])

    # Compute the autocorrelation of the highest mean genes
    top_n_autocorr[i] = tgt_autocorr_genes[:,tgt_top_genes].mean(axis=1)
    pct_zero = (tgt_baseline_trajectories_tensor[:,:,tgt_top_genes] == 0).mean(dim=(1,2), dtype=torch.float32)
    top_n_pct_zeros[i] = tonp(pct_zero)
    axs[1,0].plot(top_n_autocorr[i], c=plasma(top_n/150), label=top_n)

    # plt.plot(tonp(src_autocorr_mean - tgt_autocorr_mean), c='grey', alpha=.5, label='Difference')
axs[0,0].imshow(top_n_autocorr, cmap='viridis', aspect='auto', interpolation='none')
axs[0,0].axvline(decline_idx*10, c='white', alpha=.8)
axs[0,0].set_yticks(np.arange(len(top_ns)), top_ns)
axs[0,0].set_xlabel('Time')
axs[0,0].set_title('Autocorrelation of Top N Genes by Mean Expression')
axs[0,0].set_ylabel('Number of Genes')
for i in range(2):
    axs[i,0].set_xticks(np.linspace(0, len_tspan, num=10),
                       [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)])

plt.colorbar(axs[0,0].images[0], cax=axs[0,1])
axs[1,0].set_xlabel('Time')
axs[1,0].set_ylabel('Autocorrelation')
axs[1,0].set_title('Autocorrelation of Top N Genes by Mean Expression')
# Make a colorbar for the plasma colormap used in the second plot
axs[1,0].axvline(decline_idx*10, c='green', alpha=.3)
axs[1,1].imshow(np.linspace(0,1,256).reshape(-1,1), cmap='plasma', aspect='auto')
axs[1,1].set_yticks(np.linspace(0,256,15),
                    [f'{int(t)}' for t in np.linspace(0,150,15)]);
axs[1,1].set_ylabel('Number of Genes')
axs[1,1].set_xticks([])
axs[2,0].plot(top_n_pct_zeros.mean(axis=1), marker='o')
axs[2,0].set_title('Proportion of Zeros in Top N Genes')
axs[2,0].set_xlabel('Number of Genes')
axs[2,0].set_ylabel('Proportion Zeros')
nticks = 10
xtick_labels = [f'{int(t)}' for t in np.linspace(top_ns[0], top_ns[-1], nticks)]
axs[2,0].set_xticks(np.linspace(0,len(top_ns), nticks), 
                    xtick_labels)

# Delete the last subplot
fig.delaxes(axs[2,1])
fig.tight_layout()

#%%
# Plot a heatmap of the autocorrelation of the highest mean genes
n_genes = 10
src_top_genes = src_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:n_genes]
tgt_top_genes = tgt_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:n_genes]
src_autocorr = src_autocorr_genes[:,tonp(tgt_top_genes)]
tgt_autocorr = tgt_autocorr_genes[:,tonp(tgt_top_genes)]
# fig, axs = plt.subplots(2,2, figsize=(10,5))
fig = plt.figure(figsize=(10,5))

gs = GridSpec(2,3, width_ratios=[1,1,.05])
axs = np.zeros((gs.nrows, gs.ncols), dtype=object)
for i in range(gs.nrows):
    for j in range(gs.ncols):
        ax = fig.add_subplot(gs[i,j])
        axs[i,j] = ax

min_autocorr = min(src_autocorr.min(), tgt_autocorr.min())
max_autocorr = max(src_autocorr.max(), tgt_autocorr.max())
axs[0,0].imshow(src_autocorr.T, cmap='viridis', aspect='auto',
           interpolation='none', vmin=min_autocorr, vmax=max_autocorr)

axs[0,0].axvline(decline_idx*10, c='white', alpha=1)
axs[0,0].set_title(f'{source_genotype.capitalize()} Absolute Autocorrelation')
axs[0,1].imshow(tgt_autocorr.T, cmap='viridis', aspect='auto',
           interpolation='none', vmin=min_autocorr, vmax=max_autocorr)
axs[0,1].set_yticks([])
axs[0,1].axvline(decline_idx*10, c='white', alpha=1)

axs[0,1].set_title(f'{target_genotype.capitalize()} Absolute Autocorrelation')


axs[1,0].imshow((src_autocorr.T / src_autocorr.max(axis=0)[:,None]), cmap='plasma', aspect='auto',
           interpolation='none', vmin=0, vmax=1)

axs[1,0].axvline(decline_idx*10, c='white', alpha=1)
axs[1,0].set_title(f'{source_genotype.capitalize()} Normalized Autocorrelation')
axs[1,1].imshow((tgt_autocorr.T / tgt_autocorr.max(axis=0)[:,None]), cmap='plasma', aspect='auto',
           interpolation='none', vmin=0, vmax=1)
axs[1,1].set_yticks([])
axs[1,1].axvline(decline_idx*10, c='white', alpha=1)

axs[1,1].set_title(f'{target_genotype.capitalize()} Normalized Autocorrelation')

for ax in axs[:,0]:
    ax.set_yticks(np.arange(10), 
            [protein_id_name[idx_to_node[int(i)]] 
                for i in src_top_genes]);

for ax in axs[0,:]:
    ax.set_xticks([])
for ax in axs[1,:]:
    ax.set_xticks(np.linspace(0, len_tspan, num=10),
                [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)],
                rotation=45)
# fig.subplots_adjust(right=0.8)
# cbar_ax0 = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# Create a scalar mappable for the colorbar that maps [min_autocorr, max_autocorr] to [0,1]
import matplotlib
for ax in axs[:, 2]:
    ax.set_xticks([])

norm = matplotlib.colors.Normalize(vmin=float(min_autocorr), vmax=float(max_autocorr))
scm = ScalarMappable(norm=norm, cmap=viridis)
fig.colorbar(scm, cax=axs[0,2], label='Absolute Autocorrelation')

norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
scm = ScalarMappable(norm=norm, cmap=plasma)
fig.colorbar(scm, cax=axs[1,2], label='Relative Autocorrelation')

plt.tight_layout()
#%%
fig, axs = plt.subplots(1,1, figsize=(7,5))

autocorr_diff = src_autocorr.T - tgt_autocorr.T
max_diff = max(abs(autocorr_diff.max()), abs(autocorr_diff.min()))

axs.imshow(autocorr_diff, cmap='coolwarm', aspect='auto',
              interpolation='none', vmin=-max_diff, vmax=max_diff)
axs.set_yticks([])
axs.axvline(decline_idx*10, c='green', alpha=.3)
axs.set_title(f'Difference in Absolute Autocorrelation'
              f' ({source_genotype.capitalize()} - {target_genotype.capitalize()})')
axs.set_xticks(np.linspace(0, len_tspan, num=10),
                [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)],
                rotation=45)
axs.set_yticks(np.arange(10),
                [protein_id_name[idx_to_node[int(i)]] 
                 for i in src_top_genes]);

fig.colorbar(axs.images[0], ax=axs)
# plt.tight_layout()
#%%
n_genes = 10
src_top_genes = tonp(src_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:n_genes])
tgt_top_genes = tonp(tgt_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:n_genes])

blues = plt.get_cmap('Blues')
for i in range(n_genes):
    gene_name = protein_id_name[idx_to_node[int(src_top_genes[i])]]
    plt.plot(src_autocorr_genes[:,src_top_genes[i]], 
             label=gene_name, 
             c=blues(1 - i/n_genes), 
             alpha=.5)
plt.axvline(decline_idx*10, c='orange', alpha=.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.ylabel('Autocorrelation')
plt.xlabel('Time')
plt.xticks(np.linspace(0, len_tspan, num=10),
              [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)]);
#%%
# Plot the expression of the highest mean genes in a heatmap
fig, axs = plt.subplots(1,2, figsize=(10,5))
min_expression = min(src_baseline_trajectories[:,:,src_top_genes].min(),
                    tgt_baseline_trajectories[:,:,tgt_top_genes].min())
max_expression = max(src_baseline_trajectories[:,:,src_top_genes].max(),
                    tgt_baseline_trajectories[:,:,tgt_top_genes].max())
axs[0].imshow(src_baseline_trajectories.mean(axis=1)[:,src_top_genes].T,
              cmap='viridis', aspect='auto', interpolation='none',
              vmin=min_expression, vmax=max_expression)
axs[0].set_title(f'{source_genotype.capitalize()} Expression of Top {n_genes} Genes')
axs[1].imshow(tgt_baseline_trajectories.mean(axis=1)[:,tgt_top_genes].T,
              cmap='viridis', aspect='auto', interpolation='none',
              vmin=min_expression, vmax=max_expression)
axs[1].set_title(f'{target_genotype.capitalize()} Expression of Top {n_genes} Genes')
for ax in axs:
    ax.set_xticks(np.linspace(0, len_tspan, num=10),
                    [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)],
                    rotation=45)
    ax.axvline(decline_idx*10, c='white', alpha=.8)
axs[0].set_yticks(np.arange(n_genes),
                [protein_id_name[idx_to_node[int(i)]]
                 for i in src_top_genes]);
axs[1].set_yticks([])

plt.tight_layout()
#%%
# Plot the difference in expression of the highest mean genes in a heatmap
fig, axs = plt.subplots(1,1, figsize=(7,5))
expression_diff = tonp(src_baseline_trajectories_tensor[:,:,src_top_genes].T.mean(dim=1) - 
                       tgt_baseline_trajectories_tensor[:,:,tgt_top_genes].T.mean(dim=1))
max_diff = max(abs(expression_diff.max()), abs(expression_diff.min()))
axs.imshow(expression_diff, cmap='coolwarm', aspect='auto', interpolation='none',
              vmin=-max_diff, vmax=max_diff)
axs.set_title(f'Difference in Simulated Expression of Top {n_genes} Genes'
                f' ({source_genotype.capitalize()} - {target_genotype.capitalize()})')
axs.axvline(decline_idx*10, c='green', alpha=.3)
axs.set_xticks(np.linspace(0, len_tspan, num=10),
                [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)],
                rotation=45)
axs.set_yticks(np.arange(n_genes),
                [protein_id_name[idx_to_node[int(i)]] 
                 for i in src_top_genes]);
plt.colorbar(axs.images[0], ax=axs)

#%%
# Get the expression of the top 12 expressed genes over time in a heatmap
nrow = 4
ncol = 3
n_genes = nrow*ncol
src_mean_genes = src_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:n_genes]
tgt_mean_genes = tgt_unrolled_trajectories.mean(dim=0).argsort(descending=True)[:n_genes]

fig, axs = plt.subplots(nrow, ncol, figsize=(10,13))
for i,idx in enumerate(tgt_mean_genes):    
    ax = axs[i//ncol, i%ncol]
    expression = tonp(tgt_baseline_trajectories_tensor[:,:,idx])
    n_repeats = tgt_baseline_trajectories_tensor.shape[1]
    # Get the histogram of the expression at different timepoints
    for t in np.linspace(0, len_tspan-1, num=5, dtype=int):
        hist,bins = np.histogram(expression[t], bins=100)
        w = np.diff(bins)
        hist = hist/(n_repeats*5)
        ax.bar(height=hist, x=bins[:-1], width=w, color=viridis(t/len_tspan), alpha=.4)
    ax.set_title(f'{i+1} - {protein_id_name[idx_to_node[int(idx)]]}')
plt.tight_layout()
#%%        
artists = [Rectangle((0,0),1,1,fc=viridis(t), edgecolor='none', linewidth=0) for t in np.linspace(0,1,5)]
plt.legend(artists,
           [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=5)], 
           title='Time')
plt.axis('off')

# %%
# Cluster the autocorrelation of all the genes using hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

# src_autocorr = autocorr(src_baseline_trajectories_tensor, lag=1).abs().mean(dim=1)
tgt_rel_autocorr = tgt_autocorr_genes / tgt_autocorr_genes.max(axis=0)[None,:]
tgt_rel_autocorr[np.isnan(tgt_rel_autocorr)] = 0
# tgt_rel_autocorr = tgt_autocorr_genes

n_clusters = 20
clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(tgt_rel_autocorr.T)

cluster_labels = clustering.labels_

#%%
# Rank genes based on the distance from their minimum autocorrelation to the critical point
# Reshape the autocorrelation according to the clustering labels
clustered_autocorr = np.zeros((n_clusters, len_tspan-1))
for i in range(n_clusters):
    clustered_autocorr[i] = tgt_rel_autocorr[:,cluster_labels == i].mean(axis=1)[:len_tspan-1]

# Order the clusters by the distance from their minimum autocorrelation to the critical point
cluster_distances = np.zeros(n_clusters)
for i in range(n_clusters):
    cluster_min_idx = np.argmin(clustered_autocorr[i])
    cluster_distance = abs(cluster_min_idx - decline_idx*10)
    cluster_distances[i] = cluster_distance
    print(cluster_min_idx, decline_idx*10, cluster_distance)
cluster_distance_sort = np.argsort(cluster_distances)
# Plot the clusters as a heatmap
# %%
fig, axs = plt.subplots(1,2, figsize=(20,10))


top_n = 3
for i in range(top_n, n_clusters):
    axs[1].plot(clustered_autocorr[cluster_distance_sort][i], c='grey', alpha=.3)
for i in range(top_n):
    axs[1].plot(clustered_autocorr[cluster_distance_sort][i], label=f'Cluster {cluster_distance_sort[i]}')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
axs[1].axvline(decline_idx*10, c='black', alpha=.9)

top_cluster_gene_autocorr = []
top_cluster_gene_idxs = []
top_cluster_sizes = []
# Order the genes by their cluster
for i in range(top_n):
    cluster_gene_idxs = np.where(cluster_labels == cluster_distance_sort[i])[0]
    top_cluster_sizes.append(len(cluster_gene_idxs))
    top_cluster_gene_idxs.append(cluster_gene_idxs)
    top_cluster_gene_autocorr.append(tgt_rel_autocorr[:,cluster_gene_idxs])
top_cluster_gene_autocorr = np.concatenate(top_cluster_gene_autocorr, axis=1)
top_cluster_gene_idxs = np.concatenate(top_cluster_gene_idxs)

axs[0].imshow(top_cluster_gene_autocorr.T, cmap='viridis', aspect='auto', interpolation='none')
axs[0].axvline(decline_idx*10, c='white', alpha=1)

# Annotate the clusters on the right side of the heatmap
from matplotlib.lines import Line2D
start = 0 

tab10 = plt.get_cmap('tab10')

for i in range(top_n):
    end = start + top_cluster_sizes[i]
    line = axs[0].add_line(Line2D([len_tspan*1.02, len_tspan*1.02], 
                                  [start-.1, end-.9], linewidth=5,
                            c='black', alpha=1))
    line.set_clip_on(False)
    axs[0].text(len_tspan*1.07, (start+end-1)*.5, f'C{cluster_distance_sort[i]}',
                ha='center', va='center', rotation=90)
    axs[0].axhline(end-1+.5, c='black', alpha=.9)
    start = end

axs[0].set_yticks(np.arange(len(top_cluster_gene_idxs)),
                [protein_id_name[idx_to_node[int(i)]]
                 for i in top_cluster_gene_idxs]);
for ax in axs:
    ax.set_xticks(np.linspace(0, len_tspan, num=10),
                    [f'{int(t)}' for t in np.linspace(0, len_trajectory, num=10)],
                    rotation=45);
axs[0].set_title('Normalized Autocorrelation of Genes')
axs[1].set_title('Mean Normalized Autocorrelation of Cluster')
fig.suptitle('Gene clusters with minimum autocorrelation near the critical point', fontsize=28)
plt.tight_layout()

#%%
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.imshow(clustered_autocorr[cluster_distance_sort], cmap='viridis', aspect='auto', interpolation='none')

#%%
# Inspect the genes in each cluster
for i in cluster_distance_sort:
    print(f'Cluster {i}')
    for idx in np.where(cluster_labels == i)[0]:
        gene_name = protein_id_name[idx_to_node[int(idx)]]
        print(gene_name, end=', ')
    print('\n','-'*20)


# %%
# Get the POU5F1 gene index
gene = 'POU5F1'
gene_idx = node_to_idx[protein_name_id[gene]]
tgt_gene_autocorr = tgt_autocorr_genes[:,gene_idx]
src_gene_autocorr = src_autocorr_genes[:,gene_idx]

# Plot the autocorrelation of gene
plt.plot(src_gene_autocorr, label=source_genotype)
plt.plot(tgt_gene_autocorr, label=target_genotype)
plt.axvline(decline_idx*10, c='green', alpha=.3)
plt.legend()
plt.title(f'{gene} Autocorrelation')
# %%
# Genes with the biggest difference in autocorrelation
diff_autocorr = (src_autocorr_genes - tgt_autocorr_genes)
total_diff_autocorr = diff_autocorr.mean(axis=0)
sorted_diff_autocorr = total_diff_autocorr.argsort()[::-1]
for idx in sorted_diff_autocorr[:20]:
    gene_name = protein_id_name[idx_to_node[int(idx)]]
    print(gene_name, total_diff_autocorr[idx])

# %%
# Plot differences in autocorrelation for the top 20 genes with the biggest difference
fig, ax = plt.subplots(1,1, figsize=(7,5))
plt.imshow(diff_autocorr[:,sorted_diff_autocorr[:20]].T, 
           cmap='coolwarm', aspect='auto', interpolation='none')
plt.axvline(decline_idx*10, c='black', alpha=1)
plt.yticks(np.arange(20),
            [protein_id_name[idx_to_node[int(i)]] 
             for i in sorted_diff_autocorr[:20]]);
plt.colorbar()

#%%
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].imshow(src_autocorr_genes[:,sorted_diff_autocorr[:20]].T,
              cmap='viridis', aspect='auto', interpolation='none')
axs[1].imshow(tgt_autocorr_genes[:,sorted_diff_autocorr[:20]].T,
                cmap='viridis', aspect='auto', interpolation='none')
# %%
