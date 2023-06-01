#%%
%load_ext autoreload
%autoreload 2
#%%
import scanpy as sc
# import numpy as np
import pickle
from flow_model import GroupL1FlowModel
import torch
from tqdm import tqdm
import sys
sys.path.append('..')
import util
import matplotlib
# %%
genotype = 'wildtype'
adata = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')

# %%
# Load the models
tmstp = '20230601_143356'
outdir = f'../../output/{tmstp}'
models = pickle.load(open(f'{outdir}/models/group_l1_variance_model_wildtype.pickle', 'rb'))

# %%
X = torch.tensor(adata.X.toarray()).float()

#%%
def euler_step(x, model, dt):
    with torch.no_grad():
        dx, var = model(x)
        var[var < 0] = 0
        std = torch.sqrt(var)
        dx += torch.randn_like(dx) * std
        #*****************************
        # TODO why is this negative?
        #*****************************
        return x - dt*dx

def trajectory(x, t_span, model):
    with torch.no_grad():
        x = x.clone()
        traj = torch.zeros(len(t_span), x.shape[0], x.shape[1])
        traj[0,:,:] = x
        last_t = t_span[0]
        for i,t in enumerate(t_span[1:]):
            dt = t - last_t
            x = euler_step(x, model, dt=dt)
            last_t = t
            traj[i+1] = x

        return traj

# %%
import numpy as np
from sklearn.decomposition import PCA
cell_types = {c:i for i,c in enumerate(set(adata.obs['cell_type']))}
proj = np.array(adata.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = adata.uns['pca_mean']
pca.components_ = adata.uns['PCs']
proj = np.array(pca.transform(X))[:,0:2]
T = adata.obsm['transition_matrix']

V = util.velocity_vectors(T, X)
V_emb = util.embed_velocity(X, V, lambda x: np.array(pca.transform(x)[:,0:2]))

# %%
torch.set_num_threads(24)
start_idxs = adata.uns['initial_points_via']
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
len_trajectory = 100
num_trajectories = starts.shape[0]
t_span = torch.linspace(0,100,100, device='cpu')
trajectories = trajectory(starts, t_span, model)
#%%
from matplotlib import pyplot as plt
t = util.tonp(trajectories)
t_emb = pca.transform(t.reshape(-1, t.shape[2]))[:,:2]
t_emb = t_emb.reshape(t.shape[0], t.shape[1], -1)
# Connect the scatter plot points
# Color the points by the distance along the trajectory
colormap = matplotlib.colormaps.get_cmap('gnuplot')
colors = colormap(np.arange(len_trajectory)/len_trajectory)
cell_colors = adata.obs['cell_type'].map(cell_types)
# plt.scatter(proj[:,0], proj[:,1], s=.1,
#             c=cell_colors, cmap='tab20c')
for i in range(t_emb.shape[1]):
    # plt.plot(t_emb[:,i,0], t_emb[:,i,1], color='black', linewidth=.5, alpha=.1)
    plt.scatter(t_emb[:,i,0], t_emb[:,i,1],
                color=colors, 
                marker='o', s=.5, alpha=0.5)

#%%
# Get a color map with len_trajectory colors
import random
# Plot a scatter plot showing individual trajectories
# Get a 4x4 grid of plots
nrow = 3
ncol = 3
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20,20))
# Choose a random sample of 16 trajectories
for i,idx in enumerate(random.sample(range(num_trajectories), nrow*ncol)):
    # Plot the scatter plot
    ax = axs[i//ncol, i%ncol]
    ax.scatter(proj[:,0], proj[:,1], color='grey', alpha=1, s=.1)
    # Plot the trajectory
    ax.plot(t_emb[:,idx,0], t_emb[:,idx,1], color='black', linewidth=.5)
    ax.scatter(t_emb[:,idx,0], t_emb[:,idx,1], c=colors, 
               marker='o', s=5, zorder=100)
    # Remove the axis labels
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f'{genotype.capitalize()} sampled trajectories', fontsize=30)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
#%%
# Get a list of the cell type names
cell_types = {name:i for i,name in enumerate(sorted(adata.obs['cell_type'].unique()))}
num_cell_types = len(cell_types)
#%%
# Create a KDTree to find the nearest neighbor of each point
from sklearn.neighbors import KDTree
tree = KDTree(X)
#%%

cell_type_trajectories = np.zeros((len_trajectory, num_cell_types), dtype=int)
for i in range(len_trajectory):
    for j in range(num_trajectories):
        nearest_cell = tree.query(trajectories[i,j,:].reshape(1,-1))[1][0][0]
        cell_type = adata.obs['cell_type'][nearest_cell]
        cell_type_trajectories[i,cell_types[cell_type]] += 1
    
# %%
plt.imshow(cell_type_trajectories.T, aspect='auto', cmap='Blues', interpolation='none')
# Label the y-axis with the cell type names
plt.yticks(range(num_cell_types), cell_types);
plt.title(f'{genotype.capitalize()} cell type proportion in trajectories')
# plt.savefig(f'../figures/{genotype}_celltype_trajectories.png', dpi=300)

# plt.plot(t_emb[i,0], t_emb[i,1],
# plt.scatter(proj[start_idxs[0],0], proj[start_idxs[0],1], c='r', s=1)
# %%
minX = np.min(proj[:,0])
maxX = np.max(proj[:,0])
minY = np.min(proj[:,1])
maxY = np.max(proj[:,1])
xbuf = (maxX - minX) * 0.05
ybuf = (maxY - minY) * 0.05
n_points = 20
x_grid_points = np.linspace(minX, maxX, n_points)
y_grid_points = np.linspace(minY, maxY, n_points)
x_spacing = x_grid_points[1] - x_grid_points[0]
y_spacing = y_grid_points[1] - y_grid_points[0]
grid = np.array(np.meshgrid(x_grid_points, y_grid_points)).T.reshape(-1,2)
velocity_grid = np.zeros_like(grid)
velocity_grid[:] = np.nan
proj = np.array(pca.transform(adata.X.toarray()))[:,0:2]

#%%
# Find points inside each grid cell
velocities = torch.zeros(len(grid), X.shape[1])
variances = torch.zeros(len(grid), X.shape[1])
grid_means = torch.zeros(len(grid), X.shape[1])
for i,(x,y) in enumerate(grid):
    idx = np.where((proj[:,0] > x) & (proj[:,0] < x+x_spacing) & 
                    (proj[:,1] > y) & (proj[:,1] < y+y_spacing))[0]
    if len(idx) > 0:
        # Get the average velocity vector
        velo, var = model(X[idx,:])
        velocities[i] = velo.mean(axis=0).reshape(-1)
        variances[i] = var.mean(axis=0).reshape(-1)
        grid_means[i] = X[idx,:].mean(axis=0)
velocities = util.tonp(velocities)
variances = util.tonp(variances)
#%%
pca_embed = lambda x: np.array(pca.transform(x)[:,0:2])
velocity_grid = util.embed_velocity(grid_means, velocities, pca_embed)
var_upper_grid = util.embed_velocity(grid_means, velocities + variances, pca_embed)
var_lower_grid = util.embed_velocity(grid_means, velocities - variances, pca_embed)
#%%
# Normalize the variance vectors to have length = grid spacing
var_upper_grid = var_upper_grid / np.linalg.norm(var_upper_grid, axis=1).reshape(-1,1)
var_lower_grid = var_lower_grid / np.linalg.norm(var_lower_grid, axis=1).reshape(-1,1)
var_upper_grid[np.isnan(var_upper_grid)] = 0.0
var_lower_grid[np.isnan(var_lower_grid)] = 0.0
var_upper_grid *= x_spacing/2
var_lower_grid *= x_spacing/2

# %%
fig, axs = plt.subplots(1,1, figsize=(10,10))

ax = axs
cell_labels = [cell_types[c] for c in adata.obs['cell_type']]
# Make the scatter plot have the same range for both genotypes
# ax.scatter(proj[:,0], proj[:,1], s=1, alpha=0.2, c=cell_labels, cmap='tab20')
ax.set_xlim(minX-xbuf, maxX+xbuf)
ax.set_ylim(minY-ybuf, maxY+ybuf)
ax.set_facecolor('grey')
# add a grid to the plot
for x in x_grid_points:
    ax.axvline(x, color='white', alpha=0.1)
for y in y_grid_points:
    ax.axhline(y, color='white', alpha=0.1)
#Plot the velocity vectors
ax.scatter(proj[:,0], proj[:,1], c=cell_colors, s=.7)
for i in range(len(grid)):
    velo = velocity_grid[i,:]
    if np.abs(velo).sum() == 0:
        continue
    x,y = grid[i,:]
    # Color the arrows by the cosine similarity

    ax.arrow(x + x_spacing/2, y + y_spacing/2, velo[0], velo[1], 
             width=0.025, head_width=0.1, head_length=0.05, 
             color='orange')
    # Plot the variance
    # ax.arrow(x + x_spacing/2, y + y_spacing/2, var_upper_grid[i,0], var_upper_grid[i,1],
    #          width=0.0125, head_width=0.05, head_length=0.05,
    #          color='red')
    # ax.arrow(x + x_spacing/2, y + y_spacing/2, var_lower_grid[i,0], var_lower_grid[i,1],
    #             width=0.0125, head_width=0.05, head_length=0.05,
    #             color='red')

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'{genotype.capitalize()}', fontsize=14);
#%%
t = util.tonp(traj)
t_emb = pca.transform(t[:,0,:])
# Connect the scatter plot points
ax.plot(t_emb[:,0], t_emb[:,1], color='black', linewidth=.5)
ax.scatter(t_emb[:,0], t_emb[:,1],
            color=colormap(range(len(t_emb))), marker='o', s=3, zorder=100)
ax.legend()
plt.tight_layout()
# %%
