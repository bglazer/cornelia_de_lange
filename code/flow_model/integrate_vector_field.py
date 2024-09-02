# %%
%load_ext autoreload
%autoreload 2
# %%
import scanpy as sc
import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from flow_model import GroupL1FlowModel
from itertools import product
from potential import Potential

# %%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# %%
device = torch.device('cuda:0')

# %%
def load_data(path, device):
    data = sc.read_h5ad(path)
    X_np = data.X.toarray()
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    return data, X

# %%
def load_model(model_path, input_dim, hidden_dim, num_layers, device):
    model = GroupL1FlowModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, predict_var=True)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

# %%
def load_trajectories(path, device):
    trajectories_np = pickle.load(open(path, 'rb'))
    trajectories = torch.tensor(trajectories_np, dtype=torch.float32, device=device)
    return trajectories

# %%
def estimate_potential(vector_field_model, potential, all_points, optimizer, epochs=1000, num_points=1000):
    # all_points = trajectories.reshape(-1, trajectories.shape[2])
    for epoch in range(epochs):
        optimizer.zero_grad()
        idx = np.random.choice(all_points.shape[0], num_points)
        x = all_points[idx,:]
        x.requires_grad = True

        p = potential(x)
        grad_p = torch.autograd.grad(p, x, torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        u = vector_field_model(x)[0]
        loss = torch.nn.functional.mse_loss(grad_p, u)
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        loss.backward()
        optimizer.step()

    return potential

# %%
def initialize_potential(input_dim, hidden_dim, num_layers, device):
    potential = Potential(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    potential.to(device)
    optimizer = torch.optim.Adam(potential.parameters(), lr=1e-3)
    return potential, optimizer

# %%
def compute_grid_points(pca, x, n_grids):
    minX, maxX = np.inf, -np.inf
    minY, maxY = np.inf, -np.inf
    
    # Compute projections and find global min/max
    proj = pca.transform(to_np(x))
    minX = np.min(proj[:, 0])
    maxX = np.max(proj[:, 0])
    minY = np.min(proj[:, 1])
    maxY = np.max(proj[:, 1])
    
    # Buffer zones
    xbuf = (maxX - minX) * 0.05
    ybuf = (maxY - minY) * 0.05
    
    # Create common grid points
    x_grid_points = np.linspace(minX - xbuf, maxX + xbuf, n_grids)
    y_grid_points = np.linspace(minY - ybuf, maxY + ybuf, n_grids)
    grid_points = (x_grid_points, y_grid_points)
    
    return proj, grid_points


# %%
def bin_and_aggregate(x_bin_indices, y_bin_indices, p, n_grids):
    potential_grid = np.full(n_grids**2, np.nan)
    grid_counts = np.zeros(n_grids**2)

    for i, (x, y) in enumerate(product(range(n_grids), range(n_grids))):
        idx = np.where((x_bin_indices == x) & (y_bin_indices == y))[0]
        if len(idx) > 100:
            p_grid_mean = p[idx, :].mean(axis=0).reshape(-1)
            potential_grid[i] = p_grid_mean.item()
            grid_counts[i] = len(idx)

    grid_counts /= grid_counts.max()
    return potential_grid, grid_counts

# %%
def plot_grid(potential_grid, grid_points, label, cmap, c_range):
    fig, ax = plt.subplots(figsize=(10, 10))

    x_grid_points, y_grid_points = grid_points
    ax.set_facecolor('white')

    for x in x_grid_points:
        ax.axvline(x, color='black', alpha=0.1)
    for y in y_grid_points:
        ax.axhline(y, color='black', alpha=0.1)

    grid = np.array(np.meshgrid(x_grid_points, y_grid_points)).T.reshape(-1,2)

    vmin, vmax = c_range

    for i in range(len(grid)):
        v = potential_grid[i]
        if np.isnan(v):
            continue
        x, y = grid[i, :]
        # Normalize the potential values to 0-1 range
        v_nrm = (v - vmin) / (vmax - vmin)
        color = cmap(v_nrm)
        # alpha = grid_counts[i]
        xw = x_grid_points[1] - x_grid_points[0]
        yw = y_grid_points[1] - y_grid_points[0]

        ax.add_patch(plt.Rectangle((x, y), xw, yw,
                                   facecolor=color, alpha=1))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.suptitle(label, fontsize=18)
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm._A = []
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax

# %%
def plot_3d_height_map(potential_grid, grid_points, label):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_grid_points, y_grid_points = grid_points
    X, Y = np.meshgrid(x_grid_points, y_grid_points)

    # Reshape the potential grid to match the X, Y shape
    Z = potential_grid.reshape(len(x_grid_points), len(y_grid_points))

    # Plot the surface with a colormap
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Add a colorbar
    cbar = fig.colorbar(surf, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Potential')

    # Set labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Potential')
    ax.set_title(label)

    plt.tight_layout()
#%%
def plot_3d_scatter(x, y, z, label):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='viridis')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Potential')
    ax.set_title(label)

    plt.tight_layout()
#%%
def flatten_trajectory(trajectory):
    return trajectory.reshape(-1, trajectory.shape[2])
def to_np(tensor):
    return tensor.cpu().detach().numpy()

# %%
# Load data and models for wildtype and mutant
genotypes = ['wildtype', 'mutant']
data_paths = {
    'wildtype': '../../data/wildtype_net.h5ad',
    'mutant': '../../data/mutant_net.h5ad'
}
model_paths = {
    'wildtype': '../../output/20230607_165324/models/optimal_wildtype.torch',
    'mutant': '../../output/20230608_093734/models/optimal_mutant.torch'
}
trajectory_paths = {
    'wildtype': '../../output/20230607_165324/baseline_trajectories_wildtype.pickle',
    'mutant': '../../output/20230608_093734/baseline_trajectories_mutant.pickle'
}

models = {}
trajectories = {}

for genotype in genotypes:
    data, X = load_data(data_paths[genotype], device)
    trajectory = load_trajectories(trajectory_paths[genotype], device)
    trajectories[genotype] = trajectory
    models[genotype] = load_model(model_paths[genotype], input_dim=X.shape[1], hidden_dim=64, num_layers=3, device=device)
#%%
# Load PCA
pca = PCA()
pca.mean_ = data.uns['pca_mean']
pca.components_ = data.uns['PCs']

# %%
# Estimate potentials for wildtype and mutant
potentials = {}

num_nodes = X.shape[1]

flat_points = {}
projs = {}

for genotype in genotypes:
    flat_points[genotype] = flatten_trajectory(trajectories[genotype])

all_points = torch.concatenate([flat_points[genotype] for genotype in genotypes], axis=0)

for genotype in genotypes:
    potential, optimizer = initialize_potential(input_dim=num_nodes, hidden_dim=64, num_layers=3, device=device)
    potentials[genotype] = estimate_potential(models[genotype], potential, all_points, optimizer, epochs=1000, num_points=1000)

# %%
# Compute grid points
n_grids = 50

proj, grid_points = compute_grid_points(pca, all_points, n_grids)
#%%
potential_grids = {}
# Normalize the potential fields to 0-1 range
def norm(x):
    x_nna = x[~np.isnan(x)]
    x_min = x_nna.min()
    x_max = x_nna.max()
    return (x - x_min) / (x_max - x_min)
# Bin and aggregate the potential values
for genotype in genotypes:
    x_bin_indices = np.digitize(proj[:, 0], grid_points[0]) - 1
    y_bin_indices = np.digitize(proj[:, 1], grid_points[1]) - 1
    p_np = to_np(potentials[genotype](all_points))
    potential_grid, grid_count = bin_and_aggregate(x_bin_indices, y_bin_indices, p_np, n_grids)
    potential_grids[genotype] = potential_grid
    potential_grid_norm = norm(potential_grid)
    
    # Plot the grid
    plot_grid(potential_grid_norm, grid_points, f'{genotype.capitalize()} Potential Field', 
              cmap=plt.cm.get_cmap('viridis'), c_range=(0, 1))

#%%
# Compare the potential fields
potential_diff = norm(potential_grids['wildtype']) - norm(potential_grids['mutant'])
potential_diff = potential_diff.reshape(n_grids, n_grids)
potential_diff = potential_diff.flatten()
cmap = plt.cm.get_cmap('coolwarm')
c_range = max(abs(np.nanmin(potential_diff)), abs(np.nanmax(potential_diff)))
plot_grid(potential_diff, grid_points, 'Potential Difference WT-Mutant', 
          cmap=cmap, c_range=(-c_range, c_range))

# %%
# 3d height map
for genotype in genotypes:
    potential_grid_norm = norm(potential_grids[genotype])
    plot_3d_height_map(potential_grid_norm, grid_points, genotype)
# %%
# Get the density of points in each grid
# Project each trajectory to the PCA space
projs = {}
for genotype in genotypes:
    projs[genotype] = pca.transform(to_np(flat_points[genotype]))

# Bin and aggregate the potential values
grid_counts = {}
for genotype in genotypes:
    x_bin_indices = np.digitize(projs[genotype][:, 0], grid_points[0]) - 1
    y_bin_indices = np.digitize(projs[genotype][:, 1], grid_points[1]) - 1
    # Compute the number of points in each grid
    grid_count = np.zeros(n_grids**2)
    for i, (x, y) in enumerate(product(range(n_grids), range(n_grids))):
        idx = np.where((x_bin_indices == x) & (y_bin_indices == y))[0]
        grid_count[i] = len(idx)
    grid_counts[genotype] = grid_count
#%%   
# Find the min and max values of the grid counts across all genotypes
# so that we can use a common color scale
min_counts = np.inf
max_counts = -np.inf
for genotype in genotypes:
    min_counts = min(min_counts, grid_counts[genotype].min())
    max_counts = max(max_counts, grid_counts[genotype].max())

for genotype in genotypes:
    plot_grid(grid_counts[genotype], grid_points, f'{genotype.capitalize()} Density', 
              cmap=plt.cm.get_cmap('viridis'), c_range=(min_counts, max_counts))
# %%
# Plot the difference in density
density_diff = grid_counts['wildtype'] - grid_counts['mutant']
c_range = max(abs(np.nanmin(potential_diff)), abs(np.nanmax(potential_diff)))
fig, ax = plot_grid(potential_diff, grid_points, 
                    'WT-Mutant differences\nHeatmap: Potential Difference\nContours: Density Difference', 
                    cmap=plt.cm.get_cmap('coolwarm'), c_range=(-c_range, c_range))
# Create a sequence of colors for the contour lines from the PiYG colormap
# contour_colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, 10))
contours = ax.contour(grid_points[0], grid_points[1], 
                      density_diff.reshape(n_grids, n_grids).T, 
                      levels=10, colors='black')#contour_colors)
#ax.clabel(contours, inline=True, fontsize=8)
# Legend to show that dashed lines are negative and solid lines are positive
handles = [plt.Line2D([0], [0], color='black', linestyle='--', label='WT < Mutant'),
           plt.Line2D([0], [0], color='black', linestyle='-', label='WT > Mutant')]
ax.legend(handles=handles, title='Density Difference', fontsize=18)

# Show 
plt.show()
    # %%
for genotype in genotypes:
    random_points = np.random.choice(flat_points[genotype].shape[0], 10000)
    p_np = to_np(potentials[genotype](flat_points[genotype][random_points]))
    plot_3d_scatter(projs[genotype][random_points, 0], projs[genotype][random_points, 1], 
                    z=p_np, 
                    label=f'{genotype.capitalize()} Potential')
# %%
for genotype in genotypes:
    potential_grid_norm = norm(potential_grids[genotype])
    fig, ax = plot_grid(potential_grid_norm, grid_points, f'{genotype.capitalize()} Potential Field', 
                         cmap=plt.cm.get_cmap('viridis'), c_range=(0, 1))
    # Add a contour plot of trajectory density at the critical point (t=41)
    ax.contour(grid_points[0], grid_points[1], 
               grid_counts[genotype].reshape(n_grids, n_grids).T,
                levels=10, colors='white')
# %%
