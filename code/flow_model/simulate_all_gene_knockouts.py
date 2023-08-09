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
from textwrap import fill

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# %%
# wt = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut = sc.read_h5ad(f'../../data/mutant_net.h5ad')
# %%
# Load the models
# tmstp = '20230607_165324'  
# genotype = 'wildtype'
mut_tmstp = '20230608_093734'

genotype = 'mutant'
outdir = f'../../output/{mut_tmstp}'
ko_dir = f'{outdir}/knockout_simulations'
pltdir = f'{outdir}/knockout_simulations/figures'

#%%
# Check if ko_dir exists. This is where we will save the knockout simulations and figures
import os
if not os.path.exists(ko_dir):
    os.makedirs(ko_dir)
    os.makedirs(pltdir)

state_dict = torch.load(f'{outdir}/models/optimal_{genotype}.torch')

# %%
X = torch.tensor(mut.X.toarray()).float()
cell_types = {c:i for i,c in enumerate(set(mut.obs['cell_type']))}
proj = np.array(mut.obsm['X_pca'])
pca = PCA()
# Set the PC mean and components
pca.mean_ = mut.uns['pca_mean']
pca.components_ = mut.uns['PCs']
proj = np.array(pca.transform(X))[:,0:2]
T = mut.obsm['transition_matrix']

V = util.velocity_vectors(T, X)
V_emb = util.embed_velocity(X, V, lambda x: np.array(pca.transform(x)[:,0:2]))

# %%
torch.set_num_threads(24)
start_idxs = mut.uns['initial_points_via']
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
Xnp = util.tonp(X)
simulator = Simulator(model, X, device=device)

#%%
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}


#%%
n_repeats = 10
start_idxs = mut.uns['initial_points_via']
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
len_trajectory = 98
n_steps = len_trajectory*4
t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{outdir}/baseline_trajectories_mutant.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{outdir}/baseline_nearest_cell_idxs_mutant.pickle', 'rb'))
baseline_velo,_ = plotting.compute_velo(model=model, X=X, numpy=True)
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
all_genes = set(node_to_idx.keys())

#%%
for i,ko_gene in enumerate(all_genes):
    print(f'Knockout Gene {protein_id_name[ko_gene]:10s}: ({i+1}/{len(all_genes)})', flush=True)
    target_idx = node_to_idx[ko_gene]
    perturbation = (target_idx, 
                    torch.zeros(1, device=device))

    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats, t_span, 
                                                                    perturbation=perturbation, 
                                                                    boundary=False,
                                                                    show_progress=False)
    ko_gene_name = protein_id_name[ko_gene]
    perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    # Each trajectory consumes 376 MB of storage, for a total of ~100GB for all genes
    # So we don't store them, might have to resimulate later to get 
    # actual trajectories of interesting genes
    # Save the trajectories and nearest cell indices
    # with open(f'{ko_dir}/{ko_gene_name}_knockout_trajectories_mutant.pickle', 'wb') as f:
    #     pickle.dump(perturb_trajectories_np, f)
    # with open(f'{ko_dir}/{ko_gene_name}_knockout_nearest_idxs_mutant.pickle', 'wb') as f:
    #     pickle.dump(perturb_idxs_np, f) 
    
    # Aggregate the individual cell trajectories by mean
    mean_trajectories = perturb_trajectories.mean(dim=1)
    # Save the mean trajectories
    with open(f'{ko_dir}/{ko_gene_name}_knockout_mean_trajectories_mutant.pickle', 'wb') as f:
        pickle.dump(mean_trajectories, f)

    plotting.distribution(perturb_trajectories_np[:,::10], pca,
                          label=f'{ko_gene_name} Knockout',
                          baseline=Xnp)
    plt.savefig(f'{pltdir}/{ko_gene_name}_knockout_distribution.png')
    plt.close()
    velo,_ = plotting.compute_velo(model=model, X=X, perturbation=perturbation, numpy=True)
    plotting.arrow_grid(velos=[velo, baseline_velo], 
                        data=[mut, mut], 
                        pca=pca, 
                        labels=[f'{ko_gene_name} Knockout', 'Mutant baseline'])
    plt.savefig(f'{pltdir}/{ko_gene_name}_knockout_arrow_grid.png')
    plt.close()
    plotting.sample_trajectories(perturb_trajectories_np, Xnp, pca, f'{ko_gene_name} Knockout')
    plt.savefig(f'{pltdir}/{ko_gene_name}_knockout_trajectories.png')
    plt.close()
    trajectories = plotting.compare_cell_type_trajectories([perturb_idxs_np, baseline_idxs],
                                                           [mut, mut], 
                                                           cell_types,
                                                           [ko_gene_name, 'baseline'])
    mut_cell_type_traj, perturb_cell_type_traj = trajectories
    # Save the trajectories
    with open(f'{ko_dir}/{ko_gene_name}_knockout_cell_type_trajectories_mutant.pickle', 'wb') as f:
        pickle.dump(trajectories, f)
    plt.savefig(f'{pltdir}/{ko_gene_name}_knockout_cell_type_trajectories.png')
    plt.close()
    mut_ct = mut.obs['cell_type'].value_counts()/mut.shape[0]
    mut_ct = mut_ct[sorted(cell_types)]
    mut_total_cell_types = mut_cell_type_traj.sum(axis=1)/mut_cell_type_traj.sum()
    perturb_total_cell_types = perturb_cell_type_traj.sum(axis=1)/perturb_cell_type_traj.sum()
    # Plot side by side bar charts of the cell type proportions
    n_cells = (len(start_idxs) * n_repeats * n_steps)
    perturb_cell_type_traj = trajectories[0]
    baseline_cell_type_traj = trajectories[1]
    perturb_cell_proportions = perturb_cell_type_traj.sum(axis=1) / n_cells
    baseline_cell_proportions = baseline_cell_type_traj.sum(axis=1) / n_cells
    # Save the cell type proportions
    plotting.cell_type_proportions(proportions=(perturb_cell_proportions, 
                                                baseline_cell_proportions), 
                                   cell_types=cell_types, 
                                   labels=[f'{ko_gene_name} Knockout', 'Mutant baseline'])
    plt.savefig(f'{pltdir}/{ko_gene_name}_knockout_cell_type_proportions.png',
                bbox_inches='tight');
    plt.close();
    with open(f'{ko_dir}/{ko_gene_name}_knockout_cell_type_proportions_mutant.pickle', 'wb') as f:
        pickle.dump((perturb_cell_proportions, baseline_cell_proportions), f)

# %%
