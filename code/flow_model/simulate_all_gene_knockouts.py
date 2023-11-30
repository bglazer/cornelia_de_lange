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
from joblib import Parallel, delayed
import os
import matplotlib

#%%
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# %%
# Load the models
tmstp = '20230607_165324'  
genotype = 'wildtype'
# genotype = 'mutant'
# tmstp = '20230608_093734'
data = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')

outdir = f'../../output/{tmstp}'
ko_dir = f'{outdir}/knockout_simulations'
pltdir = f'{outdir}/knockout_simulations/figures'
datadir = f'{outdir}/knockout_simulations/data'

#%%
# Check if ko_dir exists. This is where we will save the knockout simulations and figures
import os
if not os.path.exists(ko_dir):
    os.makedirs(ko_dir)
    os.makedirs(pltdir)
    os.makedirs(datadir)

state_dict = torch.load(f'{outdir}/models/optimal_{genotype}.torch')

# %%
X = torch.tensor(data.X.toarray()).float()
Xnp = util.tonp(X)
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

#%%
cell_types = {c:i for i,c in enumerate(sorted(set(data.obs['cell_type'])))}

#%%
simulators = []
Xs = []
for i in range(4):
    model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers,
                         predict_var=True)
    model.load_state_dict(state_dict)
    device=f'cuda:{i}'
    model = model.to(device)
    Xs.append(X.to(device))
    simulators.append(Simulator(model, Xs[i], device=device, boundary=False, show_progress=False))

#%%
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}


#%%
n_repeats = 10
start_idxs = data.uns['initial_points_via']
repeats = torch.tensor(start_idxs.repeat(n_repeats)).to(device)
len_trajectory = 98
n_steps = len_trajectory*4
t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the baseline trajectories
baseline_trajectories = pickle.load(open(f'{outdir}/baseline_trajectories_{genotype}.pickle', 'rb'))
baseline_trajectories_np = baseline_trajectories
baseline_idxs = pickle.load(open(f'{outdir}/baseline_nearest_cell_idxs_{genotype}.pickle', 'rb'))
baseline_velo,_ = plotting.compute_velo(model=simulators[0].model, X=Xs[0], numpy=True)
#%%
plotting.time_distribution(baseline_trajectories_np[:,:], pca,
                        label=f'Baseline {genotype} - no knockout',
                        baseline=Xnp)
# plt.show()
plt.savefig(f'{pltdir}/baseline_{genotype}_time_distribution.png', bbox_inches='tight')
plt.close()
#%%
plotting.cell_type_distribution(baseline_trajectories_np[:,:], 
                                baseline_idxs[:,:],
                                data,
                                cell_types,
                                pca,
                                label=f'Baseline {genotype} - no knockout',
                                baseline=Xnp)
# plt.show()
plt.savefig(f'{pltdir}/baseline_{genotype}_cell_type_distribution.png', bbox_inches='tight')
plt.close()
#%%
plotting.cell_type_distribution(np.expand_dims(Xnp, axis=1), 
                                np.expand_dims(np.arange(Xnp.shape[0]), axis=1),
                                data,
                                cell_types,
                                pca,
                                label=f'Cell Type Distribution - Data',
                                baseline=None,
                                s=20)
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
all_genes = set(node_to_idx.keys())

num_gpus = 4

#%%
def simulate_knockout(ko_gene, i, repeats):
    print(f'Knockout Gene {protein_id_name[ko_gene]:10s}: ({i+1}/{len(all_genes)})', flush=True)
    
    gpu = i % num_gpus
    device = f'cuda:{gpu}'

    target_idx = node_to_idx[ko_gene]
    perturbation = (target_idx, 
                    torch.zeros(1, device=device))

    repeats = repeats.to(device)
    simulator = simulators[gpu]
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats, t_span, 
                                                                    node_perturbation=perturbation, 
                                                                    )
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
    with open(f'{datadir}/{ko_gene_name}_{genotype}_knockout_mean_trajectories.pickle', 'wb') as f:
        pickle.dump(mean_trajectories, f)

    plotting.time_distribution(perturb_trajectories_np[:,:], pca,
                               label=f'{ko_gene_name} Knockout',
                               baseline=Xnp)
    plt.savefig(f'{pltdir}/{ko_gene_name}_{genotype}_knockout_time_distribution.png',
                bbox_inches='tight')
    plt.close()
    plotting.cell_type_distribution(perturb_trajectories_np[:,:], 
                                    perturb_idxs_np[:,:],
                                    data,
                                    cell_types,
                                    pca,
                                    label=f'{ko_gene_name} Knockout',
                                    baseline=Xnp)
    plt.savefig(f'{pltdir}/{ko_gene_name}_{genotype}_knockout_cell_type_distribution.png',
                bbox_inches='tight')
    plt.close()
    velo,_ = plotting.compute_velo(model=simulator.model, X=Xs[gpu], perturbation=perturbation, numpy=True)
    plotting.arrow_grid(velos=[velo, baseline_velo], 
                        data=[data, data], 
                        pca=pca, 
                        labels=[f'{ko_gene_name} Knockout', f'{genotype.capitalize()} baseline'])
    plt.savefig(f'{pltdir}/{ko_gene_name}_{genotype}_knockout_arrow_grid.png')
    plt.close()
    plotting.sample_trajectories(perturb_trajectories_np, Xnp, pca, f'{ko_gene_name} Knockout')
    plt.savefig(f'{pltdir}/{ko_gene_name}_{genotype}_knockout_trajectories.png')
    plt.close()
    trajectories = plotting.compare_cell_type_trajectories([perturb_idxs_np, baseline_idxs],
                                                           [data, data], 
                                                           cell_types,
                                                           [ko_gene_name, 'baseline'])
    # Save the trajectories
    with open(f'{datadir}/{ko_gene_name}_{genotype}_knockout_cell_type_trajectories.pickle', 'wb') as f:
        pickle.dump(trajectories, f)
    plt.savefig(f'{pltdir}/{ko_gene_name}_{genotype}_knockout_cell_type_trajectories.png')
    plt.close()
    # Plot side by side bar charts of the cell type proportions
    perturb_cell_proportions, perturb_cell_errors = plotting.calculate_cell_type_proportion(perturb_idxs_np, data, cell_types, n_repeats, error=True)
    baseline_cell_proportions, baseline_cell_errors = plotting.calculate_cell_type_proportion(baseline_idxs, data, cell_types, n_repeats, error=True)
    # Save the cell type proportions
    plotting.cell_type_proportions(proportions=(perturb_cell_proportions, 
                                                baseline_cell_proportions), 
                                   proportion_errors=(perturb_cell_errors,
                                           baseline_cell_errors),
                                   cell_types=list(cell_types), 
                                   labels=[f'{ko_gene_name} Knockout', f'{genotype.capitalize()} baseline'])
    plt.savefig(f'{pltdir}/{ko_gene_name}_{genotype}_knockout_cell_type_proportions.png',
                bbox_inches='tight');
    plt.close();
    with open(f'{datadir}/{ko_gene_name}_{genotype}_knockout_cell_type_proportions.pickle', 'wb') as f:
        pickle.dump((perturb_cell_proportions, baseline_cell_proportions), f)

# %%
_ = Parallel(n_jobs=8)(delayed(simulate_knockout)(ko_gene, i, repeats) for i,ko_gene in enumerate(all_genes))

# %%
