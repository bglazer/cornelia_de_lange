#%%
import glob
import json
import numpy as np
import torch
import scanpy as sc
import pickle
from flow_model import GroupL1FlowModel, LinearFlowModel, ConnectedFlowModel
import torch
from simulator import Simulator
import os
import sys
sys.path.append('..')
import util
import plotting

#%%
mut_linear_tmstp = '20231212_214816'
mut_connected_tmstp = '20231212_164633'
mut_nn_tmstp = '20230608_093734'

wt_linear_tmstp = '20231213_103134'
wt_connected_tmstp = '20231213_101401'
wt_nn_tmstp = '20230607_165324'

mut_linear_outdir = f'../../output/{mut_linear_tmstp}'
mut_connected_outdir = f'../../output/{mut_connected_tmstp}'
mut_nn_outdir = f'../../output/{mut_nn_tmstp}'
wt_linear_outdir = f'../../output/{wt_linear_tmstp}'
wt_connected_outdir = f'../../output/{wt_connected_tmstp}'
wt_nn_outdir = f'../../output/{wt_nn_tmstp}'

#%%
mut_linear_logdir = f'{mut_linear_outdir}/logs'
mut_linear_logs = [open(f).readlines() for f in glob.glob(f'{mut_linear_logdir}/*.log')]
mut_linear_final_loss = [json.loads(l[-1])['val_mean_loss'] for l in mut_linear_logs]
wt_linear_logdir = f'{wt_linear_outdir}/logs'
wt_linear_logs = [open(f).readlines() for f in glob.glob(f'{wt_linear_logdir}/*.log')]
wt_linear_final_loss = [json.loads(l[-1])['val_mean_loss'] for l in wt_linear_logs]

# %%
mut_linear_final_loss = np.array(mut_linear_final_loss)
mut_linear_mean_loss = np.mean(mut_linear_final_loss)
wt_linear_final_loss = np.array(wt_linear_final_loss)
wt_linear_mean_loss = np.mean(wt_linear_final_loss)

# %%
mut_nn_logdir = f'{mut_nn_outdir}/logs'
mut_nn_logs = [open(f).readlines() for f in glob.glob(f'{mut_nn_logdir}/*.log')]
mut_nn_final_loss = [json.loads(l[-1])['val_mean_loss'] for l in mut_nn_logs]
mut_nn_mean_loss = np.mean(mut_nn_final_loss)

wt_nn_logdir = f'{wt_nn_outdir}/logs'
wt_nn_logs = [open(f).readlines() for f in glob.glob(f'{wt_nn_logdir}/*.log')]
wt_nn_final_loss = [json.loads(l[-1])['val_mean_loss'] for l in wt_nn_logs]
wt_nn_mean_loss = np.mean(wt_nn_final_loss)
# %%
mut_connected_logdir = f'{mut_connected_outdir}/logs'
mut_connected_log = open(f'{mut_connected_logdir}/connected_model_mutant.log').readlines()
mut_connected_loss = json.loads(mut_connected_log[-1])['mean_validation_loss']

wt_connected_logdir = f'{wt_connected_outdir}/logs'
wt_connected_log = open(f'{wt_connected_logdir}/connected_model_wildtype.log').readlines()
wt_connected_loss = json.loads(wt_connected_log[-1])['mean_validation_loss']
# %%
print('Wildtype')
print('Linear model loss:   ', wt_linear_mean_loss)
print('Connected model loss:', wt_connected_loss)
print('L1-NN model loss:    ', wt_nn_mean_loss)
print('Linear Penalty:  ', (wt_connected_loss - wt_linear_mean_loss)/wt_connected_loss)
print('L1-NN model loss:', (wt_connected_loss - wt_nn_mean_loss)/wt_connected_loss)
print('-'*80)
print('Mutant')
print('Linear model loss:   ', mut_linear_mean_loss)
print('Connected model loss:', mut_connected_loss)
print('L1-NN model loss:    ', mut_nn_mean_loss)
print('Linear Penalty:  ', (mut_connected_loss - mut_linear_mean_loss)/mut_connected_loss)
print('L1-NN model loss:', (mut_connected_loss - mut_nn_mean_loss)/mut_connected_loss)

#%%
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

#%%
mut_data = sc.read_h5ad(f'../../data/mutant_net.h5ad')
mut_X = torch.tensor(mut_data.X.toarray()).float()
wt_data = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
wt_X = torch.tensor(wt_data.X.toarray()).float()
torch.set_num_threads(40)
#%%
cell_types = {c:i for i,c in enumerate(sorted(set(mut_data.obs['cell_type'])))}

n_repeats = 10
mut_start_idxs = mut_data.uns['initial_points_nmp']
wt_start_idxs = wt_data.uns['initial_points_nmp']
mut_repeats = torch.tensor(mut_start_idxs.repeat(n_repeats))
wt_repeats = torch.tensor(wt_start_idxs.repeat(n_repeats))
len_trajectory = mut_data.uns['best_trajectory_length']
n_steps = mut_data.uns['best_step_size']*len_trajectory

t_span = torch.linspace(0, len_trajectory, n_steps)

#%%
# Load the baseline trajectories
mut_data_cell_proportions = mut_data.obs['cell_type'].value_counts()/mut_data.shape[0]
sorted_cell_types = sorted(cell_types.keys())
mut_data_cell_proportions = mut_data_cell_proportions[sorted_cell_types].to_numpy()

wt_data_cell_proportions = wt_data.obs['cell_type'].value_counts()/wt_data.shape[0]
sorted_cell_types = sorted(cell_types.keys())
wt_data_cell_proportions = wt_data_cell_proportions[sorted_cell_types].to_numpy()

#%%
def simulate(model, X, data, repeats, label, datadir):
    device = f'cuda:0'
    simulator = Simulator(model, X.to(device), device=device, boundary=False, show_progress=False)
    repeats_gpu = repeats.to(device)
    perturb_trajectories, perturb_nearest_idxs = simulator.simulate(repeats_gpu, t_span)
    # perturb_trajectories_np = util.tonp(perturb_trajectories)
    perturb_idxs_np = util.tonp(perturb_nearest_idxs)
    # Delete full trajectories so that we can free the GPU memory
    del perturb_trajectories
    del perturb_nearest_idxs
    # Tell the garbage collector to free the GPU memory
    torch.cuda.empty_cache()

    perturb_cell_proportions, perturb_cell_errors = plotting.calculate_cell_type_proportion(perturb_idxs_np, data, cell_types, n_repeats, error=True)
    
    proportion_dict = {'perturb_proportions':perturb_cell_proportions, 
                       'perturb_errors': perturb_cell_errors}
    with open(f'{datadir}/{label}_cell_type_proportions.pickle', 'wb') as f:
        pickle.dump(proportion_dict, f)
    
    return perturb_cell_proportions, perturb_cell_errors

#%%
hidden_dim = 64
num_layers = 3
num_nodes = mut_X.shape[1]

mut_connected_model_state = torch.load(f'{mut_connected_outdir}/models/connected_model_mutant.torch')
mut_l1nn_model_state = torch.load(f'{mut_nn_outdir}/models/optimal_mutant.torch')
mut_linear_model_state = pickle.load(open(f'{mut_linear_outdir}/models/linear_flow_model_mutant.pickle','rb'))

wt_connected_model_state = torch.load(f'{wt_connected_outdir}/models/connected_model_wildtype.torch')
wt_l1nn_model_state = torch.load(f'{wt_nn_outdir}/models/optimal_wildtype.torch')
wt_linear_model_state = pickle.load(open(f'{wt_linear_outdir}/models/linear_flow_model_wildtype.pickle','rb'))

mut_l1nn_model = GroupL1FlowModel(input_dim=num_nodes, 
                              hidden_dim=hidden_dim, 
                              num_layers=num_layers,
                              predict_var=True)
mut_l1nn_model.load_state_dict(mut_l1nn_model_state)
mut_l1nn_model = mut_l1nn_model.to('cuda:0')

wt_l1nn_model = GroupL1FlowModel(input_dim=num_nodes, 
                              hidden_dim=hidden_dim, 
                              num_layers=num_layers,
                              predict_var=True)
wt_l1nn_model.load_state_dict(wt_l1nn_model_state)
wt_l1nn_model = wt_l1nn_model.to('cuda:0')
#%%
mut_linear_model = LinearFlowModel(input_dim=num_nodes, 
                                   predict_var=True)
for i,model in enumerate(mut_linear_model_state):
    mut_linear_model.models[i] = model
mut_linear_model = mut_linear_model.to('cuda:0')

wt_linear_model = LinearFlowModel(input_dim=num_nodes, 
                                  predict_var=True)
for i,model in enumerate(wt_linear_model_state):
    wt_linear_model.models[i] = model
wt_linear_model = wt_linear_model.to('cuda:0')
#%%
num_nodes = mut_X.shape[1]
hidden_dim = num_nodes*2
num_layers = 3

mut_connected_model = ConnectedFlowModel(input_dim=num_nodes, 
                                         output_dim=num_nodes, 
                                         hidden_dim=hidden_dim, 
                                         num_layers=num_layers, 
                                         predict_var=True)
mut_connected_model.load_state_dict(mut_connected_model_state)
mut_connected_model = mut_connected_model.to('cuda:0')

wt_connected_model = ConnectedFlowModel(input_dim=num_nodes, 
                                        output_dim=num_nodes, 
                                        hidden_dim=hidden_dim, 
                                        num_layers=num_layers, 
                                        predict_var=True)
wt_connected_model.load_state_dict(wt_connected_model_state)
wt_connected_model = wt_connected_model.to('cuda:0')
#%%
# Simulate the connected model trajectories
#(model, X, data, label, datadir)
mut_connected_cell_proportions, connected_cell_errors = simulate(mut_connected_model, mut_X, mut_data, mut_repeats, 'connected_mutant', mut_connected_outdir)
wt_connected_cell_proportions, connected_cell_errors = simulate(wt_connected_model, wt_X, wt_data, wt_repeats, 'connected_wildtype', wt_connected_outdir)
#%%
# Simulate the L1-NN model trajectories
mut_l1nn_cell_proportions, l1nn_cell_errors = simulate(mut_l1nn_model, mut_X, mut_data, mut_repeats, 'l1nn_mutant', mut_nn_outdir)
wt_l1nn_cell_proportions, l1nn_cell_errors = simulate(wt_l1nn_model, wt_X, wt_data, wt_repeats, 'l1nn_wildtype', wt_nn_outdir)
#%%
# Simulate the linear model trajectories
mut_linear_cell_proportions, linear_cell_errors = simulate(mut_linear_model, mut_X, mut_data, mut_repeats, 'linear_mutant', mut_linear_outdir)
wt_linear_cell_proportions, linear_cell_errors = simulate(wt_linear_model, wt_X, wt_data, wt_repeats, 'linear_wildtype', wt_linear_outdir)


#%%
def calculate_errors(perturb_cell_proportions, data_cell_proportions):
    ds = np.abs(perturb_cell_proportions - data_cell_proportions)
    d = ds.sum()

    # misses = ds > (perturb_cell_errors*2 + baseline_cell_errors*2)
    # miss_cell_types = [idx_to_cell_type[i] for i in np.where(misses)[0]]
    print(f'{d}')#, {",".join(miss_cell_types)}', flush=True)
    for i,c in enumerate(sorted_cell_types):
        print(f'{c:5s}: {perturb_cell_proportions[i]:.5f} {data_cell_proportions[i]:.5f}')

print('Connected model errors:')
print('Wildtype:')
calculate_errors(wt_connected_cell_proportions, wt_data_cell_proportions)
print('Mutant:')
calculate_errors(mut_connected_cell_proportions, mut_data_cell_proportions)
print('-'*80)

print('L1-NN model errors:')
print('Wildtype:')
calculate_errors(wt_l1nn_cell_proportions, wt_data_cell_proportions)
print('Mutant:')
calculate_errors(mut_l1nn_cell_proportions, mut_data_cell_proportions)
print('-'*80)

print('Linear model errors:')
print('Wildtype:')
calculate_errors(wt_linear_cell_proportions, wt_data_cell_proportions)
print('Mutant:')
calculate_errors(mut_linear_cell_proportions, mut_data_cell_proportions)
print('-'*80)

# %%
# Compare the simulation wt vs mut proportions to data wt vs mut proportions. 
def wt_vs_mut_proportions(wt_sim_proportions, mut_sim_proportions, wt_data_proportions, mut_data_proportions):
    sim_wt_gt_mut = np.sign(wt_sim_proportions - mut_sim_proportions)
    data_wt_gt_mut = np.sign(wt_data_proportions - mut_data_proportions)
    correct = sim_wt_gt_mut == data_wt_gt_mut
    print(f'Correct: {correct.sum()}')
    for i,c in enumerate(sorted_cell_types):
        print(f'{c:5s}: {sim_wt_gt_mut[i] == data_wt_gt_mut[i]}')

print('Connected model:')
wt_vs_mut_proportions(wt_connected_cell_proportions, mut_connected_cell_proportions, wt_data_cell_proportions, mut_data_cell_proportions)
print('-'*80)
print('L1-NN model:')
wt_vs_mut_proportions(wt_l1nn_cell_proportions, mut_l1nn_cell_proportions, wt_data_cell_proportions, mut_data_cell_proportions)
print('-'*80)
print('Linear model:')
wt_vs_mut_proportions(wt_linear_cell_proportions, mut_linear_cell_proportions, wt_data_cell_proportions, mut_data_cell_proportions)
print('-'*80)
# %%
print(np.abs(wt_linear_cell_proportions - mut_linear_cell_proportions).sum())
print(np.abs(wt_l1nn_cell_proportions - mut_l1nn_cell_proportions).sum())

# %%
