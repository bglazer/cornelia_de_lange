# %%
from glob import glob
import matplotlib.pyplot as plt
import pickle        
import json
import numpy as np
import random
#%%
tmstp = '20230607_150648'

# %%
logfiles = glob(f'../../output/{tmstp}/logs/*.log')
# %%
logs = {}
for file in logfiles:
    model_idx = int(file.split('/')[-1].split('_')[3])
    logs[model_idx] = []
    with open(file, 'r') as f:
        for line in f:
            log = json.loads(line)
            logs[model_idx].append(log)

#%%
genotype = 'wildtype'
params = pickle.load(open(f'../../output/{tmstp}/params.pickle','rb'))
# l1_alphas = params['l1_alphas']

# %%
# Plot the validation loss for each model
validation_loss = {}
for model_idx in logs:
    hidden_dim = params[model_idx]['hidden_dim']
    l1_alpha = params[model_idx]['l1_alpha']
    if hidden_dim not in validation_loss:
        validation_loss[hidden_dim] = {}
    if l1_alpha not in validation_loss[hidden_dim]:
        validation_loss[hidden_dim][l1_alpha] = []
    if 'val_mean_loss' in logs[model_idx][-1]:
        validation_loss[hidden_dim][l1_alpha].append(logs[model_idx][-1]['val_mean_loss'])
#%%
fig, axs = plt.subplots(1, 1, figsize=(5,5))
n_l1_alphas = len(validation_loss[16])
for i, hidden_dim in enumerate(validation_loss):
    axs.violinplot(validation_loss[hidden_dim].values(), 
                      positions=range(n_l1_alphas),
                      points=200,
                      showextrema=False,
                      showmedians=True)
    # Annotate each violin plot with the median validation loss
    for j, l1_alpha in enumerate(validation_loss[hidden_dim]):
        axs.annotate(xy=(j-.5, np.median(validation_loss[hidden_dim][l1_alpha])+1e-4), 
                        text=f'{np.median(validation_loss[hidden_dim][l1_alpha]):.2e}', 
                        fontsize=6)
    axs.set_xticks(range(len(validation_loss[hidden_dim])), 
                   [f'{x:.1f}' for x in validation_loss[hidden_dim].keys()])
    axs.set_ylim([0, .001])
    # Label each subplot
    axs.set_title(f'Hidden Dimension: {hidden_dim}')
plt.tight_layout()

#%%
# Plot validation loss versus active nodes
active_nodes = {}
for model_idx in logs:
    last_log = logs[model_idx][-1]
    l1_alpha = last_log['l1_alpha']
    if l1_alpha < 100:
        if l1_alpha not in active_nodes:
            active_nodes[l1_alpha] = []
        active_nodes[l1_alpha].append((last_log['active_inputs'], last_log['val_mean_loss']))
#%%
fig, axs = plt.subplots(len(active_nodes), 2, figsize=(10, 20))
combined = np.array([list(x) for x in active_nodes.values()]).reshape(-1, 2)
max_active_nodes = combined[:,0].max()
max_validation_loss = combined[:,1].max()
bins = np.histogram(combined[:,0], bins=50)[1]

for i, l1_alpha in enumerate(active_nodes):
    xy = np.array(active_nodes[l1_alpha])
    axs[i][0].scatter(xy[:,0], np.log10(xy[:,1]), s=1, alpha=.3)
    # axs[i][0].set_ylim([0, .01])
    axs[i][0].set_xlim([0, max_active_nodes])
    axs[i][0].set_title(f'l1_alpha: {l1_alpha:.1f}')
    counts, bins,_ = axs[i][1].hist(xy[:,0], bins=bins)
    # axs[i][1].set_ylim([0, xy[:,0].shape[0]/10])
    axs[i][1].set_xlim([-1, max_active_nodes])
    # Annotate the number of zeros in xy[:,0]
    ymax = counts.max() * 1.1
    axs[i][1].annotate(xy=(5, ymax*.8),
                       text=f'{(xy[:,0] == 0).sum()/xy[:,0].shape[0] *100:.0f}% zeros',
                       fontsize=7)
    axs[i][1].set_yticks(range(0, int(ymax), 10))
plt.tight_layout()

#%% 
# Plot the validation loss distributions with 64, 32, and 16 nodes
fig, ax = plt.subplots(len(validation_loss), 1, figsize=(5, 5))
# Get the bins from the 64 node distribution
combined = np.log10(np.array([list(x.values()) for x in validation_loss.values()]).flatten())
bins = np.histogram(combined, bins=50)[1]
for i, hidden_dim in enumerate(validation_loss):
    d = np.log10(np.array(list(validation_loss[hidden_dim].values())).flatten())
    ax.hist(d, bins=bins, alpha=.3)
    ax.set_title(f'Hidden Dimension: {hidden_dim}')
    ax.axvline(np.median(d), color='red')
    ax.annotate(xy=(np.median(d), 100), text=f'{np.median(d):.3f}', fontsize=7)
plt.tight_layout()