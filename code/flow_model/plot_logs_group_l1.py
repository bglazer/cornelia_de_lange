# %%
from glob import glob
import matplotlib.pyplot as plt
import pickle        
import json
import numpy as np
import random
#%%
tmstp = '20230525_161335'

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
# %%
validation_loss = {}
active_nodes = {}
train_loss = {}
for model_idx in logs:
    validation_loss[model_idx] = []
    active_nodes[model_idx] = []
    train_loss[model_idx] = []
    for log in logs[model_idx]:
        if 'iteration' in log:
            validation_loss[model_idx].append(log['validation_loss'])
            active_nodes[model_idx].append(log['active_inputs'])
        else:
            train_loss[model_idx].append(log['train_loss'])
#%%
genotype = 'wildtype'
params = pickle.load(open(f'../../output/{tmstp}/params/group_l1_flow_model_{genotype}.pickle','rb'))
l1_alphas = params['l1_alphas']

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
xaxis = np.array(active_nodes[0]).flatten()[::-1]
for model_idx in validation_loss:
    l = np.array(validation_loss[model_idx])
    scaled_l = l / np.max(l)
    axs.plot(range(len(xaxis)), scaled_l, label=model_idx, c='grey', alpha=0.3)
axs.set_xlabel('Active Nodes')
axs.set_ylabel('Best Validation Loss')
tick_interval = 1
axs.set_xticks(range(len(xaxis))[::tick_interval]);
# Reverse the x-axis so that the number of active nodes is decreasing
# Also, rotate the labels so that they are readable
axs.set_xticklabels(xaxis[::-tick_interval], rotation=45);
#%%
# Find the pruning iteration when the validation loss is within 10% of the worst validation loss
collapse_iterations = []
for model_idx in validation_loss:
    l = np.array(validation_loss[model_idx])
    scaled_l = l / np.max(l)
    idx = np.where(scaled_l > 0.9)[0][0]
    collapse_iterations.append(idx)

collapse_iterations = np.array(collapse_iterations)
counter = np.zeros(len(xaxis))
for idx in collapse_iterations:
    counter[idx] += 1
fig, axs = plt.subplots(1, 1, figsize=(7.5, 5))
tick_interval = 5
axs.set_xticks(xaxis[::-tick_interval]);
axs.bar(x=xaxis[::-1], height=(counter/counter.sum())[::1], 
        width=1, align="edge")
axs.set_ylim([0, 1])
axs.set_xticklabels(xaxis[::-tick_interval], rotation=45);
axs.set_xlabel('Active Nodes')
# Set the x tick labels
tick_interval = 3

axs.set_ylabel('Percentage of models')
axs.set_title('Iteration when validation loss is within 10% of the worst validation loss');

# %%
# Plot the difference between the validation loss with zero active nodes and the best 
# validation loss 
zero_active_change = []
for model_idx in validation_loss:
    ratio = validation_loss[model_idx][-1] / np.min(validation_loss[model_idx])
    zero_active_change.append(ratio)
zero_active_change = np.array(zero_active_change)

#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(zero_active_change, bins=50);
axs[0].set_title('Histogram of ratios') 
axs[0].set_xlabel('Validation Loss (Zero active)/ Best Validation Loss')
axs[0].set_ylabel('Count')
axs[0].axvline(1, color='red')
print(np.mean(zero_active_change))
print((zero_active_change > 1).sum() / len(zero_active_change))
axs[1].plot(sorted(zero_active_change))
axs[1].set_ylabel('Validation Loss (Zero active)/ Best Validation Loss')
axs[1].set_title('Sorted ratios');
axs[1].axvline(1, color='red')

#%%
# Plot the difference between the validation loss with zero active nodes and the best 
# validation loss 
zero_active_change = []
for model_idx in validation_loss:
    ratio = validation_loss[model_idx][-1] / validation_loss[model_idx][-2]
    zero_active_change.append(ratio)
zero_active_change = np.array(zero_active_change)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(zero_active_change, bins=50);
axs[0].set_title('Histogram of ratios') 
axs[0].set_xlabel('Validation Loss (Zero active)/ Last Validation Loss')
axs[0].set_ylabel('Count')
axs[0].axvline(1, color='red')
print("Percentage models with higher validation loss with zero active inputs:\n", 
      (zero_active_change > 1).sum() / len(zero_active_change)*100, '%')
axs[1].plot(sorted(zero_active_change))
axs[1].set_ylabel('Validation Loss (Zero active)/ Last Validation Loss')
axs[1].set_title('Sorted ratios');
axs[1].axhline(1, color='red', linewidth=.5);

# %%
# Plot a sample of training loss curves
n_samples = 16
# Arrange 16 plots in a 4x4 grid
fig, axs = plt.subplots(4, 8, figsize=(20, 20))
random_model_idxs = list(random.sample(train_loss.keys(), k=n_samples))
for i,model_idx in enumerate(random_model_idxs):
    i = i*2
    axs[i // 8, i % 8].plot(train_loss[model_idx])
    losses = validation_loss[model_idx]
    max_loss = np.max(losses)
    nrm_losses = losses / max_loss
    n = len(losses)
    d = (nrm_losses)**2 + (np.arange(n,0,-1)/n)**2
    knee = np.argmin(d)
    
    j = i+1
    axs[i // 8, j % 8].plot(validation_loss[model_idx], c='orange')
    axs[i // 8, j % 8].scatter(knee, validation_loss[model_idx][knee], c='blue', s=50)
# Set the title for the entire plot
plt.suptitle('Training and Validation Loss', fontsize=30)
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# %%
knees = []
best_active_nodes = []
best_alphas = []
for model_idx in validation_loss:
    losses = validation_loss[model_idx]
    max_loss = np.max(losses)
    nrm_losses = losses / max_loss
    n = len(losses)
    d = (nrm_losses)**2 + (np.arange(n,0,-1)/n)**2
    knee = np.argmin(d)
    knees.append(knee)
    best_active_nodes.append(active_nodes[model_idx][knee])
    best_alphas.append(l1_alphas[knee])

#%%
# Find the overall MSE validation loss for the collection of models
best_val_losses = np.zeros(len(validation_loss))
for model_idx in validation_loss:
    losses = validation_loss[model_idx]
    knee = knees[model_idx]
    loss = validation_loss[model_idx][knee]
    best_val_losses[model_idx] = loss
print(np.mean(best_val_losses))
#%%
plt.hist(best_active_nodes, bins=50);
print(np.median(best_active_nodes))
print((np.array(best_active_nodes) > 0).sum(), len(best_active_nodes))

# %%
from collections import Counter
print(Counter(best_alphas).most_common())

# %%
