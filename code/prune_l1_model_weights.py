#%%
from flow_model import L1FlowModel
import torch 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util import tonp, velocity_vectors 
from sklearn.cluster import AgglomerativeClustering
from torch.nn.utils.prune import l1_unstructured
from copy import deepcopy
from threshold_prune import loss_threshold_l1_prune

#%%
via = pickle.load(open(f'../data/mutant_pseudotime.pickle', 'rb'))

#%%
num_nodes = via.data.shape[1]
hidden_dim = num_nodes*2
num_layers = 3

device = 'cuda:0'
model = L1FlowModel(input_dim=num_nodes, 
                            output_dim=num_nodes, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers).to(device)
tmstp = '20230330_172620'
model.load_state_dict(torch.load(f'../models/l1_flow_model_wildtype_{tmstp}.torch'))


# %%
# Get the input weights of the trained model
data = via.data
idxs = torch.arange(data.shape[0])
starts = torch.tensor(data[idxs], device=device)
pV, original_weights = model(starts)
original_weights = tonp(original_weights)
# Plot a histogram of the input weights
plt.hist(np.log(original_weights.flatten()), bins=100);
# %%
# Plot the distribution of the number of input weights that are non-zero per node
# Get the 90th percentile of the input weights
pct = np.percentile(np.abs(original_weights), 90)
# Number of input weights that are non-zero per node
plt.hist(np.sum(np.abs(original_weights) > pct, axis=0));
# %%
# Hierarchical clustering of the input weights
cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
xlabels = cluster.fit_predict(original_weights)
ylabels = cluster.fit_predict(original_weights.T)
# Sort the data by both the x and y labels
xidx = np.argsort(xlabels)
yidx = np.argsort(ylabels)
# Plot the input weights, ordered by the clustering
plt.imshow(original_weights[xidx,:][:,yidx])
plt.grid(False)

#%%
pruned_weights = deepcopy(model.model[0])
# Prune 90% of the input weights, then replot
l1_unstructured(pruned_weights, name='weight', amount=0.9)
pruned_weights = tonp(pruned_weights.weight)
xlabels = cluster.fit_predict(pruned_weights)
ylabels = cluster.fit_predict(pruned_weights.T)
# Sort the data by both the x and y labels
xidx = np.argsort(xlabels)
yidx = np.argsort(ylabels)
# Plot the input weights, ordered by the clustering
plt.imshow(pruned_weights[xidx,:][:,yidx]>0)
# Remove the grid lines from the plot
plt.grid(False)

# %%
T = via.sc_transition_matrix(smooth_transition=1)
X = via.data
mutant_expression_tensor = torch.tensor(X, device=device)
pV = model(mutant_expression_tensor)

pca_ = pickle.load(open(f'../data/pca_mutant.pickle', 'rb'))

def embed(X):
    return pca_.transform(X)[:,1:3]

V = velocity_vectors(T, via.data)
Vgpu = torch.tensor(V, device=device)


#%%
fig, ax1 = plt.subplots()
ax1.plot(prune_pcts, losses)
# Make the x-axis logarithmic
ax1.set_xscale('log')
# label the axes
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Loss')
# Plot the number of nonzero weights on a second y-axis
ax2 = plt.twinx()
ax2.plot(prune_pcts, nonzero_counts, color='orange')
# Label the second y-axis
ax2.set_ylabel('Number of nonzero weights')
# Make a custom legend that manually labels the two lines
lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
labels = ['Loss', 'Number of nonzero weights']
# Put the legend below the main plot
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
## %%

# %%
print(f'Original loss: {original_loss:.9f}, increased by 10% {original_loss*1.1:.9f}')
# Find the closest loss in the list of losses to the 10% increase
losses = np.array(losses)
idx = np.argmin(np.abs(losses - original_loss*1.1))
print(f'Closest loss: {losses[idx]:.9f}, num nonzero weights: {nonzero_counts[idx]}')
threshold = np.percentile(tonp(original_weights), prune_pcts[idx]*100)
plt.hist(np.sum(np.abs(tonp(original_weights))>threshold, axis=0), bins=100);

# %%
# Save the pruned model
pruned_model = deepcopy(model)
l1_unstructured(pruned_model.model[0], name='weight', amount=prune_pcts[idx])
torch.save(pruned_model.state_dict(), f'../models/l1_flow_model_wildtype_{tmstp}_pruned.torch')
# %%
