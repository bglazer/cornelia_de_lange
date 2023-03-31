#%%
from flow_model import L1FlowModel
import torch 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util import tonp, velocity_vectors 
from sklearn.cluster import AgglomerativeClustering

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
pV, input_weights = model(starts)
input_weights = tonp(input_weights)
# Plot a histogram of the input weights
plt.hist(np.log(input_weights.flatten()), bins=100);
# %%
threshold = 2.5e-4
# Number of input weights that are non-zero per node
plt.hist(np.sum(np.abs(input_weights) > threshold, axis=0));
# %%
# Hierarchical clustering of the input weights
cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
cluster.fit_predict(input_weights)
# Plot the input weights, ordered by the clustering
plt.imshow(input_weights[np.argsort(cluster.labels_),:])

#%% 
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
# Make a copy of the original model
from copy import deepcopy

mse = torch.nn.MSELoss()
pV, original_weights = model(starts)
velocity = Vgpu[idxs]
original_loss = mse(pV, velocity).item()
print(f'Original loss: {original_loss:.9f}')

with torch.no_grad():
    min_weight = torch.log10(torch.min(torch.abs(original_weights)))
    max_weight = torch.log10(torch.max(torch.abs(original_weights)))
    thresholds = np.logspace(min_weight.item(), max_weight.item(), base=10, num=200)

    losses = []
    nonzero_counts = []
    for threshold in thresholds:
        pruned_model = deepcopy(model)
        zero_weights = torch.abs(pruned_model.model[0].weight) < threshold
        num_nonzero_weights = torch.sum(~zero_weights)
        pruned_model.model[0].weight[zero_weights] = 0
        pV, input_weights = pruned_model(starts)
        velocity = Vgpu[idxs]
        # Compute the loss between the predicted and true velocity vectors
        loss = mse(pV, velocity)
        losses.append(loss.item())
        nonzero_counts.append(num_nonzero_weights.item())
        #print(f'Theshold: {threshold:.3e}, Loss: {loss.item():.9f}, Num nonzero weights: {num_nonzero_weights.item()}')

#%%
fig, ax = plt.subplots()
ax.plot(thresholds, losses)
# Make the x-axis logarithmic
ax.set_xscale('log')
# label the axes
ax.set_xlabel('Threshold')
ax.set_ylabel('Loss')
# Plot the number of nonzero weights on a second y-axis
ax2 = plt.twinx()
ax2.plot(thresholds, nonzero_counts, color='orange')
# Label the second y-axis
ax2.set_ylabel('Number of nonzero weights')
# Make a custom legend that manually labels the two lines
lines = [ax.get_lines()[0], ax2.get_lines()[0]]
labels = ['Loss', 'Number of nonzero weights']
# Put the legend below the main plot
ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
## %%

# %%
print(f'Original loss: {original_loss:.9f}, increased by 10% {original_loss*1.1:.9f}')
# Find the closest loss in the list of losses to the 10% increase
losses = np.array(losses)
idx = np.argmin(np.abs(losses - original_loss*1.1))
print(f'Closest loss: {losses[idx]:.9f}, num nonzero weights: {nonzero_counts[idx]}')
plt.hist(np.sum(np.abs(tonp(original_weights))>thresholds[idx], axis=0), bins=100);

# %%
