#%%
import torch
from flow_model import ConnectedFlowModel
import pickle
from util import tonp, plot_arrows, embed_velocity, velocity_vectors
import numpy as np

#%%
via = pickle.load(open(f'../data/mutant_pseudotime.pickle', 'rb'))

#%%
num_nodes = via.data.shape[1]
hidden_dim = num_nodes*2
num_layers = 3

device = 'cuda:0'
model = ConnectedFlowModel(input_dim=num_nodes, 
                            output_dim=num_nodes, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers).to(device)
tmstp = '20230330_152219'
model.load_state_dict(torch.load(f'../models/connected_flow_model_wildtype_{tmstp}.torch'))

# %%
T = via.sc_transition_matrix(smooth_transition=1)
X = via.data
mutant_expression_tensor = torch.tensor(X, device=device)
pV = model(mutant_expression_tensor)

pca_ = pickle.load(open(f'../data/pca_mutant.pickle', 'rb'))

def embed(X):
    return pca_.transform(X)[:,1:3]

V = velocity_vectors(T, via.data)

embedding = embed(via.data)
V_emb = embed_velocity(via.data, V, embed)

idxs = np.arange(0, via.data.shape[0])
dpV = embed_velocity(X=X,
                    velocity=tonp(pV),
                    embed_fn=embed)


#%%
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Calculate the cosine similarity between the true and predicted vectors
similarities = np.zeros(len(idxs))
pVnp = tonp(pV)
for i in range(len(idxs)):
    similarities[i] = cosine_similarity(V[i], pVnp[i])
similarities[np.isnan(similarities)] = 0
# Normalize the similarities to be between 0 and 1
min = -1
max = 1
similarities = (similarities - min) / (max - min)
# Invert the similarities so that the most dissimilar vectors have the highest alpha
alphas = 1-similarities

# %%

plot_arrows(idxs=idxs,
            points=embedding, 
            V=V_emb*10,
            pV=dpV*10,
            sample_every=5,
            scatter=False,
            save_file=f'../figures/embedding/predicted_vs_pseudotime_mutant_vector_field.png',
            c=via.single_cell_pt_markov,
            alphas=similarities,
            s=.5)
# %%
