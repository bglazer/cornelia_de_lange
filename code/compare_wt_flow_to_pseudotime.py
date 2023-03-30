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
# Set rows with any nan to all zero
V[np.isnan(V).sum(axis=1) > 0] = 0

embedding = embed(via.data)
V_emb = embed_velocity(via.data, V, embed)

idxs = np.arange(0, via.data.shape[0])
dpv = embed_velocity(X=X,
                    velocity=tonp(pV),
                    embed_fn=embed)

plot_arrows(idxs=idxs,
            points=embedding, 
            V=V_emb*10,
            pV=dpv*10,
            sample_every=5,
            scatter=False,
            save_file=f'../figures/embedding/predicted_vs_pseudotime_mutant_vector_field.png',
            c=via.single_cell_pt_markov,
            s=.5)
# %%
