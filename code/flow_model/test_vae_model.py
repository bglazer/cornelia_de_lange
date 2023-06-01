#%%
%load_ext autoreload
%autoreload 2
#%%
from flow_model import VAEFlowModel
import torch

#%%
num_nodes = 20
flow_model = VAEFlowModel(input_dim=num_nodes, hidden_dim=10, latent_dim=10, num_layers=2).to('cuda:0')

#%%
# input is num_samples x num_nodes
from torch.optim import Adam
optimizer = Adam(flow_model.parameters(), lr=1e-3)
# breakpoint()
# %%
x = torch.randn((42, num_nodes), device='cuda:0')
y = torch.randn((10,42, num_nodes), device='cuda:0')
#%%
flow_model(x, nsamples=10)
#%%
loss = torch.nn.MSELoss(reduction='mean')
for i in range(100):
    optimizer.zero_grad()
    yhat = flow_model(x)
    l=loss(yhat, y)
    l1 = flow_model.group_l1.mean()
    
    total = l + l1
    total.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, l.item(), total.item(), flush=True)

print(flow_model.models[0].group_l1, flush=True)

breakpoint()

# %%
