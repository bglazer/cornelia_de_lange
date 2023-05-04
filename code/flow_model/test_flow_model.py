#%%
from flow_model import L1FlowModel
import torch

#%%
num_nodes = 20
flow_model = L1FlowModel(input_dim=num_nodes, hidden_dim=10, num_layers=2).to('cuda:0')

#%%
# input is num_samples x num_nodes
from torch.optim import Adam
optimizer = Adam(flow_model.parameters(), lr=1e-3)
# breakpoint()
# %%
y = torch.zeros((42, num_nodes), device='cuda:0')
loss = torch.nn.MSELoss(reduction='mean')
for i in range(500):
    optimizer.zero_grad()
    yhat = flow_model(torch.zeros((42, num_nodes), device='cuda:0'))
    l=loss(yhat, y)
    l1 = torch.mean(torch.abs(torch.cat([m.l1 for m in flow_model.models])))
    total = l + l1
    total.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, l.item(), l1.item(), total.item(), flush=True)

print(flow_model.models[0].l1, flush=True)

breakpoint()
