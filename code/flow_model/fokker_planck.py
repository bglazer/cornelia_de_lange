#%%
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
# import scanpy as sc
# from torch import relu
from torch.nn.functional import leaky_relu, relu
from scipy.stats import rv_histogram
from flow_model import MLP

#%%
# genotype='wildtype'
# dataset = 'net'
# adata = sc.read_h5ad(f'../../data/{genotype}_{dataset}.h5ad')
# data = adata.layers['raw']
# X = data.toarray()
device = 'cuda:0'

#%%
# This generates a scipy continuous distribution from a histogram
def gen_p(x, bins=100):
    hist = np.histogram(x, bins=bins, density=True)
    phist = rv_histogram(hist, density=True)
    return phist

# %%
# The u(x) drift term of the Fokker-Planck equation, modeled by a neural network
class Ux(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers, device):
        super(Ux, self).__init__()
        # The drift term of the Fokker-Planck equation, modeled by a neural network
        self.model = MLP(1, 1, hidden_dim, n_layers).to(device)

    def forward(self, x):
        return self.model(x)
    
    # Compute the partial derivative of u(x) with respect to x
    def dx(self, x):
        y = self.model(x)
        y.backward(torch.ones_like(y, device=x.device))
        xgrad = x.grad.clone()
        x.grad.zero_()
        return xgrad

# The p(x,t) term of the Fokker-Planck equation, modeled by a neural network
class Pxt(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers, device):
        super(Pxt, self).__init__()
        self.model = MLP(2, 1, hidden_dim, n_layers).to(device)
        self.device = device

    # Convert scalar t to a tensor of the same shape as x, for input to the model
    def t_(self, t, shape):
        return torch.ones(shape, device=self.device)*t
    
    def p(self, x, t):
        return torch.exp(self.model(torch.hstack((x, t))))

    # Compute the probability density p(x,t) using the neural network
    def forward(self, x, t):
        t = self.t_(t, x.shape)
        p = self.p(x, t)
        return p
    
    # Compute the partial derivative of p(x,t) with respect to x
    def dx(self, x, t):
        x.requires_grad = True
        x.grad.zero_()
        t = self.t_(t, x.shape)
        y = self.p(x, t)
        y.backward(torch.ones_like(y, device=x.device))
        xgrad = x.grad.clone()
        x.grad.zero_() 
        return xgrad

    # Compute the partial derivative of p(x,t) with respect to t
    def dt(self, x, t):
        t = self.t_(t, x.shape)
        t.requires_grad_(True)
        y = self.p(x, t)
        y.backward(torch.ones_like(y, device=x.device))
        tgrad = t.grad.clone()
        t.grad.zero_()
        return tgrad


#%%
# TODO instead of using a for loop, should use a combined
# tensor across all timesteps. Not sure if this will 
# consume all memory though.
# xs = x.repeat((1,ts.shape[0])).T.unsqueeze(2)
# tss = ts.repeat((x.shape[0],1)).T.unsqueeze(2).to(device)
# xts = torch.concatenate((xs,tss), dim=2)
# ps = pxt.model(xts)

# Integral of d/dx p(x,t) dt
def Spxt_dx_dt(pxt, x, hx, steps=100):
    ts = torch.linspace(0, 1, steps, device=pxt.device)
    ht = ts[1] - ts[0]
    sum = torch.zeros(x.shape, device=pxt.device, requires_grad=True)
    for t in ts:
        dx = (pxt(x+hx, t) - pxt(x-hx, t))/(2*hx)
        sum = sum + (dx*ht)
    return sum

#%%
# Generate simple test data
# Initial distribution
X0 = torch.randn((1000, 1), device=device)
# Final distribution is initial distribution plus another dist. shifted by 4
X1 = torch.randn((1000, 1), device=device)+4
X1 = torch.vstack((X0, X1))
# Sort X1
# X1 = X1[torch.argsort(X1, dim=0).squeeze()]
# Plot the two distributions
_=plt.hist(X0.cpu().numpy(), bins=30, alpha=.3)
_=plt.hist(X1.cpu().numpy(), bins=30, alpha=.3)
#%%
#%%
epochs = 500
steps = 1
hx = 1e-3
ht = 1e-3
#%%
# Generate a continuous distribution from the data
pD = gen_p(X1.cpu().numpy(), bins=100)
# Initialize the neural networks
pxt = Pxt(20, 3, device)
ux = Ux(20, 3, device)
pxt_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-4)
ux_optimizer = torch.optim.Adam(ux.parameters(), lr=1e-3)

#%%
# TODO this should be its own distribution
pD0 = torch.tensor(pD.pdf(X0.cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
pD0_cdf = pD0.cumsum(dim=0)/pD0.sum()
px = torch.tensor(pD.pdf(X1.cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
# px_cdf = px.cumsum(dim=0)/px.sum()

for i in range(epochs):
    pxt_optimizer.zero_grad()
    ux_optimizer.zero_grad()
    # x = X1.clone().detach().requires_grad_(True)
    for j in range(steps):
        x = pD.rvs(size=(1000,1))
        px = torch.tensor(pD.pdf(x), device=device, dtype=torch.float32, requires_grad=False)
        x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=True)
        ppx = (pxt(x, t=1) - pxt(x, t=0) + ux(x).detach() * Spxt_dx_dt(pxt, x, hx=hx))/ux.dx(x)
        # ppx = ppx.clip(0)
        
        # ppx_cdf = ppx.cumsum(dim=0)/ppx.sum()
        # Loss is the difference between the predicted and actual probability density
        # across the whole data distribution p(x ~ X1)
        # and the initial distribution p(x,t=0) = p(x ~ X0) 
        # pxt0 = pxt(X0, t=0)
        # pxt0_cdf = pxt0.cumsum(dim=0)/pxt0.sum()
        l_px = ((px - ppx)**2).mean() #+ ((pxt(X0, t=0) - pD0)**2).mean()
        # l_px = 
        #+ ((pxt(X0, t=0) - pD0)**2).mean()# + \
        #    ((px_cdf - ppx_cdf)**2).mean() + ((pxt0_cdf - pD0_cdf)**2).mean()
        l_px.backward()
    torch.nn.utils.clip_grad_norm_(pxt.parameters(), .001)
    # pxt_optimizer.step()
    pxt_optimizer.step()

    pxt_optimizer.zero_grad()
    ux_optimizer.zero_grad()

    # TODO not sure I need to do this again, or even at all
    # x = X1.clone().detach().requires_grad_(True)
    # for j in range(steps):
    #     x = pD.rvs(size=(1000,1))
    #     x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=True)
    #     # x = X1.cpu().numpy()
    #     ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
    #     for t in ts:
    #         # TODO check the sign on pxt.dt
    #         l_fp = (pxt.dt(x, t) + ux(x) * pxt.dx(x, t) + pxt(x, t).detach() * (ux(x+hx) - ux(x-hx))/(2*hx))**2
    #         l_fp.mean().backward()
    # torch.nn.utils.clip_grad_norm_(ux.parameters(), .001)
    # ux_optimizer.step()

    print(f'{i} l_fp={float(l_fp.mean()):.5f}, l_px={float(l_px.mean()):.5f}')
    # print(f'{i} l_fp={0:.5f}, l_px={float(l_px.mean()):.5f}')

#%%
xs = torch.arange(X1.min(), X1.max(), .1, device=device)[:,None]
xs.requires_grad_(True)
ppxs = (pxt(xs, t=1) - pxt(xs, t=0) + ux(xs).detach() * Spxt_dx_dt(pxt, xs, hx=hx))/ux.dx(xs)
px = torch.tensor(pD.pdf(xs.detach().cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
plt.plot(xs.detach().cpu().numpy(), ppxs.cpu().detach().numpy(), label='Model')
plt.plot(xs.detach().cpu().numpy(), px.cpu().numpy(), label='Data')
plt.legend()

# %%
colors = matplotlib.colormaps.get_cmap('viridis')
xs = torch.arange(X1.min(), X1.max(), .1, device=device)[:,None]
for t in torch.linspace(0, 1, 100, device=device):
    pxst = pxt(xs, t)
    plt.plot(xs.cpu().numpy(), pxst.cpu().detach().numpy(), c=colors(float(t)))
    plt.plot(xs.cpu().numpy(), pD.pdf(xs.cpu().numpy()), c='k', alpha=1)
# plt.colorbar()
#%%
ts = torch.linspace(0, 1, 100, device=device)
ppx = torch.zeros_like(xs)
for t in ts:
    pxst = pxt(xs, t)
    ppx = ppx + pxst/len(ts)
plt.plot(xs.cpu().numpy(), ppx.cpu().detach().numpy())
plt.plot(xs.cpu().numpy(), pD.pdf(xs.cpu().numpy()), c='k', alpha=1)
    
#%%
# Plot the cdf of the data and the final distribution
# plt.plot(X1.cpu().numpy(), px_cdf.cpu().numpy(), label='Data CDF')
# ppx_cdf = ppx.clip(0).cumsum(dim=0)/ppx.clip(0).sum()
# plt.plot(X1.cpu().numpy(), ppx_cdf.detach().cpu().numpy(), label='Model CDF')
# plt.legend()
# %%
xs = torch.arange(X1.min(), X1.max(), .1, device=device)[:,None]
ts = torch.linspace(0, 1, 100, device=device)
pxst = torch.zeros(xs.shape[0], ts.shape[0], device=device)
for i,t in enumerate(ts):
    pxst[:,i] = pxt(xs, t).squeeze()

# Cumulative mean of p(x,t) at each timestep t
cum_pxst = pxst.cumsum(dim=1) / torch.arange(1, ts.shape[0]+1, device=device)[None,:]
plt.imshow(cum_pxst.cpu().detach().numpy(), aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()

#%%
plt.imshow(pxst.cpu().detach().numpy(), aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()

# %%
# for x in xs:
#     print(float(ux(x)))
# %%
plt.plot(xs.cpu().detach().numpy(), ux(xs).cpu().detach().numpy())
# %%
