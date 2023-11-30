#%%
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import rv_histogram, gamma
from torch.nn import Linear, ReLU
from flow_model import L1FlowModel

# Generic Dense Multi-Layer Perceptron (MLP), which is just a stack of linear layers with ReLU activations
# input_dim: dimension of input
# output_dim: dimension of output
# hidden_dim: dimension of hidden layers
# num_layers: number of hidden layers
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, input_bias=True):
        super(MLP, self).__init__()
        layers = []
        layers.append(Linear(input_dim, hidden_dim, bias=input_bias))
        layers.append(ReLU())
        for i in range(num_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(ReLU())
            # TODO do we need batch norm here?
        layers.append(Linear(hidden_dim, output_dim, bias=False))
        # layers.append(LeakyReLU())
        # Register the layers as a module of the model
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#%%
# This generates a scipy continuous distribution from a histogram
def gen_p(x, bins=100):
    hist = np.histogram(x, bins=bins, density=True)
    phist = rv_histogram(hist, density=True)
    return phist

#%%
# Zero Inflated Gamma distribution
class ZIG():
    def __init__(self, x):
        self.x = x
        # Find probability of zero entries
        self.p = (x==0).mean()
        # Use scipy to fit a gamma distribution to the data
        # Get the nonzero entries
        xnz = x[x>0]
        alpha, loc, beta = gamma.fit(xnz)
        self.gamma = gamma(alpha, loc, beta)

    def pdf(self, x):
        p = np.zeros_like(x)
        # Nonzero entries have a probability defined by the gamma distribution
        # TODO this won't work on the GPU with torch tensors
        # TODO this isn't actually a probability distribution?
        p[x>0] = self.gamma.pdf(x[x>0])
        p[x==0] = self.p

        return p 


# %%
# The u(x) drift term of the Fokker-Planck equation, modeled by a neural network
class Ux(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, device):
        super(Ux, self).__init__()
        # The drift term of the Fokker-Planck equation, modeled by a neural network
        self.model = MLP(input_dim=in_dim, 
                         output_dim=1, 
                         hidden_dim=hidden_dim, 
                         num_layers=n_layers).to(device)

    def forward(self, x):
        return self.model(x)
    
    # Compute the derivative of u(x) with respect to x
    def dx(self, x, hx=1e-3):
        xgrad = (self(x+hx) - self(x-hx))/(2*hx)
        return xgrad

# The p(x,t) term of the Fokker-Planck equation, modeled by a neural network
class Pxt(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers, device):
        super(Pxt, self).__init__()
        self.model = MLP(2, 1, hidden_dim, n_layers).to(device)
        self.device = device

    def p(self, x, ts):
        xs = x.repeat((1,ts.shape[0])).T.unsqueeze(2)
        tss = ts.repeat((x.shape[0],1)).T.unsqueeze(2).to(device)
        xts = torch.concatenate((xs,tss), dim=2)
        ps = torch.exp(pxt.model(xts))
        return ps

    # Compute the probability density p(x,t) using the neural network
    def forward(self, x, ts):
        p = self.p(x, ts)
        return p
    
    # Compute the partial derivative of p(x,t) with respect to x
    def dx(self, x, ts, hx=1e-3):
        xgrad = (self.p(x+hx, ts) - self.p(x-hx, ts))/(2*hx)
        return xgrad

    # Compute the partial derivative of p(x,t) with respect to t
    def dt(self, x, ts, ht=1e-3):
        tgrad = (self.p(x, ts+ht) - self.p(x, ts-ht))/(2*ht)
        return tgrad

#%%
device = 'cuda:0'
#%%
genotype='wildtype'
dataset = 'net'
adata = sc.read_h5ad(f'../../data/{genotype}_{dataset}.h5ad')
# data = adata.layers['raw']
#%%
cell_types = adata.obs['cell_type']
nmp_cells = cell_types[cell_types=='NMP']
gene = 'POU5F1'
geneX = adata[:,gene].X.toarray()
X = adata.X.toarray()
X0 = geneX[cell_types=='NMP']
X1 = geneX.copy()

#%%
# Generate simple test data
# Initial distribution
# X0 = torch.randn((1000, 1), device=device)
# # Final distribution is initial distribution plus another normal shifted by +4
# X1 = torch.randn((1000, 1), device=device)+4
# X1 = torch.vstack((X0, X1))
# Plot the two distributions
_=plt.hist(X0, bins=30, alpha=.3)
_=plt.hist(X1, bins=30, alpha=.3)
#%%
X0 = torch.tensor(X0, device=device, dtype=torch.float32, requires_grad=False)
X1 = torch.tensor(X1, device=device, dtype=torch.float32, requires_grad=False)
#%%
epochs = 500
steps = 1
hx = 1e-3
ht = 1e-3
#%%
# Generate a continuous distribution from the data
pD = gen_p(X1.cpu().numpy(), bins=100)
pD0 = gen_p(X0.cpu().numpy(), bins=100)
# Generate a set of points to evaluate the distribution
span = X1.max() - X1.min()
x0 = torch.linspace(X1.min()-span*.25, X1.max()+span*.25, 1000, device=device, requires_grad=False)[:,None]
# x0 = torch.linspace(X1.min(), X1.max(), 1000, device=device, requires_grad=False)[:,None]
pX_D0 = torch.tensor(pD0.pdf(x0.cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
z_pX_D0 = float(pX_D0.sum())
pX = torch.tensor(pD.pdf(x0.cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
z_pX = float(pX.sum())
# pX_D0 = pX_D0 / z_pX_D0
# pX = pX / z_pX
#%%
# Initialize the neural networks
n_genes = X.shape[1]
pxt = Pxt(20, 3, device)
ux = Ux(1, 20, 3, device)
# Initialize the optimizers
pxt_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-3)
ux_optimizer = torch.optim.Adam(ux.parameters(), lr=1e-3)
##%%
# # This is a pre-training step to get p(x, t=0) to match the initial condition
# pxt0_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-3)
zero = torch.zeros(1)
# for i in range(1000):
#     pxt0_optimizer.zero_grad()
#     l_pD0 = ((pxt(x0, ts=zero) - pX_D0)**2).mean()
#     l_pD0.backward()
#     # torch.nn.utils.clip_grad_norm_(pxt.parameters(), .001)
#     pxt0_optimizer.step()
#     print(f'{i} l_pD0={float(l_pD0.mean()):.5f}')
#%%
ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
ht = ts[1] - ts[0]

l_Spxts = np.zeros(epochs)
l_p0s = np.zeros(epochs)
l_fps = np.zeros(epochs)
l_us = np.zeros(epochs)
l_pxt_dx = np.zeros(epochs)
for epoch in range(epochs):
    # Sample from the data distribution
    x = pD.rvs(size=1000)
    px = torch.tensor(pD.pdf(x), device=device, dtype=torch.float32, requires_grad=False)
    x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=False)[:,None]

    pxt_optimizer.zero_grad()

    # This is the initial condition p(x, t=0)=p_D0 
    l_p0 = ((pxt(x0, ts=zero) - pX_D0)**2).mean()
    l_p0.backward()

    # Boundary conditions
    # Ensure that p(x,t) is zero at the boundaries
    xlt0 = torch.arange(-1,-1e-5,.01, device=device, requires_grad=False)[:,None]
    l_pxt_bc = (pxt(xlt0, ts)**2).mean()
    l_pxt_bc.backward()

    # This is the marginal p(x) = int p(x,t) dt
    Spxt = pxt(x, ts)[1:].sum(dim=0) * ht
    # Ensure that the marginal p(x) matches the data distribution
    l_Spxt = ((Spxt[:,0] - px)**2).mean()
    l_Spxt.backward()

    # Smoothness regularization of the d/dx p(x,t) term
    # l_pxt_dt = ((pxt.dt(x, ts+ht) - pxt.dt(x, ts-ht)/(2*ht))**2).mean()*.00
    # l_pxt_dt.backward()

    # Record the losses
    l_Spxts[epoch] = float(l_Spxt.mean())
    l_p0s[epoch] = float(l_p0.mean())   

    low = float(X1.min())
    high = float(X1.max())
    l = low-.25*(high-low)
    h = high+.25*(high-low)
    x = torch.arange(l, h, .01, device=device)[:,None]

    # for epoch in range(epochs):
    ux_optimizer.zero_grad()

    # This is the calculation of the term that ensures the
    # derivatives match the Fokker-Planck equation
    # d/dx p(x,t) = -d/dt (u(x) p(x,t))
    up_dx = (ux(x+hx) * pxt(x+hx, ts) - ux(x-hx) * pxt(x-hx, ts))/(2*hx)
    pxt_dts = pxt.dt(x, ts)
    pxt_dts[:,x<0] = 0
    l_fp = ((pxt_dts + up_dx)**2).mean()

    # Set ux(x) to zero at the boundaries
    # l_ux_bc = (ux(xlt0)**2).mean()
    # l_fp += l_ux_bc

    l_fp.backward()

    # Penalize the magnitude of u(x)
    # l_u = (ux(x)**2).mean()*0
    # l_u.backward()

    # Take a gradient step
    pxt_optimizer.step()
    ux_optimizer.step()
    l_fps[epoch] = float(l_fp.mean())
    # l_us[epoch] = float(l_u)
    print(f'{epoch} l_px={float(l_Spxt.mean()):.5f}, l_p0={float(l_p0.mean()):.5f}, '
          f'{epoch} l_fp={float(l_fp.mean()):.5f}')

#%%
plt.title('Loss curves')
plt.plot(l_fps[10:], label='l_fp')
plt.plot(l_Spxts[10:], label='l_Spxt')
plt.plot(l_p0s[10:], label='l_p0')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# %%
# Plot the predicted p(x,t) at each timestep t
colors = matplotlib.colormaps.get_cmap('viridis')
viridis = matplotlib.colormaps.get_cmap('viridis')
greys = matplotlib.colormaps.get_cmap('Greys')
purples = matplotlib.colormaps.get_cmap('Purples')
low = float(X1.min())
high = float(X1.max())
l = low-.25*(high-low) 
h = high+.25*(high-low)
xs = torch.arange(l, h, .01, device=device)[:,None]

pxts = pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = ux(xs).squeeze().cpu().detach().numpy()
up_dx = (ux(xs+hx) * pxt(xs+hx, ts) - ux(xs-hx) * pxt(xs-hx, ts))/(2*hx)
pxt_dts = pxt.dt(xs, ts)
up_dx = up_dx.detach().cpu().numpy()[:,:,0]
pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()
#%%
plt.title('p(x,t)')
plt.plot(xs,  pD.pdf(xs), c='k', alpha=1)
plt.plot(xs, pD0.pdf(xs), c='k', alpha=1)
for i in range(0, ts.shape[0], len(ts)//10):
    t = ts[i]
    plt.plot(xs, pxts[:,i], c=colors(float(t)))
plt.xlabel('x')
plt.ylabel('p(x,t)')
# Add a colorbar to show the timestep
sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, label='timestep - t')

# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
plt.title('Cumulative mean of p(x,t)')
cum_pxt = pxts.cumsum(axis=1) / np.arange(1, ts.shape[0]+1)
plt.imshow(cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep - t')
plt.colorbar()
# %%
# This plots the error of the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
pxs = pD.pdf(xs)
plt.title('Error of cumulative mean of p(x,t)')
plt.imshow(pxs[:,None] - cum_pxt, aspect='auto', interpolation='none', cmap='RdBu')
plt.ylabel('x')
plt.xlabel('timestep - t')
plt.colorbar()

#%%
# This is the individual p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
plt.title('p(x,t) at each timestep t')
plt.imshow(pxts, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep - t')
plt.colorbar()
# %%
# Plot the u(x) term for all x
fig, ax1 = plt.subplots(1,1, figsize=(10,5))
plt.title('u(x) vs p(x)')
ax1.plot(xs, uxs, label='u(x)')
# Add vertical and horizontal grid lines
ax1.grid()
ax1.axhline(0, c='r', alpha=1)
ax1.set_ylabel('u(x)')
ax1.set_xlabel('x')
ax2 = ax1.twinx()
ax2.plot(xs, pD.pdf(xs), c='k', alpha=1, label='p(x)')
ax2.set_ylabel('p(x)')
fig.legend()
#%%
# Plot the Fokker Planck terms
for i in range(1):#0,len(ts),len(ts)//10):
    plt.plot(xs, pxt_dts[i,:], c='r')
    plt.plot(xs, up_dx[i,:], c='blue')
labels = ['d/dt p(x,t)', 'd/dx u(x) p(x,t)']
plt.legend(labels)
# %%
# Euler-Maruyama method for solving stochastic differential equations
# This is a test to see if the Euler-Maruyama method can reproduce the
# Fokker-Planck equation solution
# dX = u(X)dt + sigma(X)dW
# Sample from the initial distribution
x = pD0.rvs(size=1000)
x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=False)[:,None]
xts = []
ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
ht = ts[1] - ts[0]
dx_means = []
for t in ts:
    # Compute the drift term
    u = ux(x)
    # Compute the diffusion term
    dW = torch.randn_like(x) * torch.sqrt(ht)
    sigma = torch.ones_like(x)
    # Compute the change in x
    dx = u * ht + sigma * dW
    # Update x
    x = x + dx
    dx_means.append(float(dx.mean()))
    # Boundary Condition for gene expression data: Ensure that x is non-negative
    x[x<0] = 0
    xts.append(x.cpu().detach().numpy())
xts = np.concatenate(xts, axis=1)
#%%
# Plot the resulting probability densities at each timestep
low = float(xs.min())
high = float(xs.max())
bins = np.linspace(low, high, 100)
w = bins[1] - bins[0]
for i in range(0, ts.shape[0]-1, ts.shape[0]//10):
    t = ts[i]
    heights,bins = np.histogram(xts[:,i], 
                                bins=bins,
                                density=True)
    z = heights.sum()
    plt.bar(bins[:-1], heights/z, width=w, color=colors(float(t)), alpha=.2)
# Accumulate pxts into buckets of the same shape as the histogram
pxt_bins = np.zeros((bins.shape[0]-1, ts.shape[0]))
ranges = np.linspace(0, pxts.shape[0]-1, len(bins), dtype=int) 
pxt_bins = np.add.reduceat(pxts, ranges, axis=0) 
pxt_bins = pxt_bins / pxt_bins.sum(axis=0)

# for i in range(0, pxts.shape[1], pxts.shape[1]//10):
#     plt.plot(bins, pxt_bins[:,i], color='blue', alpha=.2)
# labels = ['Simulation', 'Fokker-Planck theoretical']
# artists = [plt.Line2D([0], [0], color=c, alpha=.2) for c in ['red', 'blue']]
# plt.legend(artists, labels)
sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, label='timestep - t')
plt.xlabel('x')
plt.ylabel('p(x,t)')
plt.title('p(x,t) Simulation')

#%%
# Plot the mean of the change in x at each timestep
plt.title('Mean of dX')
plt.plot(dx_means)
plt.xlabel('timestep (t)')
plt.ylabel('Mean of dX')
#%%
# Plot the cumulative distribution of the simulated data at each timestep
sim_pxts = np.zeros((bins.shape[0]-1, ts.shape[0]))
for i in range(0, ts.shape[0]):
    heights,bins = np.histogram(xts[:,i], 
                                bins=bins,
                                density=True)
    sim_pxts[:,i] = heights

sim_cum_pxt = sim_pxts.cumsum(axis=1) / np.arange(1, sim_pxts.shape[1]+1)[None,:]

# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
plt.title('Cumulative mean of p(x,t)')
plt.imshow(sim_cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.xticks(np.arange(0, ts.shape[0], ts.shape[0]//10), [f'{x:.1f}' for x in np.linspace(0, 1, 10)])
plt.yticks(np.arange(0, bins.shape[0], bins.shape[0]//10), [f'{x:.1f}' for x in np.linspace(xs.min(), xs.max(), 10)])
plt.colorbar() 

#%%
# Plot the final cumulative distribution of simulations versus the data distribution
plt.title('Final cumulative distribution of simulations vs data distribution')
z = sim_cum_pxt[:,-1].sum()
plt.plot(bins[:-1], sim_cum_pxt[:,-1]/z, label='Simulation')
z = pD.pdf(bins[:-1]).sum()
plt.plot(bins[:-1], pD.pdf(bins[:-1])/z, label='Data')
plt.legend()
plt.ylabel('Cumulative distribution')
plt.xlabel('x')


# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
pX = pD.pdf(bins[:-1])
sim_cum_pxt_err = (pX[:,None] - sim_cum_pxt)**2

plt.title('Error of Cumulative mean of p(x,t)\n'
          f'Error ∫pxt(x,t)dt = {l_Spxt:.5f}\n'
          f'Error Simulation = {sim_cum_pxt_err[:,-1].mean():.5f}')
plt.imshow(sim_cum_pxt_err, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar() 
# %%
