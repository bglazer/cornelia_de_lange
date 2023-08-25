import torch
from tqdm import tqdm

class Simulator():
    def __init__(self, model, X, device):
        self.model = model
        self.device = device
        # print('Finding max distance')
        dist, idxs = torch.sort(torch.cdist(X, X), dim=1)
        closest = dist[:,1] 
        d = closest.flatten()
        l = d.shape[0]
        q_idx = int(l*.99)
        d_sorted = torch.sort(d)[0]
        q = d_sorted[q_idx]
        self.max_distance = q
        self.X = X

    def euler_step(self, x, dt, boundary=True):
        with torch.no_grad():
            dx, var = self.model(x)
            var[var < 0] = 0
            std = torch.sqrt(var)
            inside = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
            nearest_idxs = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
            noise = torch.zeros_like(x, device=self.device)
            while True:
                num_outside = (~inside).sum()
                # Generate samples from a standard normal distribution
                r = torch.randn(num_outside, noise.shape[1], device=self.device)
                # Scale by the predicted standard deviation
                noise[~inside] = r * std[~inside]
                #*****************************
                # TODO why is this negative?
                #*****************************
                x1 = x - dt*(dx+noise)
                x1 = torch.clamp(x1, min=0.0)
                # Find the nearest neighbor of the points not on the interior
                dist, idxs = torch.sort(torch.cdist(x1[~inside], self.X), dim=1)
                closest = dist[:,1]
                nearest_idxs[~inside] = idxs[:,1]
                inside_query = closest < self.max_distance
                # Update the index of points on the interior
                inside[~inside] = inside_query
                if torch.all(inside) or (not boundary):
                    break
                
            return x1, nearest_idxs

    def simulate(self, start_idxs, t_span, node_perturbation=None, boundary=True, show_progress=True):
        with torch.no_grad():
            x = self.X[start_idxs,:].to(self.device)
            last_t = t_span[0]
            traj = torch.zeros(len(t_span), x.shape[0], x.shape[1], device=self.device)
            traj[0,:,:] = x.clone()
            nearest_idxs = torch.zeros(len(t_span), x.shape[0], dtype=torch.long, device=self.device)
            nearest_idxs[0,:] = start_idxs
            if node_perturbation is not None:
                perturb_idx, perturb_val = node_perturbation
                traj[0,:,perturb_idx] = perturb_val
            for i,t in tqdm(enumerate(t_span[1:]), total=len(t_span)-1, disable=(not show_progress)):
                dt = t - last_t
                x, idxs = self.euler_step(traj[i], dt, boundary=boundary)
                traj[i+1,:,:] = x
                nearest_idxs[i+1,:] = idxs
                if node_perturbation is not None:
                    traj[i+1,:,perturb_idx] = perturb_val
                last_t = t
            return traj, nearest_idxs

