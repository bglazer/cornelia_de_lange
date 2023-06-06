import scipy
import numpy as np
import torch
from scipy.spatial import KDTree
from joblib import Parallel, delayed
from tqdm import tqdm

class Simulator():
    def __init__(self, model, X, device):
        self.model = model
        self.device = device
        print('Finding max distance')
        d = torch.cdist(X, X).flatten()
        l = d.shape[0]
        q_idx = int(l*.95)
        d_sorted = torch.sort(d)[0]
        q = d_sorted[q_idx]
        self.max_distance = q
        self.X = X

    def euler_step(self, x, dt):
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
                # Find the nearest neighbor of the points not on the interior
                dist, idxs = torch.sort(torch.cdist(x1[~inside], self.X), dim=1)
                closest = dist[:,1]
                nearest_idxs[~inside] = idxs[:,1]
                inside_query = closest < self.max_distance
                # Update the index of points on the interior
                inside[~inside] = inside_query
                if torch.all(inside):
                    break
                
            return x1, nearest_idxs

    def simulate(self, start_idxs, t_span, knockout_idx=None):
        x = self.X[start_idxs,:].to(self.device)
        last_t = t_span[0]
        traj = torch.zeros(len(t_span), x.shape[0], x.shape[1], device=self.device)
        traj[0,:,:] = x.clone()
        nearest_idxs = torch.zeros(len(t_span), x.shape[0], dtype=torch.long, device=self.device)
        nearest_idxs[0,:] = start_idxs
        if knockout_idx is not None:
            traj[0,:,knockout_idx] = 0.0
        for i,t in tqdm(enumerate(t_span[1:]), total=len(t_span)-1):
            dt = t - last_t
            x, idxs = self.euler_step(traj[i], dt)
            traj[i+1,:,:] = x
            nearest_idxs[i+1,:] = idxs
            if knockout_idx is not None:
                traj[i+1,:,knockout_idx] = 0.0
            last_t = t
        return traj, nearest_idxs

