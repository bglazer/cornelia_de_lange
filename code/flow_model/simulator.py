import numpy as np
import torch
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from tqdm import tqdm

class Simulator():
    def __init__(self, model, X):
        self.model = model
        print('Building KDTree')
        self.tree = KDTree(X, leaf_size=100)
        print('Querying KDTree')
        self.nearest_neighb_dists = self.tree.query(X, k=2)[0][:,1]
        self.max_distance = np.percentile(self.nearest_neighb_dists, 99)

    def euler_step(self, x, dt):
        with torch.no_grad():
            dx, var = self.model(x)
            var[var < 0] = 0
            std = torch.sqrt(var)
            inside = np.zeros(x.shape[0], dtype=bool)
            noise = torch.zeros_like(x)
            while True:
                num_outside = (~inside).sum()
                noise[~inside] = torch.randn(num_outside, noise.shape[1]) * std[~inside]
                #*****************************
                # TODO why is this negative?
                #*****************************
                x1 = x - dt*(dx+noise)
                dist, _ = self.tree.query(x1[~inside], k=2)
                inside_query = dist[:,1] < self.max_distance
                inside[~inside] = inside_query
                if np.all(inside):
                    break
                
            return x1

    def simulate(self, x, t_span, knockout_idx=None):
        last_t = t_span[0]
        traj = torch.zeros(len(t_span), x.shape[0], x.shape[1])
        traj[0,:,:] = x.clone()
        if knockout_idx is not None:
            traj[0,:,knockout_idx] = 0.0
        for i,t in tqdm(enumerate(t_span[1:]), total=len(t_span)-1):
            dt = t - last_t
            traj[i+1] = self.euler_step(traj[i], dt)
            if knockout_idx is not None:
                traj[i+1,:,knockout_idx] = 0.0
            last_t = t
        return traj

    def trajectory(self, x, t_span, knockout_idx, n_repeats=1, n_parallel=1):
        with torch.no_grad():
            x = x.clone()
            parallel = Parallel(n_jobs=n_parallel, verbose=0)
            jobs = []
            for j in range(n_repeats):
                jobs.append(delayed(self.simulate)(x, t_span, knockout_idx))
            trajectories = parallel(jobs)
            traj = torch.concatenate(trajectories, dim=1)
            return traj
