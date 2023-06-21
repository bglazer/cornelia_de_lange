#%%
import matplotlib.pyplot as plt
import pickle
import numpy as np
#%%
wt_tmstp = '20230607_165324'  
mut_tmstp = '20230608_093734'
outdir = f'../../output/'
#%%
wt_paths = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_paths_wildtype.pickle', 'rb'))
# %%
protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {pid: '/'.join(names) for pid, names in protein_id_name.items()}
protein_name_id = {protein_id_name[pid]: pid for pid in wt_paths.keys()}
#%%
pid = protein_name_id['NANOG']
paths = wt_paths[pid]
# %%
path_lens = [len(path) for path in paths]
max_len = max(path_lens)
levels = [set() for i in range(max_len)]
for path in paths:
    for i,node in enumerate(path[::-1]):
        levels[i].add(node)
widths = [len(level) for level in levels]
max_width = max(widths)
for j,level in enumerate(levels):
    level = list(level)
    levels[j] = {pid:i for i,pid in enumerate(level)}
#%%
# Plot the paths
def plot_paths(levels, paths):
    fig, ax = plt.subplots(figsize=(10,10))
    xy = {}
    for i,level in enumerate(levels):
        y = [i for j in range(len(level))]
        x = level.values()
        xy.update({pid:(x,i) for pid,x in level.items()})
        ax.scatter(x, y, s=100, label=f'Level {i}', c='grey')
        
    colors= plt.cm.tab20(np.linspace(0,1,len(paths)))
    for j, path in enumerate(paths):
        path = path[::-1]
        for i in range(len(path)-1):
            pid1 = path[i]
            pid2 = path[i+1]
            x1,y1 = xy[pid1]
            x2,y2 = xy[pid2]
            ax.plot([x1,x2],[y1,y2], c=colors[j])
    for pid,(x,y) in xy.items():
        name = protein_id_name[pid]
        ax.text(x, y, name, fontsize=10, fontdict={'family':'monospace'})
    # ax.set_xticks([])
    # ax.set_yticks([]);
# %%
import torch
from itertools import combinations, permutations, combinations_with_replacement
from tqdm import tqdm

#%%
def optimize_placement_torch(levels, steps=100, spacing=10):
    levels = [list(l) for l in levels]
    widths = [len(level) for level in levels]
    max_width = max(widths)
    vars = {}

    # Get a random uniform placement
    xs = [torch.rand(len(level), requires_grad=True)*max_width*spacing for level in levels]
    # xs = [torch.rand(len(level), requires_grad=True) for level in levels]
    xs = torch.nn.ParameterList(xs)

    optimizer = torch.optim.Adam(xs)
    # for i in tqdm(range(steps)):
    for step in range(steps):
        lengths = []
        for j, path in enumerate(paths):
            path = path[::-1]
            for i in range(len(path)-1):
                pid1 = path[i]
                pid2 = path[i+1]
                idx1 = levels[i].index(pid1)
                idx2 = levels[i+1].index(pid2)
                x1 = xs[i][idx1]
                x2 = xs[i+1][idx2]
                lengths.append(torch.abs(x1 - x2))
        xdists = []
        for i in range(len(levels)):
            for j,k in combinations(range(len(levels[i])), 2):
                x1 = xs[i][j]
                x2 = xs[i][k]
                if torch.abs(x1 - x2) < spacing:
                    xdists.append(spacing-torch.abs(x1 - x2))
                # print(xdists)
        xdist_loss = torch.mean(torch.stack(xdists))
        path_loss = torch.mean(torch.stack(lengths)) 
        if step==0:
            print(xdist_loss.item(), path_loss.item())
        loss = path_loss + xdist_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f'Loss: {loss.item()}')
    optimized_levels = [{} for i in range(len(levels))]
    for i,level in enumerate(levels):
        for j,pid in enumerate(level):
            optimized_levels[i][pid] = xs[i][j].item()
    print(xdist_loss.item(), path_loss.item())

    return optimized_levels
    
#%%
from mip import Model, MINIMIZE, BINARY
# Based on https://doi.org/10.1109/PacificVis.2018.00025
def optimize_placement(levels):
    model = Model(sense=MINIMIZE)
    above = [{} for level in levels]

    # Decision variables
    for j, level in enumerate(levels):
        combos = list(combinations_with_replacement(level, r=2))
        for i, (pid1, pid2) in enumerate(combos):
            # Add a binary variable for each pair of nodes, 
            # indicating whether pid1 is above pid2
            above[j][f'x_{pid1}_{pid2}'] = model.add_var(var_type=BINARY)
            above[j][f'x_{pid2}_{pid1}'] = model.add_var(var_type=BINARY)
            # Add the constraint that one of the nodes must be above the other
            model.add_constr(above[j][f'x_{pid1}_{pid2}'] + above[j][f'x_{pid2}_{pid1}'] == 1)
    for j, level in enumerate(levels):
        combos = list(combinations(level, 3))
        for i, (pid1, pid2, pid3) in enumerate(combos):
            # Add a second order transitivity constraint, 
            # so that if pid1 is above pid2, and pid2 is above pid3,
            # then pid1 is above pid3
            model.add_constr(above[j][f'x_{pid3}_{pid1}'] >= above[j][f'x_{pid3}_{pid2}'] + above[j][f'x_{pid2}_{pid1}'] - 1)
    # Convert paths to individual links
    links = [[] for i in range(len(levels)-1)]
    for path in paths:
        path = path[::-1]
        for i in range(len(path)-1):
            pid1 = path[i]
            pid2 = path[i+1]
            links[i].append((pid1, pid2))
    # Create variables indicating whether two links are crossing
    crossings = [{} for i in range(len(links))]
    for i, level in enumerate(links):
        combos = list(combinations(level, 2))
        for j, (link1, link2) in enumerate(combos):
            u1, v1 = link1
            u2, v2 = link2
            crossings[i][f'c_({u1}_{v1})_({u2}_{v2})'] = model.add_var(var_type=BINARY)
            # Add constraint that activates the crossing variable if the links are crossing
            model.add_constr(above[i][f'x_{u2}_{u1}'] + above[i+1][f'x_{v1}_{v2}'] + crossings[i][f'c_({u1}_{v1})_({u2}_{v2})'] >= 1)
            model.add_constr(above[i][f'x_{u1}_{u2}'] + above[i+1][f'x_{v2}_{v1}'] + crossings[i][f'c_({u1}_{v1})_({u2}_{v2})'] >= 1)

    for level in crossings:
        for var in level.values():
            model.objective += var
    status = model.optimize(max_seconds=60)
    ordering = []
    for i,level in enumerate(above):
        above_counts = {k:0 for k in levels[i].keys()}
        for var, value in level.items():
            n1, n2 = var.split('_')[1:]
            v = int(value.x)
            above_counts[n1] += v
        sorted_counts = sorted(above_counts.items(), key=lambda x: x[1], reverse=True)
        ordering.append({pid: rank for rank,(pid,count) in enumerate(sorted_counts)})
    return ordering
#%%
optimized_placement = optimize_placement(levels)
plot_paths(optimized_placement, paths, center=True)
# %%
