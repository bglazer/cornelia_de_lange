#%%
import matplotlib.pyplot as plt
import pickle
import numpy as np
from itertools import combinations, combinations_with_replacement
from mip import Model, MINIMIZE, BINARY, xsum

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
def plot_paths(levels, paths, center=False):
    fig, ax = plt.subplots(figsize=(10,10))
    xy = {}
    for y,level in enumerate(levels):
        ys = [y for _ in range(len(level))]
        xs = level.values()
        if center:
            xs = [x - widths[y]/2 for x in xs]
        xy.update({pid:(x,y) for pid,x in zip(level.keys(), xs)})
        
    colors= plt.cm.tab20(np.linspace(0,1,len(paths)))
    starts = {path[0]:0 for path in paths}
    for j, path in enumerate(paths):
        # Color the starting node the same color as the path
        if len(path) > 1:
            pid0 = path[0]
            ax.scatter(xy[pid0][0], xy[pid0][1], s=300-starts[pid0]*90, c=colors[j], zorder=starts[pid0])
            print('starts', protein_id_name[pid0], 300-starts[pid0]*90)
            starts[pid0] += 1
        path = path[::-1]
        for i in range(len(path)-1):
            pid1 = path[i]
            pid2 = path[i+1]
            x1,y1 = xy[pid1]
            x2,y2 = xy[pid2]
            ax.plot([x1,x2],[y1,y2], c=colors[j])
        for i in range(0,len(path)-1):
            x,y = xy[path[i]]
            ax.scatter(x, y, s=100, c='grey')

    for pid,(x,y) in xy.items():
        name = protein_id_name[pid]
        ax.text(x-len(name)/15, y+.05, name, fontsize=10, fontdict={'family':'monospace'})
    ax.set_xticks([])
    ax.set_yticks([]);

# Based on https://doi.org/10.1109/PacificVis.2018.00025
def optimize_placement(levels, max_count=None, verbose=False):
    model = Model(sense=MINIMIZE)
    above = [{} for level in levels]

    if verbose:
        print('Adding variables', flush=True)
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
    crossings = {}
    for i, level in enumerate(links):
        combos = list(combinations(level, 2))
        for j, (link1, link2) in enumerate(combos):
            u1, v1 = link1
            u2, v2 = link2
            crossings[f'c_({u1}_{v1})_({u2}_{v2})'] = model.add_var(var_type=BINARY)
            # Add constraint that activates the crossing variable if the links are crossing
            model.add_constr(above[i][f'x_{u2}_{u1}'] + above[i+1][f'x_{v1}_{v2}'] + crossings[f'c_({u1}_{v1})_({u2}_{v2})'] >= 1)
            model.add_constr(above[i][f'x_{u1}_{u2}'] + above[i+1][f'x_{v2}_{v1}'] + crossings[f'c_({u1}_{v1})_({u2}_{v2})'] >= 1)
            if f'c_({u1}_{v1})_({u2}_{v2})' in crossings and f'c_({u1}_{v2})_({u2}_{v1})' in crossings:
                model.add_constr(crossings[f'c_({u1}_{v1})_({u2}_{v2})'] + crossings[f'c_({u1}_{v2})_({u2}_{v1})'] == 1)
    # Add objective function
    if verbose:
        print('Adding objective')
    model.objective = xsum(list(crossings.values()))
    # Optimize
    if verbose:
        print('Optimizing', flush=True)
    if max_count is None:
        status = model.optimize()
    else:
        status = model.optimize(max_seconds=max_count)
    if verbose:
        print(status)
    # Sort the nodes by their rank, i.e. how many other nodes they're above
    # this gives the overall ordering of the nodes that minimizes crossings
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
optimized_placement = optimize_placement(levels, max_count=None, verbose=True)
#%%
plot_paths(optimized_placement, paths, center=True)
# %%
