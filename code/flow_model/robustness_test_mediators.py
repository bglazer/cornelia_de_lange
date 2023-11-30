#%%
# %load_ext autoreload
# %autoreload 2
#%%
import pickle
import numpy as np
import pickle
import torch
import numpy as np
from mediators import find_mediators, find_bridges
import random
from joblib import Parallel, delayed
import os
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# %%
# genotype = 'wildtype'
# tmstp = '20230607_165324'
genotype = 'mutant'
tmstp = '20230608_093734'
outdir = f'../../output/{tmstp}'
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
all_genes = set(node_to_idx.keys())
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
name_protein_id = {name: id for id, name in protein_id_name.items()}
graph = pickle.load(open(f'../../data/filtered_graph.pickle', 'rb'))

#%%
# Load the target input list
with open(f'{outdir}/optimal_{genotype}_active_inputs.pickle', 'rb') as f:
    target_active_genes = pickle.load(f)
#%%
# Add edges randomly to the graph
def add_random_edges(graph, percent_edges):
    graph = graph.copy()
    # Get the number of edges to add
    num_edges = int(len(graph.edges())*percent_edges)
    # Get the edges to add
    for i in range(num_edges):
        src = random.choice(list(graph.nodes()))
        dst = random.choice(list(graph.nodes()))
        # Don't add self loops
        while src == dst or graph.has_edge(src, dst):
            dst = random.choice(list(graph.nodes()))
        graph.add_edge(src, dst)
    return graph

def remove_random_edges(graph, percent_edges):
    graph = graph.copy()
    # Get the number of edges to remove
    num_edges = int(len(graph.edges())*percent_edges)
    # Get the edges to remove
    edges = list(graph.edges())
    for i in range(num_edges):
        edge = random.choice(edges)
        graph.remove_edge(edge[0], edge[1])
        edges.remove(edge)
    return graph
#%%
def mediators_with_noise(graph, target_active_genes, percent_edges, iteration, verbose=False):
    mediated_with_addition = []
    mediated_with_subtraction = []
    mediated_with_both = []
    
    for pct in percent_edges:
        print(f'Iteration {iteration+1}, percent edges: {pct}', flush=True)
        pct = pct/100
        # Find the mediators with a graph with added edges
        graph_with_addition = add_random_edges(graph, pct)
        results = find_bridges(target_active_genes,
                               knowledge_graph=graph_with_addition,
                              #  all_shortest_paths=None,
                               verbose=verbose,
                               threshold=0.01)
        mediator_probs, mediated_interactions = results #, all_shortest_paths = results
        mediated_with_addition.append(mediated_interactions)

        # Find the mediators with a graph with subtracted edges
        graph_with_subtraction = remove_random_edges(graph, pct)
        results = find_bridges(target_active_genes,
                               knowledge_graph=graph_with_subtraction,
                              #  all_shortest_paths=None,
                               verbose=verbose,
                               threshold=0.01)
        mediator_probs, mediated_interactions = results #, all_shortest_paths = results
        mediated_with_subtraction.append(mediated_interactions)

        # Find the mediators with a graph with both added and subtracted edges in equal proportions
        graph_with_both = add_random_edges(graph, pct)
        graph_with_both = remove_random_edges(graph_with_both, pct)        
        results = find_bridges(target_active_genes,
                               knowledge_graph=graph_with_both,
                              #  all_shortest_paths=None,
                               verbose=verbose,
                               threshold=0.01)
        mediator_probs, mediated_interactions = results #, all_shortest_paths = results
        mediated_with_both.append(mediated_interactions)

    return mediated_with_addition, mediated_with_subtraction, mediated_with_both

#%%
percent_edges = np.arange(0, 101, 5, dtype=int)
n_repeats = 100
#%%
# mediators_with_noise(graph, target_active_genes, percent_edges, 0)
#%%
parallel = Parallel(n_jobs=40, verbose=10)
results = parallel(delayed(mediators_with_noise)(graph, target_active_genes, percent_edges, i) for i in range(n_repeats))
# %%
with open(f'{outdir}/mediators_with_noise_{genotype}.pickle', 'wb') as f:
    pickle.dump(results, f)
#%%
with open(f'{outdir}/mediators_with_noise_{genotype}.pickle', 'rb') as f:
    results = pickle.load(f)

# %%
print('Robustness of mediators to noise')
from collections import namedtuple
mediated = namedtuple('mediated', ('mediator', 'src', 'dst'))
zero_noise_mediators = set()
for repeat in range(len(results)):
    for mediator, mediated_interactions in results[repeat][0][0].items():
        # for interaction in mediated_interactions:
        # zero_noise_mediators.add(mediated(mediator, interaction[0], interaction[1]))
        zero_noise_mediators.add(mediator)
zero_noise_mediators = set(zero_noise_mediators)

# k is the index of the return value from mediators_with_noise
# i.e. with_addition=0, with_subtraction=1, with_both=2
# TODO fix this so that mediators_with_noise returns a dictionary

def count_mediator_appearances_with_noise(results, k):
    mediators_at_percent_edges = {mediator: {pct: 0 for pct in percent_edges[1:]} 
                                  for mediator in zero_noise_mediators}
    for i, pct in enumerate(percent_edges[1:]):
        for repeat in range(len(results)):
            for mediator, mediated_interactions in results[repeat][k][i].items():
                # for interaction in mediated_interactions:
                # m = mediated(mediator, interaction[0], interaction[1])
                if mediator in zero_noise_mediators:
                    mediators_at_percent_edges[mediator][pct]+=1
                        
    return mediators_at_percent_edges

noisy_mediators_addition = count_mediator_appearances_with_noise(results, 0)
noisy_mediators_subtraction = count_mediator_appearances_with_noise(results, 1)
noisy_mediators_both = count_mediator_appearances_with_noise(results, 2)

mediator_noise_auc = {mediator: {'addition':0, 
                                 'subtraction':0,
                                 'both':0}
                      for mediator in zero_noise_mediators}
pct_threshold = 0.5
theshold = n_repeats*pct_threshold
mediator_noise_threshold = {mediator: {'addition':None,
                                        'subtraction':None,
                                        'both':None}
                             for mediator in zero_noise_mediators}


n_pcts = len(percent_edges[1:])
for mediator in zero_noise_mediators:
    for pct in percent_edges[1:]:
        num_occurrences = noisy_mediators_addition[mediator][pct]
        mediator_noise_auc[mediator]['addition'] += num_occurrences/n_pcts
        if num_occurrences < theshold and mediator_noise_threshold[mediator]['addition'] is None:
            mediator_noise_threshold[mediator]['addition'] = pct 

    for pct in percent_edges[1:]:
        num_occurrences = noisy_mediators_subtraction[mediator][pct]
        mediator_noise_auc[mediator]['subtraction'] += noisy_mediators_subtraction[mediator][pct]/n_pcts
        if num_occurrences < theshold and mediator_noise_threshold[mediator]['subtraction'] is None:
            mediator_noise_threshold[mediator]['subtraction'] = pct 
        
    for pct in percent_edges[1:]:
        num_occurrences = noisy_mediators_both[mediator][pct]
        mediator_noise_auc[mediator]['both'] += noisy_mediators_both[mediator][pct]/n_pcts
        if num_occurrences < theshold and mediator_noise_threshold[mediator]['both'] is None:
            mediator_noise_threshold[mediator]['both'] = pct 
# Sort by average auc across all noise types
def key(mediator):
    return (mediator_noise_auc[mediator]['both'] + \
            mediator_noise_auc[mediator]['addition'] + \
            mediator_noise_auc[mediator]['subtraction'])/3

# for interaction in sorted(mediator_noise_auc, key=lambda x: key(x), reverse=True):
for mediator in sorted(mediator_noise_auc, key=lambda x: key(x), reverse=True):
    # mediator, src, dst = interaction
    na=f'>{percent_edges[-1]}'
    print(mediator)
    print(f'mediator={protein_id_name[mediator]:10s} '
        #   f'{protein_id_name[src]+"->"+protein_id_name[dst]:18s} '
          f'AUC      : {" ".join([f"{noise_type}={auc:.1f}" for noise_type, auc in mediator_noise_auc[mediator].items()])}')
    print(f'mediator={protein_id_name[mediator]:10s} '
        #   f'{protein_id_name[src]+"->"+protein_id_name[dst]:18s} '
          f'threshold: {" ".join([f"{noise_type}={str(threshold) if threshold else na}" for noise_type, threshold in mediator_noise_threshold[mediator].items()])}')
    print('-')
#%%
print('Mediators with all AUCs > 50')
for mediator in mediator_noise_auc:
    if mediator_noise_auc[mediator]['both'] > 50 and mediator_noise_auc[mediator]['addition'] > 50 and mediator_noise_auc[mediator]['subtraction'] > 50:
        print(protein_id_name[mediator], mediator_noise_auc[mediator])

#%%
print('Mediators with all thresholds >= 50')
robust_mediators = []
for mediator in mediator_noise_threshold:
    both = mediator_noise_threshold[mediator]['both']
    addition = mediator_noise_threshold[mediator]['addition']
    subtraction = mediator_noise_threshold[mediator]['subtraction']
    both = both if both else 100
    addition = addition if addition else 100
    subtraction = subtraction if subtraction else 100
    if both >= 50 and addition >= 50 and subtraction >= 50:
        print(protein_id_name[mediator])
        robust_mediators.append(mediator)
pickle.dump(robust_mediators, open(f'{outdir}/robust_mediators_{genotype}.pickle', 'wb'))

# %%
print('Robustness of individual mediator interactions to noise')
from collections import namedtuple
mediated = namedtuple('mediated', ('mediator', 'src', 'dst'))
zero_noise_mediators = set()
for repeat in range(n_repeats):
    for mediator, mediated_interactions in results[repeat][0][0].items():
        for interaction in mediated_interactions:
            zero_noise_mediators.add(mediated(mediator, interaction[0], interaction[1]))
zero_noise_mediators = set(zero_noise_mediators)

# k is the index of the return value from mediators_with_noise
# i.e. with_addition=0, with_subtraction=1, with_both=2
# TODO fix this so that mediators_with_noise returns a dictionary

def count_mediator_appearances_with_noise(results, k):
    mediators_at_percent_edges = {mediator: {pct: 0 for pct in percent_edges[1:]} 
                                  for mediator in zero_noise_mediators}
    for i, pct in enumerate(percent_edges[1:]):
        for repeat in range(n_repeats):
            for mediator, mediated_interactions in results[repeat][k][i].items():
                for interaction in mediated_interactions:
                    m = mediated(mediator, interaction[0], interaction[1])
                    if m in zero_noise_mediators:
                        mediators_at_percent_edges[m][pct]+=1
                        
    return mediators_at_percent_edges

noisy_mediators_addition = count_mediator_appearances_with_noise(results, 0)
noisy_mediators_subtraction = count_mediator_appearances_with_noise(results, 1)
noisy_mediators_both = count_mediator_appearances_with_noise(results, 2)

mediator_noise_auc = {mediator: {'addition':0, 
                                 'subtraction':0,
                                 'both':0}
                      for mediator in zero_noise_mediators}
theshold = n_repeats*pct_threshold
mediator_noise_threshold = {mediator: {'addition':None,
                                        'subtraction':None,
                                        'both':None}
                             for mediator in zero_noise_mediators}

n_pcts = len(percent_edges[1:])
for mediator in zero_noise_mediators:
    for pct in percent_edges[1:]:
        num_occurrences = noisy_mediators_addition[mediator][pct]
        mediator_noise_auc[mediator]['addition'] += num_occurrences/n_pcts
        if num_occurrences < theshold and mediator_noise_threshold[mediator]['addition'] is None:
            mediator_noise_threshold[mediator]['addition'] = pct 

    for pct in percent_edges[1:]:
        num_occurrences = noisy_mediators_subtraction[mediator][pct]
        mediator_noise_auc[mediator]['subtraction'] += noisy_mediators_subtraction[mediator][pct]/n_pcts
        if num_occurrences < theshold and mediator_noise_threshold[mediator]['subtraction'] is None:
            mediator_noise_threshold[mediator]['subtraction'] = pct 
        
    for pct in percent_edges[1:]:
        num_occurrences = noisy_mediators_both[mediator][pct]
        mediator_noise_auc[mediator]['both'] += noisy_mediators_both[mediator][pct]/n_pcts
        if num_occurrences < theshold and mediator_noise_threshold[mediator]['both'] is None:
            mediator_noise_threshold[mediator]['both'] = pct 
# Sort by average auc across all noise types
def key(mediator):
    return (mediator_noise_auc[mediator]['both'] + \
            mediator_noise_auc[mediator]['addition'] + \
            mediator_noise_auc[mediator]['subtraction'])/3
#%%
for interaction, auc in sorted(mediator_noise_auc.items(), key=lambda x: key(x[0]), reverse=True):
    mediator, src, dst = interaction
    if auc['both'] > 50 and auc['addition'] > 50 and auc['subtraction'] > 50:
        na=f'>{percent_edges[-1]}'
        print(mediator)
        print(f'mediator={protein_id_name[mediator]:10s} '
            f'{protein_id_name[src]+"->"+protein_id_name[dst]:18s} '
            f'AUC      : {" ".join([f"{noise_type}={auc:.1f}" for noise_type, auc in mediator_noise_auc[interaction].items()])}')
        print(f'mediator={protein_id_name[mediator]:10s} '
            f'{protein_id_name[src]+"->"+protein_id_name[dst]:18s} '
            f'threshold: {" ".join([f"{noise_type}={str(threshold) if threshold else na}" for noise_type, threshold in mediator_noise_threshold[interaction].items()])}')
        print('-')


# %%

# %%
# mediated_interactions = {}
# for mediator_interactions in sorted(zero_noise_mediators):
#     mediator, src, dst = mediator_interactions
#     mediated_interactions[mediator] = mediated_interactions.get(mediator, []) + [(src, dst)]
# for mediator, interactions in mediated_interactions.items():
#     for interaction in interactions:
#         src, dst = interaction
#         print(f'{protein_id_name[mediator]+":":8s}{protein_id_name[src]+"->"+protein_id_name[dst]}')
#     print('-'*80)
# %%
