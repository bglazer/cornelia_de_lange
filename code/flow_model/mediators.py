import pickle
from scipy.stats import hypergeom
import networkx as nx
from tqdm import tqdm
import scipy
import numpy as np
from itertools import product

protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v: k for k, v in node_to_idx.items()}

# Calculate the percentage of shortest paths that a mediator appears in for each knockout gene
def count_mediators(all_paths):
    mediators = {}
    for path in all_paths:
        source = path[0]
        target = path[-1]
        for mediator in path[1:-1]:
            if mediator not in mediators:
                mediators[mediator] = set()
            mediators[mediator].add((source, target))

    return mediators

def mediator_probability(model_shortest_paths, all_shortest_paths, verbose=False):
    model_mediators = count_mediators(model_shortest_paths)
    num_pairs = len(set([(path[0], path[-1]) for path in model_shortest_paths]))
    all_mediators = count_mediators(all_shortest_paths)
    mediator_probs = {}
    for mediator in model_mediators:
        # Population size - Total possible number of shortest paths
        M = len(node_to_idx)**2
        # Hits in population - Number of times we see this mediator in a shortest path between any two nodes in the knowledge graph
        n = len(all_mediators[mediator])
        # Sample size - Number of node pairs in the model
        N = num_pairs
        # Hits in sample - number of times we see this mediator in a shortest path between two nodes in our model
        k = len(model_mediators[mediator])
        # Probability of observing k or more matches 
        p = 1-hypergeom.cdf(k-1, M, n, N)
        mediator_probs[mediator] = p
        if verbose:
            print(protein_id_name[mediator], p)
            print('M=',M)
            print('n=',n)
            print('N=',N)
            print('k=',k)
    mediator_probs = {k: v for k, v in sorted(mediator_probs.items(), key=lambda item: item[1])}
    return mediator_probs, model_mediators

def find_mediators(target_active_inputs, knowledge_graph, threshold=0.01, all_shortest_paths=None, verbose=False):
    model_shortest_paths = []
    no_paths = 0
    undirected_knowledge_graph = knowledge_graph.to_undirected()

    for target, active_inputs in target_active_inputs.items():
        for active_input in active_inputs:
            if active_input in knowledge_graph:
                try:
                    shortest_paths = list(nx.all_shortest_paths(undirected_knowledge_graph, active_input, target))
                except:
                    if verbose: print(f'No path {protein_id_name[active_input]} ({active_input}) '
                                      f'-> {protein_id_name[target]} ({target})')
                    no_paths += 1
                    continue
                for path in shortest_paths:
                    model_shortest_paths.append(path)

    if verbose: print(f'Number of of source-target pairs with no path in knowledge graph: {no_paths}')

    if all_shortest_paths is None:
        all_shortest_paths = []
        for target in tqdm(node_to_idx, disable=not verbose):
            for source in node_to_idx:
                try:
                    paths = list(nx.all_shortest_paths(undirected_knowledge_graph, source, target))
                    all_shortest_paths.extend(paths)
                except nx.NetworkXNoPath:
                    if verbose: print(f'No path {source} -> {target}')
            
    mediator_probs, model_mediators = mediator_probability(model_shortest_paths, 
                                                           all_shortest_paths, 
                                                           verbose=verbose)
    
    if verbose:
        for mediator, prob in mediator_probs.items():
            if prob < threshold:
                print(protein_id_name[mediator], prob)
                for interaction in sorted(model_mediators[mediator]):
                    source, target = interaction
                    print(f'    {protein_id_name[source]} -> {protein_id_name[target]}')

    significant_mediators = {mediator: prob for mediator, prob in mediator_probs.items() if prob < threshold}
    significant_interactions = {mediator: model_mediators[mediator] for mediator in significant_mediators}
    return significant_mediators, significant_interactions, all_shortest_paths

def count_bridges(node_pairs, D):
    bridge_targets = {}
    for pair in node_pairs:
        src, tgt = pair
        if src in node_to_idx and tgt in node_to_idx:
            src_idx = node_to_idx[src]
            tgt_idx = node_to_idx[tgt]
            # Combined distance from the bridge to the source and target
            s = D[:,src_idx] + D[:,tgt_idx]
            min_idxs = np.where(s == s.min())[0]
            for min_idx in min_idxs:
                if min_idx not in idx_to_node:
                    continue
                if s.min() == np.inf:
                    continue
                if min_idx == src_idx or min_idx == tgt_idx:
                    continue
                # If the bridge length is greater than the direct shortest path
                #  between the source and target, then it is not a bridge
                if s.min() > D[src_idx, tgt_idx]:
                    continue
                bridge_node = idx_to_node[min_idx]
                if bridge_node not in bridge_targets:
                    bridge_targets[bridge_node] = []
                bridge_targets[bridge_node].append((src, tgt))
    return bridge_targets

def find_bridges(target_active_inputs, knowledge_graph, threshold=0.01, verbose=False):
    sorted_nodes = sorted(node_to_idx, key=lambda x: node_to_idx[x])
    
    g = nx.DiGraph()
    g.add_nodes_from(knowledge_graph.nodes)
    g.add_edges_from(knowledge_graph.edges)
    sg = nx.to_scipy_sparse_array(g, nodelist=sorted_nodes)
    D = scipy.sparse.csgraph.shortest_path(sg, directed=True, unweighted=True)

    node_pairs = []
    for target, active_inputs in target_active_inputs.items():
        tgt_idx = node_to_idx[target]
        for active_input in active_inputs:
            node_pairs.append((active_input, target))
    model_bridges = count_bridges(node_pairs, D)    

    all_node_pairs = product(sorted_nodes, sorted_nodes)
    all_bridges = count_bridges(all_node_pairs, D)    
    
    num_pairs = 0
    for target, active_inputs in target_active_inputs.items():
        num_pairs += len(active_inputs)
        
    bridge_probs = {}
    for bridge in model_bridges:
        # Population size - Total possible number of shortest paths
        M = len(node_to_idx)**2
        # Hits in population - Number of times we see this bridge in a shortest path between any two nodes in the knowledge graph
        n = len(all_bridges[bridge])
        # Sample size - Number of node pairs in the model
        N = num_pairs
        # Hits in sample - number of times we see this bridge in a shortest path between two nodes in our model
        k = len(model_bridges[bridge])
        # Probability of observing k or more matches 
        p = 1-hypergeom.cdf(k-1, M, n, N)
        bridge_probs[bridge] = p
        if verbose:
            print(protein_id_name[bridge], p)
            print('M=',M)
            print('n=',n)
            print('N=',N)
            print('k=',k)
    bridge_probs = {k: v 
                    for k, v 
                    in sorted(bridge_probs.items(), key=lambda item: item[1])
                    if v < threshold}
    model_bridges = {k: v for k, v in model_bridges.items() if k in bridge_probs}

    return bridge_probs, model_bridges
    