from itertools import permutations, product
from segment_polarity_model import SegmentPolarityModel
from mc_boomer.action import Action, Source
from mc_boomer.boolean_model import BooleanModel
# TODO if this works, make it part of mc_boomer module
from experiment import Experiment
import random
from mc_boomer.search_state import SearchState

def emptyModel(graph):
    # TODO make this part of BooleanModel init
    rules = {}
    for node in graph.nodes:
        rules[node] = ([],[])
    return BooleanModel(rules)

def randomModel(actions, graph, n_actions):
    # TODO add "remove/delete" actions to the action list
    # TODO this would require modifying takeAction as well
    model = emptyModel(graph)
    state = SearchState(model, 
                        None,
                        # TODO how does stop prior interact with first step of search process?
                        0.0,
                        min_edges=0,
                        max_edges=len(model.nodes),
                        actions=actions)
    for i in range(n_actions):
        action,prior = random.choice(state.getPossibleActions())
        state = state.takeAction(action)
    return state.model, state.actions
    

def allActions(graph):
    actions = {}

    for src, dst, data in graph.edges(data=True):
        if data['effect'] is None:
            effects = ['a','i']
        else:
            effects = [data['effect']]

        # Create edges in both directions, activating and inhibiting
        # This comes from STRINGdb evidence of PPI interactions
        # So, we don't know the exact nature of the interaction, or the direction
        prior = data['weight']
        for effect in effects:
            action = Action((Source(src),), dst, effect)
            actions[action] = prior
            if not data['directed']:
                action = Action((Source(dst),), src, effect)
                actions[action] = prior
            
    # calculate the mean weight of the STRINGdb interactions
    # We use this as the prior for the TF interactions, because we don't have any 
    # predefined prior information for them. 
    # TODO another option is to just give everything a 1.0 prior, then normalize
    weight = 0.0
    weightcount = 0
    for src, dst, data in graph.edges(data=True):
        if data['type'] == 'protein_protein':
            weight += data['weight']
            weightcount += 1
    meanweight = weight/weightcount            
    for action, prior in actions.items():
        if prior is None:
            actions[action] = meanweight

    return actions

def experiments(datafile, graph):
    experiments = []
    start_state = {}
    end_state = []

    with open(datafile) as data:
        for line in data:        
            gene, end = line.strip().split(',')
            end = bool(end)
            start = not end 
            if gene in graph.nodes:
                start_state[gene] = start
                end_state.append((gene,end))
    start_states = [start_state]
    end_states = {(tuple(end_state),):1}
    experiment = Experiment(start_states, end_states, None)
    experiments.append(experiment)
    return experiments

