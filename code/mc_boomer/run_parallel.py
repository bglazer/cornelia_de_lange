from mc_boomer.mcts import explorationDecay, constantExploration
import json
from datetime import datetime
import random
import run
import initialize
import pickle
from joblib import Parallel, delayed

num_steps = int(5e4)
num_processes = 100
n_parallel = 100
#num_steps = int(5e4)
explorationPolicy = explorationDecay
# Number of nodes in the graph
min_edges = 119
# Total number of edges in the graph
max_edges = 248
explorationConstant = tuple([random.uniform(.2,.5) for i in range(num_processes)])
stop_prior = tuple([random.uniform(.6,.9) for i in range(num_processes)])
rave_equiv_param = tuple([random.randint(1000,500000) for i in range(num_processes)])
#rave_equiv_param = tuple([None for i in range(num_processes)])
keep_tree = True
cache = False
nested = True
threshold = .95

parameters = {
    'num_steps': num_steps,
    'explorationConstant': explorationConstant,
    'num_processes': num_processes,
    'min_edges': min_edges,
    'max_edges': max_edges,
    'explorationPolicy': explorationPolicy.__name__,
    'stop_prior': stop_prior,
    'rave_equiv_param': rave_equiv_param,
    'keep_tree': keep_tree,
    'cache': cache,
    'nested': nested,
    'experiments': True,
    'reward':'avg',
    'threshold':threshold
}

timestamp = datetime.strftime(datetime.now(), format='%Y%m%d-%H%M')
data_dir = '../../data'
output_dir = '../../output/mc_boomer'

datafile = f'{data_dir}/tiana_etal_differential_expression.csv'
graph = pickle.load(open(f'{data_dir}/filtered_graph.pickle','rb'))

#n_actions = random.randint(min_edges-50, min_edges+50)
#model, actions = initialize.randomModel(actions, graph, n_actions)
actions = initialize.allActions(graph)
model = initialize.emptyModel(graph)
experiments = initialize.experiments(datafile, graph)

def start(job):
    parameter_file = open(f'{output_dir}/parameters/parameters-job-{job}.json','w')
    json.dump(parameters, parameter_file)
    parameter_file.close()

    output_string = f'job-{job}_tmstp-{timestamp}'
    run.search(actions,
               model,
               experiments,
               stop_prior = stop_prior[job],
               min_edges = min_edges,
               max_edges = max_edges,
               num_steps = num_steps,
               explorationConstant = explorationConstant[job],
               explorationPolicy = explorationPolicy,
               rave_equiv_param = rave_equiv_param[job],
               keep_tree = keep_tree,
               cache = cache,
               nested = nested,
               threshold=threshold,
               stats_file = None, #f'{output_dir}/stats/search_stats-{output_string}.csv',
               model_file = f'{output_dir}/models/models-{output_string}.txt.gz',
               run_file = f'{output_dir}/logs/run_log-{output_string}.log')

parallel = Parallel(n_jobs = n_parallel)
parallel(delayed(start)(i) for i in range(num_processes))
