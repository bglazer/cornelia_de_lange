from experiment import Experiment
import pickle

g = pickle.load(open('../../data/filtered_graph.pickle', 'rb'))
e = Experiment(None, None, g, None)
actions = e.allActions()
for action in actions:
    print(action)

print(len(actions))

model, actions = e.initialState()
