from mc_boomer.action import Action, Source
from mc_boomer.boolean_model import BooleanModel

class Experiment():
    def __init__(self, start_states, end_states, conditions, random_starts=None):
        if start_states != None and random_starts != None:
            raise Exception('Should only provide either start_states and random_starts, but not both')
        elif start_states is None and random_starts is None:
            raise Exception('Should only provide either start_states and random_starts, both were None')
        self.start_states = start_states
        self.end_states = end_states
        self.conditions = conditions
        self.random_starts = random_starts
