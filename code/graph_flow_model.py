from torch.nn import Linear, Sequential, ReLU, LeakyReLU
import torch
import networkx as nx
from torchdyn.core import NeuralODE

# Generic Dense Multi-Layer Perceptron (MLP), which is just a stack of linear layers with ReLU activations
# input_dim: dimension of input
# output_dim: dimension of output
# hidden_dim: dimension of hidden layers
# num_layers: number of hidden layers
def MLP(input_dim, output_dim, hidden_dim, num_layers):
    layers = []
    layers.append(Linear(input_dim, hidden_dim))
    layers.append(LeakyReLU())
    for i in range(num_layers - 1):
        layers.append(Linear(hidden_dim, hidden_dim))
        layers.append(LeakyReLU())
        # TODO do we need batch norm here?
    layers.append(Linear(hidden_dim, output_dim, bias=False))
    layers.append(LeakyReLU())
    return Sequential(*layers)

# FlowModel is a neural ODE that takes in a state and outputs a delta
class FlowModel(torch.nn.Module):
    def __init__(self, num_layers, graph, data_idxs):
        super(FlowModel, self).__init__()
        # Make an MLP for each node in the networkx graph
        # with one input to the MLP for each incoming edge and a 
        self.graph = graph
        self.data_idxs = data_idxs
        self.models = {}
        # self.models = torch.nn.ModuleList(self.models)

        # Remove nodes that are not in the input data
        non_data_nodes = []
        for node in graph.nodes():
            if node not in data_idxs:
                print(f'Error: node {node} is in the graph but it is not in the input data')
                non_data_nodes.append(node)
        graph.remove_nodes_from(non_data_nodes)

        # Remove nodes that have no incoming edges
        removed = True
        while removed:
            zero_indegree_nodes = []
            removed = False
            for node, indegree in graph.in_degree():
                if indegree == 0:
                    print(f'Error: node {node} has no incoming edges')
                    zero_indegree_nodes.append(node)
                    removed = True
            graph.remove_nodes_from(zero_indegree_nodes)

        # Add self loops to the graph
        for node in graph.nodes():
            if not graph.has_edge(node, node):
                graph.add_edge(node, node)
        
        # Convert the graph into a list of indexes
        self.adj_list = {}
        non_data_nodes = []
        for node in self.graph.nodes():
            adj = []
            for edge in self.graph.in_edges(node):
                src = edge[0]
                if src in self.data_idxs:
                    adj.append(data_idxs[src])
                else:
                    raise Exception(f'Error: node {src} is in the graph but it is not in the input data')
            self.adj_list[node] = adj
        
        for node in self.adj_list:
            input_dim = len(graph.in_edges(node))
            # TODO make this a parameter
            # minimum hidden dimension is 10
            hidden_dim = max(10, input_dim*2)
            model = MLP(input_dim, 1, hidden_dim, num_layers)
            self.models[node] = model

        self.models = torch.nn.ModuleDict(self.models)
        
    def forward(self, state):
        # Apply each model to the portion of the state vector that
        #  corresponds to the inputs to the node in the graph
        output = torch.zeros(state.shape, device=state.device)
        for node, model in self.models.items():
            inputs = state[:,self.adj_list[node]]
            delta = model(inputs)
            output[:,self.data_idxs[node]] = delta[:,0]

        return output

    # def trajectory(self, state, tspan):
    #     # state is a tensor of shape (num_nodes, num_states)
    #     # tspan is a tensor of shape (num_timesteps,)
    #     trajectory = self.neural_ode.trajectory(state, tspan)
    #     return trajectory