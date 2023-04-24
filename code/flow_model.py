from torch.nn import Linear, LeakyReLU
from torch import FloatTensor
import torch
import networkx as nx

# Generic Dense Multi-Layer Perceptron (MLP), which is just a stack of linear layers with ReLU activations
# input_dim: dimension of input
# output_dim: dimension of output
# hidden_dim: dimension of hidden layers
# num_layers: number of hidden layers
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, input_bias=True):
        super(MLP, self).__init__()
        layers = []
        # First layer has same number of inputs and outputs for interpretability of the input weights
        layers.append(Linear(input_dim, hidden_dim, bias=input_bias))
        layers.append(LeakyReLU())
        for i in range(num_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim))
            layers.append(LeakyReLU())
            # TODO do we need batch norm here?
        layers.append(Linear(hidden_dim, output_dim, bias=False))
        layers.append(LeakyReLU())
        # Register the layers as a module of the model
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        x = self.layers[0](x)

        for layer in self.layers[1:]:
            x = layer(x)

        return x
        
# Subclass of the MLP that has a separate L1 multiplier for each node in the graph
class L1_MLP(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(L1_MLP, self).__init__(input_dim, output_dim, hidden_dim, num_layers)
        # Create a Parameter of size (1, input_dim) that will be multiplied by the first layer
        # to regularize the input weights
        l1 = torch.randn(1, hidden_dim)
        self.l1 = torch.nn.Parameter(l1, requires_grad=True)
        l1_multiplier = self.l1.repeat(input_dim,1).T
        self.l1_multiplier = torch.nn.Parameter(l1_multiplier, requires_grad=True)
    
    def forward(self, x):
        # Multiply the first layer by the l1 multiplier
        x = ((self.layers[0].weight * self.l1_multiplier) @ x.T).T
        
        for layer in self.layers[1:]:
            x = layer(x)
        
        return x
    

# FlowModel is a neural ODE that takes in a state and outputs a delta
class ConnectedFlowModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(ConnectedFlowModel, self).__init__()
        self.model = MLP(input_dim, output_dim, hidden_dim, num_layers)
        # self.neural_ode = NeuralODE(self.model, sensitivity='adjoint')  

    def forward(self, state):
        # state is a tensor of shape (num_nodes, num_states)
        delta = self.model(state)
        return delta
    
class L1FlowModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(L1FlowModel, self).__init__()
        self.models = []
        # Create a separate MLP for each node in the graph
        for i in range(input_dim):
            node = L1_MLP(input_dim, 1, hidden_dim, num_layers)
            self.models.append(node)
        self.models = torch.nn.ModuleList(self.models)

    def forward(self, state):
        delta = torch.zeros_like(state)
        # For each node, run the corresponding MLP and insert the output into the delta tensor
        for i,model in enumerate(self.models):
            # state is a tensor of shape (num_nodes, num_states)
            delta[:,i] = model(state).squeeze()
            # Add the weights of the first layer to a list
        # Get the weights of the first layer
        return delta

class GraphFlowModel(torch.nn.Module):
    def __init__(self, num_layers, graph, data_idxs):
        super(GraphFlowModel, self).__init__()
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

