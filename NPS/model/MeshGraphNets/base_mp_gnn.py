# Lint as: python3
# pytorch port
# ============================================================================
"""Core learned graph net model."""

import collections
import functools
import torch
import torch.nn as nn
# from torch_scatter import scatter
from NPS.model.common import MLP_


EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn_node, model_fn_edge, name='GraphNetBlock'):
        super(GraphNetBlock, self).__init__()
        self.node_fn = model_fn_node()
        self.edge_fn = model_fn_edge()

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        # sender_features = tf.gather(node_features, edge_set.senders)
        # receiver_features = tf.gather(node_features, edge_set.receivers)
        sender_features = node_features[edge_set.senders]
        receiver_features = node_features[edge_set.receivers]
        features = [sender_features, receiver_features, edge_set.features]
        return self.edge_fn(torch.cat(features, dim=-1))

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            # features.append(scatter_sum(edge_set.features, edge_set.receivers, 0, dim_size=num_nodes))
            features.append(torch.zeros((num_nodes,edge_set.features.shape[1]), dtype=edge_set.features.dtype, device=edge_set.features.device).scatter_add_(0,edge_set.receivers[:,None].expand(-1,edge_set.features.shape[-1]),edge_set.features))
        return self.node_fn(torch.cat(features, dim=-1))

    def forward(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""

        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features += graph.node_features
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
            input_size_node, input_size_edge,
            output_size,
            latent_size_node, latent_size_edge,
            num_layers,
            message_passing_steps,
            nlayer_mlp_encdec=-1,
            activation='relu',
            dropout=0,
            args=None,
            preprocess_edgeset=False,
            norm=1,
            n_edge_set = 1, **kwargs):
        super(EncodeProcessDecode, self).__init__()
        # print(f'debug input_size_node {input_size_node} input_size_edge {input_size_edge} {[ output_size,        latent_size_node, latent_size_edge, num_layers,message_passing_steps, activation, n_edge_set]} {kwargs}')
        self.input_size_node = input_size_node
        self.input_size_edge = input_size_edge
        self._output_size = output_size
        self._num_layers = num_layers
        self.nlayer_mlp_encdec = num_layers if nlayer_mlp_encdec==-1 else nlayer_mlp_encdec
        self._message_passing_steps = message_passing_steps
        self.activation = activation #my_activations[activation]
        self.norm = bool(norm)
        self.dropout = dropout
        self._latent_size_node = latent_size_node
        self._latent_size_edge = latent_size_edge
        self.n_edge_set = 1
        self.args = args
        self.preprocess_edgeset = preprocess_edgeset
        self.init_encoder_decoder()
        self.init_processor(latent_size_node, latent_size_edge)

    def init_processor(self, latent_size_node, latent_size_edge):
        model_fn_node = functools.partial(self._make_mlp, input_size=latent_size_node+  latent_size_edge, output_size=latent_size_node, nhid=latent_size_node, layer_norm=self.norm)
        model_fn_edge = functools.partial(self._make_mlp, input_size=latent_size_edge+2*latent_size_node, output_size=latent_size_edge, nhid=latent_size_edge, layer_norm=self.norm)
        message_passing_layers = [GraphNetBlock(model_fn_node, model_fn_edge) for _ in range(self._message_passing_steps)]
        self.message_passing = nn.Sequential(*message_passing_layers)

    def _make_mlp(self, input_size, output_size, nhid=None, layer_norm=True, nlayer=None, dropout_first=True):
        """Builds an MLP."""
        if nlayer is None:
            nlayer = self._num_layers
        widths = [self._latent_size if nhid is None else nhid] * nlayer
        network = MLP_(input_size, widths, output_size, activate_final=False, activation=self.activation, layer_norm=layer_norm, conv=False, dropout=self.dropout, dropout_first=dropout_first)
        return network

    def init_encoder_node(self):
        self.encoder_node = self._make_mlp(self.input_size_node, self._latent_size_node, self._latent_size_node, nlayer=self.nlayer_mlp_encdec, dropout_first=False, layer_norm=self.norm)

    def init_encoder_decoder(self):
        encoders_edge =    [self._make_mlp(self.input_size_edge, self._latent_size_edge, self._latent_size_edge, dropout_first=False, layer_norm=self.norm) for _ in range(self.n_edge_set)]
        self.encoders_edge = nn.ModuleList(encoders_edge)
        self.init_encoder_node()
        self.init_decoder()

    def init_decoder(self):
        self.decoder = self._make_mlp(self._latent_size_node, self._output_size, self._latent_size_node, layer_norm=False, nlayer=self.nlayer_mlp_encdec)

    def encode(self, graph):
        """Encodes node and edge features into latent features."""
        node_latents = self.encoder_node(graph.node_features)
        new_edges_sets = []
        for i,edge_set in enumerate(graph.edge_sets):
            latent = self.encoders_edge[i](edge_set.features)
            new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)

    def decode(self, graph):
        """Decodes node features from graph."""
        return self.decoder(graph.node_features)

    def preprocess(self, g, **kwx):
        if not self.preprocess_edgeset:
            return g
        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=g.edge_features,#self._edge_normalizer(edge_features, is_training),
            receivers=g.edge_index[0],
            senders=g.edge_index[1])
        g.edge_sets=[mesh_edges]
        return g

    def forward(self, graph, **kwx):
        """Encodes and processes a multigraph, and returns node features."""
        graph = self.preprocess(graph, **kwx)
        latent_graph = self.encode(graph)
        latent_graph = self.message_passing(latent_graph)
        return self.decode(latent_graph)
