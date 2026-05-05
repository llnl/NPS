# Lint as: python3
# pytorch port 

# ============================================================================
"""Model for neural phase simulation."""

import torch
import torch.nn as nn

from . import common, base_mp_gnn #, normalization
from .NPS_model import BaseNPSGNNModel
from NPS.model.common import vector_pbc


# DEPRECATED! do not use
class processor_only(base_mp_gnn.EncodeProcessDecode):
    """Encode-Process-Decode GraphNet model with encoder/decoder replaced by reversible 2-way autoencoder."""
    def init_encoder_decoder(self):
        encoders_edge =    [self._make_mlp(self.input_size_edge, self._latent_size_edge, self._latent_size_edge) for _ in range(self.n_edge_set)]
        self.encoders_edge = nn.ModuleList(encoders_edge)
        skip_dim_matching_layer = False
        if skip_dim_matching_layer: # WILL crash
            self.encoder_node = lambda x: x
            self.decoder = lambda x: x
        else:
            self.encoder_node = self._make_mlp(self.input_size_node, self._latent_size_node, self._latent_size_node, nlayer=self.nlayer_mlp_encdec)
            self.decoder = self._make_mlp(self._latent_size_node, self._output_size, self._latent_size_node, layer_norm=False, nlayer=self.nlayer_mlp_encdec)


class NPS_autoencoder(BaseNPSGNNModel):
    """Dimension reduction with autoencoder. raw_shape=(-1,W,H(,D),C), out_shape=(-1,w,h(,d),c)"""
    def __init__(self, autoencoder, *arg, dim=2, **kwarg):
        super().__init__(*arg, dim=dim, **kwarg)
        self.autoencoder = autoencoder
        self.ch_first_list = (0, dim+1) + tuple(range(1, dim+1))
        self.ch_last_list = (0,) + tuple(range(2, dim+2)) + (1,)
        self.nfeat_node_embedding = 0

    def input_to_channel_first(self, x, shape):
        return x.reshape(shape).permute(self.ch_first_list)

    def output_to_channel_first(self, x, shape):
        return x.reshape(shape).permute(self.ch_first_list)

    def forward(self, inputs, **kwx):
        # batch = inputs['aux_vars']['batch']
        x_in = self.autoencoder(self.input_to_channel_first(inputs['x'], inputs['aux_vars']['bwh']+(-1,)))
        x_in = torch.cat(x_in, 1).permute(self.ch_last_list)[:, None]
        inputs_enc = common.array2graph(x_in, self.dim, periodic=self.periodic, device=x_in.device, time_dim=True)
        graph = self.preprocess(inputs_enc, is_training=False)
        per_node_network_output = self._learned_model(graph)
        per_node_network_output = self.output_to_channel_first(per_node_network_output, inputs_enc['aux_vars']['bwh']+(-1,))
        per_node_network_output = self.autoencoder(torch.split(per_node_network_output, per_node_network_output.shape[1]//2, 1), False)
        per_node_network_output = per_node_network_output.permute(self.ch_last_list).reshape((-1, self.nfeat_out))
        return self._update(inputs, per_node_network_output)

