# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from line_profiler import profile
from torch.nn import BatchNorm1d, Linear, ModuleList, Sequential
from torch_geometric.nn import GINEConv, global_add_pool, GPSConv, pool, unpool
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import GraphConv


def exists(val):
    return val is not None


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        softmax=True,
        final_activation=None,
    ):
        super().__init__()
        hidden = []
        self.input_layer = nn.Linear(in_features, hidden_features, bias=True)
        for _i in range(hidden_layers):
            hidden.append(nn.SiLU())
            hidden.append(nn.Linear(hidden_features, hidden_features, bias=True))
        hidden.append(nn.SiLU())
        self.hidden = nn.Sequential(*hidden)
        self.output_layer = nn.Linear(hidden_features, out_features, bias=True)
        if softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = None
        self.final_activation = None
        if final_activation == "SiLU":
            self.final_activation = nn.SiLU()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()

    def forward(self, coords):
        l1 = self.input_layer(coords)
        l2 = self.hidden(l1)
        output = self.output_layer(l2)
        if self.softmax is not None:
            output = self.softmax(output)
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output


class ShapeMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        input_channel_dim,
        batch_norm=False,
        softmax=True,
        final_activation=None,
        garment_embedding_dim=None,
    ):
        super().__init__()
        channels = input_channel_dim
        layers = []
        while channels > 1:
            next_channels = max(1, channels // 4)
            layers.append(nn.Conv1d(channels, next_channels, kernel_size=1))
            if next_channels > 1:
                layers.append(nn.SiLU())
            channels = next_channels
        self.conv_layers = nn.Sequential(*layers)
        self.input_fc = nn.Linear(in_features, hidden_features, bias=True)
        if garment_embedding_dim is not None:
            self.input_fc = nn.Linear(
                in_features + garment_embedding_dim, hidden_features, bias=True
            )
        hidden = []
        for _i in range(hidden_layers):
            hidden.append(nn.SiLU())
            hidden.append(nn.Linear(hidden_features, hidden_features, bias=True))
            if batch_norm:
                hidden.append(nn.LayerNorm(hidden_features))
        hidden.append(nn.SiLU())
        self.hidden = nn.Sequential(*hidden)
        self.output_layer = nn.Linear(hidden_features, out_features, bias=True)
        if softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = None
        self.final_activation = None
        if final_activation == "SiLU":
            self.final_activation = nn.SiLU()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()

    def forward(self, x, garment_embedding=None):
        batch_size = x.shape[0]
        x = self.conv_layers(x).view(batch_size, -1)
        if garment_embedding is not None:
            x = torch.cat([x, garment_embedding], dim=-1)
        x = self.input_fc(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        if self.softmax is not None:
            x = self.softmax(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class SirenNetPerPoint(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        hidden_activation=None,
        final_activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
                dropout=dropout,
                activation=hidden_activation,
            )

            self.layers.append(layer)

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x, mods=None):
        if mods is not None:
            mods = tuple(
                mods.transpose(0, 1)
            )  # [batch x num_layer x num_hidden_features] --> num_layer x [batch x num_hidden_features]
            x = x.transpose(
                0, 1
            )  # [batch x num_points x inp_dim] --> [num_points x batch x inp_dim] {this is because modulations will be repeated across all bones}
            for layer, mod in zip(self.layers, mods):
                x = layer(x)
                x = x * mod
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.last_layer(x)

        if mods is not None:
            x = x.transpose(
                0, 1
            )  #  [num_points x batch x out_dim] --> [batch x num_points x out_dim]

        return x


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        hidden_activation=None,
        final_activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
                dropout=dropout,
                activation=hidden_activation,
            )

            self.layers.append(layer)

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x, mods=None, shifts=None):
        if mods is not None and shifts is None:
            mods = tuple(
                mods.transpose(0, 1)
            )  # [batch x num_layer x num_hidden_features] --> num_layer x [batch x num_hidden_features]
            for layer, mod in zip(self.layers, mods):
                x = layer(x)
                x = x * mod
        elif mods is not None and shifts is not None:
            mods = tuple(
                mods.transpose(0, 1)
            )  # [batch x num_layer x num_hidden_features] --> num_layer x [batch x num_hidden_features]
            shifts = tuple(
                shifts.transpose(0, 1)
            )  # [batch x num_layer x num_hidden_features] --> num_layer x [batch x num_hidden_features]
            for layer, mod, shift in zip(self.layers, mods, shifts):
                x = layer(x)
                x = x * mod + shift
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.last_layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, L=6):
        """
        L: Number of frequency bands (default: 6)
        """
        super().__init__()
        self.L = L

    def forward(self, x, include_input=True):
        """
        x: (N, 3) tensor of 3D coordinates
        Returns: (N, 3 * 2 * L) tensor of encoded features
        """
        freq_bands = 2 ** torch.arange(self.L, device=x.device) * torch.pi
        out = []
        if include_input:
            out.append(x)
        for freq in freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module, redraw_interval: int | None = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            print("Redrawing projections...")
            fast_attentions = [
                module
                for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class GraphTransformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        node_emb_dim: int,
        edge_emb_dim: int,
        pe_dim: int,
        num_layers: int,
        attn_type: str,
        cond_dim: int,
        nodes_feat_out_dim: int,
        pool_k: int,
        attn_kwargs: dict | None = None,
        predict_deltas: bool = True,
        perf_attn_redraw_interval: int | None = None,
    ):
        super().__init__()
        self.node_emb = Linear(node_emb_dim, channels - pe_dim)
        self.pe_lin = Linear(pe_dim, pe_dim)
        self.pe_norm = BatchNorm1d(pe_dim)
        self.edge_emb = Linear(edge_emb_dim, channels)
        self.predict_deltas = predict_deltas
        self.redraw_interval = perf_attn_redraw_interval
        self.pool_k = int(pool_k)

        # nodes encoder
        self.convs = ModuleList()
        for _ in range(num_layers):
            mpnn = Sequential(
                Linear(channels, channels),
                nn.SiLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(
                channels,
                GINEConv(mpnn),
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )
            self.convs.append(conv)
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=self.redraw_interval if attn_type == "performer" else None,
        )

        # global encoder
        self.pooling = pool.SAGPooling(channels, 0.5)
        self.global_convs = ModuleList()
        for _ in range(num_layers):
            global_mpnn = Sequential(
                Linear(channels, channels),
                nn.SiLU(),
                # pooling is done in 'forward'
            )
            global_convs = GPSConv(
                channels,
                GINEConv(global_mpnn),
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )
            self.global_convs.append(global_convs)
        self.final_pooling = pool.TopKPooling(
            channels, ratio=self.pool_k, min_score=None
        )

        self.redraw_global_projection = RedrawProjection(
            self.global_convs,
            redraw_interval=self.redraw_interval if attn_type == "performer" else None,
        )

        if self.predict_deltas:
            # nodes conditional encoder
            self.cond_edge_emb = Linear(edge_emb_dim, channels + cond_dim)
            self.conditional_convs = ModuleList()
            for _ in range(num_layers):
                cond_mpnn = Sequential(
                    Linear(channels + cond_dim, channels + cond_dim),
                    nn.SiLU(),
                    Linear(channels + cond_dim, channels + cond_dim),
                )
                cond_conv = GPSConv(
                    channels + cond_dim,
                    GINEConv(cond_mpnn),
                    heads=1,
                    attn_type=attn_type,
                    attn_kwargs=attn_kwargs,
                )
                self.conditional_convs.append(cond_conv)

            # nodes delta predictor
            self.delta_conv = GraphConv(
                channels + cond_dim, nodes_feat_out_dim, aggr="mean"
            )

            self.redraw_conditional_projection = RedrawProjection(
                self.conditional_convs,
                redraw_interval=(
                    self.redraw_interval if attn_type == "performer" else None
                ),
            )

    @profile
    def forward(
        self, x, pe, edge_index, edge_attr, batch, x_cond=None, stage="encoding"
    ):
        if stage == "encoding":
            x_pe = self.pe_norm(pe)
            x_emb = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
            edge_feats = self.edge_emb(edge_attr)
            # local encoding
            for conv in self.convs:
                x_emb = conv(x_emb, edge_index, batch, edge_attr=edge_feats)
            x_local = x_emb.clone()
            # global encoding
            pooled_edge_index = edge_index.clone()
            pooled_batch = batch.clone()
            pooled_edge_feats = edge_feats.clone()
            for conv in self.global_convs:
                x_emb = conv(
                    x_emb, pooled_edge_index, pooled_batch, edge_attr=pooled_edge_feats
                )
                pooled = self.pooling(
                    x_emb,
                    edge_index=pooled_edge_index,
                    edge_attr=pooled_edge_feats,
                    batch=pooled_batch,
                )
                x_emb, pooled_edge_index, pooled_edge_feats, pooled_batch = (
                    pooled[0],
                    pooled[1],
                    pooled[2],
                    pooled[3],
                )
            # global pooling
            if self.pool_k > 0:
                x_global = self.final_pooling(
                    x_emb,
                    edge_index=pooled_edge_index,
                    edge_attr=pooled_edge_feats,
                    batch=pooled_batch,
                )
                x_global = x_global[0]  # pooled node features only
            else:
                x_global = global_add_pool(x_emb, pooled_batch)

            return x_local, x_global

        elif stage == "decoding":
            # conditional local decoding
            if self.predict_deltas:
                x_conditioned = torch.cat([x, x_cond], dim=-1)
                edge_feats_cond = self.cond_edge_emb(edge_attr)
                for conv in self.conditional_convs:
                    x_conditioned = conv(
                        x_conditioned, edge_index, batch, edge_attr=edge_feats_cond
                    )
                x_delta = self.delta_conv(x_conditioned, edge_index)

            return x_delta


class PreActivationResidualBlock(nn.Module):
    def __init__(self, dim, use_layer_norm=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.activation = nn.SiLU()
        self.linear = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear(x)
        return x + residual  # Skip connection


class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, target_size=1):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings, mask=None):
        # token_embeddings: (batch, seq_len, hidden_size)
        # mask: (batch, seq_len)
        scores = self.attention(token_embeddings).squeeze(-1)  # (batch, seq_len)``
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Mask out padding tokens
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        pooled = torch.bmm(weights.unsqueeze(1), token_embeddings).squeeze(
            1
        )  # (batch, hidden_size)
        return pooled


class ConditioningEmbedder(nn.Module):
    def __init__(
        self,
        config_params,
        dropout=0.1,
        use_noact_layer=False,
    ):
        super().__init__()

        self.pos_embed_type = config_params["pos_enc"]
        self.pos_embed_dim = config_params["pos_enc_dim"]
        self.in_dim = config_params["in_dim"]
        self.max_token_len = config_params["max_token_len"]
        self.hidden_dim = config_params["hidden_dim"]
        self.hidden_layers = config_params["hidden_layers"]
        self.nheads = config_params["nheads"]
        self.dim_feedforward = 4 * config_params["hidden_dim"]
        self.out_dim = config_params["out_dim"]
        self.use_decoder = config_params["use_decoder"]
        self.query_cond_dim = config_params["query_cond_dim"]
        self.use_noact_layer = use_noact_layer
        # Positional encoding
        if "fourier" in self.pos_embed_type:
            self.pos_embed = PositionalEncoding(L=self.pos_embed_dim)
            self.in_dim += 3 * 2 * self.pos_embed_dim
            self.query_cond_dim += 3 * 2 * self.pos_embed_dim

        self.proj_in = nn.Linear(self.in_dim, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            activation=nn.SiLU(),
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.hidden_layers
        )

        if self.use_decoder:
            self.decoder_proj_in = nn.Linear(
                self.hidden_dim + self.query_cond_dim, self.hidden_dim
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                activation=nn.SiLU(),
                batch_first=True,
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=self.hidden_layers
            )

        # recommended to intialize the transformer layers manually, otherwise they will use same initial prameters
        for module in self.transformer_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.use_decoder:
            for module in self.transformer_decoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # last encoder layer without activation
        if self.use_noact_layer:
            noact_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dim_feedforward=self.dim_feedforward,
                activation=self.no_activation,
                batch_first=True,
            )
            self.noact_encoder_layer = nn.TransformerEncoder(
                noact_encoder_layer, num_layers=1
            )

        if self.out_dim > 0:
            self.token_pooling = SelfAttentionPooling(self.hidden_dim)
            self.proj_out = nn.Linear(self.hidden_dim, self.out_dim)

    def no_activation(self, x):
        return x

    @profile
    def forward(self, x, x_memory=None, x_cond=None, attn_mask=None):
        _batch_size = x.shape[0]  # noqa: F841
        _seq_len = x.shape[1]  # noqa: F841
        _input_dim = x.shape[2]  # noqa: F841
        # positional encoding
        if "fourier" in self.pos_embed_type:
            x_posenc = self.pos_embed(x[:, :, :3])
            x = torch.cat([x_posenc, x[:, :, 3:]], dim=-1)
        # input projection
        x = self.proj_in(x)
        # encode
        x = self.transformer_encoder(x, mask=attn_mask)
        # decode
        if self.use_decoder and x_memory is not None:
            if x_cond is not None:
                if "fourier" in self.pos_embed_type:
                    x_cond = self.pos_embed(x_cond)
                x = torch.cat([x, x_cond], dim=-1)
                x = self.decoder_proj_in(x)
            x = self.transformer_decoder(x, x_memory)
        # no activation
        x_noact = None
        if self.use_noact_layer:
            x_noact = self.noact_encoder_layer(x)
        # pool
        x_out = None
        if self.out_dim > 0:
            x_pooled = self.token_pooling(x)
            x_out = self.proj_out(x_pooled)
        return x, x_noact, x_out


# ======================================================================================================================


class HyperModulator(nn.Module):
    def __init__(
        self,
        hypermod_config,
        max_num_bones=8,
        predict_deltas=True,
        train_stage="stage_1",
    ):
        super().__init__()

        self.max_num_bones = max_num_bones
        self.predict_deltas = predict_deltas
        self.use_seperate_modheads = hypermod_config["use_seperate_modheads"]
        self.use_pattern_latents = hypermod_config["use_pattern_latents"]
        self.drape_cond_type = hypermod_config["drape_cond_type"]
        self.drape_cond_dim = hypermod_config["drape_cond_dim"]
        self.verts_in_feat_dim = 5  # (XYZ + Pinning Label + Bone Label)
        self.verts_in_feat_dim += (
            self.drape_cond_dim if self.drape_cond_type == "input" else 0
        )
        self.verts_out_feat_dim = (
            3 + self.max_num_bones
        )  # (Delta XYZ + Delta Skinning Weights) ---> Shape specific
        self.edge_in_feat_dim = 5  # ( geodesic + point-pair-features{diffvec, angle(diffvec,norm1), angle(diffvec,norm2), angle(norm1,norm2)} )
        self.graph_use_panels_xyz = hypermod_config["graph_use_panels_xyz"]
        self.use_graph_transformer = hypermod_config["use_graph_transformer"]
        self.use_bone_encodings = hypermod_config["use_bone_encodings"]
        self.cond_dim = 0
        self.mesh_proj_dim = 0
        self.train_stage = train_stage

        self.mesh_encoder = GraphTransformer(
            channels=hypermod_config["graph_hidden_dim"],
            node_emb_dim=self.verts_in_feat_dim,
            edge_emb_dim=self.edge_in_feat_dim,
            nodes_feat_out_dim=self.verts_out_feat_dim,
            pool_k=hypermod_config["graph_pool_k"],
            pe_dim=hypermod_config["graph_pe_dim"],
            num_layers=hypermod_config["graph_hidden_layers"],
            attn_type=hypermod_config["graph_attn_type"],
            cond_dim=hypermod_config["bones_feats_out_dim"],
            predict_deltas=self.predict_deltas,
            perf_attn_redraw_interval=hypermod_config["graph_attn_redraw_interval"],
        )
        # garment embedder
        garment_embedder_config = hypermod_config["garment_embedder_config"]
        self.garment_tokenizer = ConditioningEmbedder(
            config_params=garment_embedder_config,
        )
        self.cond_dim += hypermod_config["garment_embedder_config"]["out_dim"]

        # body shape embedder
        shape_embedder_config = hypermod_config["shape_embedder_config"]
        self.shape_embedder = ConditioningEmbedder(
            config_params=shape_embedder_config,
        )
        self.cond_dim += hypermod_config["shape_embedder_config"]["out_dim"]

        # garment vertex feature predictor
        vertex_feature_extractor_config = hypermod_config[
            "vertex_feature_extractor_config"
        ]
        self.vertex_feature_extractor = ConditioningEmbedder(
            config_params=vertex_feature_extractor_config,
            use_noact_layer=True,
        )

        # garment bones feature encoder
        bone_feature_extractor_config = hypermod_config["bone_feature_extractor_config"]
        self.bone_feature_extractor = ConditioningEmbedder(
            config_params=bone_feature_extractor_config,
            use_noact_layer=True,
        )

        if self.use_pattern_latents:
            pattern_proj_dim = (
                hypermod_config["pattern_latent_dim"][0]
                * hypermod_config["pattern_latent_dim"][1]
            )
            self.cond_dim += pattern_proj_dim

        self.mesh_proj_dim = hypermod_config["mesh_proj_dim"]
        self.mesh_global_projection = nn.Linear(
            hypermod_config["graph_hidden_dim"] * hypermod_config["graph_pool_k"],
            self.mesh_proj_dim,
        )

        # shared hypermodulator
        hypermodulator = []
        # inp_dim = self.cond_dim + self.mesh_proj_dim
        inp_dim = self.cond_dim
        hypermodulator.append(
            nn.Linear(inp_dim, hypermod_config["hidden_dim"], bias=True)
        )
        hypermodulator.append(nn.SiLU())
        if hypermod_config["residual_blocks"]:
            for _i in range(hypermod_config["hidden_layers"]):
                hypermodulator.append(
                    PreActivationResidualBlock(
                        hypermod_config["hidden_dim"],
                        use_layer_norm=hypermod_config["layer_norm"],
                    )
                )
        else:
            for _i in range(hypermod_config["hidden_layers"]):
                if hypermod_config["layer_norm"]:
                    hypermodulator.append(nn.LayerNorm(hypermod_config["hidden_dim"]))
                hypermodulator.append(
                    nn.Linear(
                        hypermod_config["hidden_dim"],
                        hypermod_config["hidden_dim"],
                        bias=True,
                    )
                )
                hypermodulator.append(nn.SiLU())
        self.hypermodulator = nn.Sequential(*hypermodulator)

        self.num_modulation_heads = hypermod_config["out_dim"][1]
        if self.use_seperate_modheads:
            mod_out_dim = hypermod_config["out_dim"][0]
            # separate hypermodulator head for each layer of the network being modulated
            self.modulation_heads = ModuleList()
            for _i in range(self.num_modulation_heads):
                mod_head = []
                mod_head.append(
                    nn.Linear(hypermod_config["hidden_dim"], mod_out_dim, bias=True)
                )
                for _layer_i in range(hypermod_config["modheads_hidden_layers"]):
                    mod_head.append(nn.SiLU())
                    mod_head.append(nn.Linear(mod_out_dim, mod_out_dim, bias=True))
                self.modulation_heads.append(nn.Sequential(*mod_head))
        else:
            mod_out_dim = hypermod_config["out_dim"][0] * hypermod_config["out_dim"][1]
            self.modulation_heads = nn.Linear(
                hypermod_config["hidden_dim"], mod_out_dim, bias=True
            )

    @profile
    def forward(self, pattern_latents, shape_latents, graph_data, drape_cond_feat=None):
        _device = graph_data.x.device  # noqa: F841
        batch_size = shape_latents.shape[0]

        hyper_modulations = None
        mesh_local_feats = None
        mesh_global_feats = None
        _mesh_out_feats = None  # noqa: F841
        _bones_out_feats = None  # noqa: F841

        cond_emb = []

        _batch_indices = graph_data.batch  # noqa: F841

        # shape embedder
        shape_tokens, _, shape_global_emb = self.shape_embedder(x=shape_latents)
        cond_emb.append(shape_global_emb)

        # mesh encoder
        mesh_local_feats, mesh_global_feats = self.mesh_encoder(
            graph_data.x,
            graph_data.pe,
            graph_data.edge_index,
            graph_data.edge_attr,
            graph_data.batch,
            stage="encoding",
        )
        mesh_local_feats = mesh_local_feats.repeat(batch_size, 1, 1)
        mesh_global_feats = mesh_global_feats.repeat(batch_size, 1, 1)

        # garment vertices & body cross-attention
        vertex_tokens, _, _ = self.garment_tokenizer(
            x=mesh_local_feats, x_memory=shape_tokens, x_cond=drape_cond_feat
        )

        # per-vertex prediction
        _verts_out_feats, verts_deltas, vertex_global_emb = (
            self.vertex_feature_extractor(x=vertex_tokens)
        )
        cond_emb.append(vertex_global_emb)

        # bones graph features
        bones_lod_indices = graph_data.bones_lod_sampled
        max_lod = max(list(bones_lod_indices.keys()))
        bones_sampled_indices = bones_lod_indices[max_lod]
        bones_local_feats = mesh_local_feats.clone()[:, bones_sampled_indices, :]
        bones_drape_cond_feat = drape_cond_feat.clone()[:, bones_sampled_indices, :]

        # garment bones & body cross-attention
        bones_tokens, _, _ = self.garment_tokenizer(
            x=bones_local_feats,
            x_memory=mesh_global_feats,
            x_cond=bones_drape_cond_feat,
        )
        _, bones_deltas, _ = self.bone_feature_extractor(x=bones_tokens)

        if self.train_stage == "stage_1":
            return verts_deltas, bones_deltas, bones_tokens

        else:
            # combine all condition embeddings
            hypermod_inp_emb = shape_global_emb + vertex_global_emb
            # # predict modulations
            hypermod_feats = self.hypermodulator(hypermod_inp_emb)
            if self.use_seperate_modheads:
                hyper_modulations = []
                for hypermod_head in self.modulation_heads:
                    mods = hypermod_head(hypermod_feats)
                    hyper_modulations.append(mods)
                hyper_modulations = torch.stack(hyper_modulations)
                # move batch first
                hyper_modulations = hyper_modulations.transpose(0, 1)
            else:
                hyper_modulations = self.modulation_heads(hypermod_feats)
                hyper_modulations = hyper_modulations.view(
                    batch_size, self.num_modulation_heads, -1
                )

            return verts_deltas, bones_deltas, bones_tokens, hyper_modulations

        # # unmerge graphs
        # verts_feats = []
        # verts_deltas = []
        # for i in range(batch_size):
        #     verts_feats.append(mesh_local_feats[graph.batch==i])

        # if self.use_pattern_encodings:
        #     x = torch.einsum('bsi,sio->bso', x, self.projection_weights).squeeze(-1)
        #     x = self.projection_activation(x)
        # if self.use_graph_conv and graph_data is not None:
        #     x_graph_1 = self.graph_conv_1(graph_data.x, graph_data.edge_index)
        #     x_graph_1 = nn.SiLU()(x_graph_1)
        #     x_pooled_1 = self.pool_1(x_graph_1, graph_data.edge_index)
        #     x_graph_2, pooled_edges = x_pooled_1[0], x_pooled_1[1]
        #     x_graph_2 = self.graph_conv_2(x_graph_2, pooled_edges)
        #     x_graph_2 = nn.SiLU()(x_graph_2)
        #     x_pooled_2 = self.pool_2(x_graph_2, pooled_edges)
        #     x_graph_3, pooled_edges = x_pooled_2[0], x_pooled_2[1]
        #     x_graph_3 = self.graph_conv_3(x_graph_3, pooled_edges)
        #     x_graph_3 = nn.SiLU()(x_graph_3)
        #     x_pooled_3 = self.pool_3(x_graph_3, pooled_edges)
        #     x_graph_4, pooled_edges = x_pooled_3[0], x_pooled_3[1]
        #     x_graph_4 = self.graph_conv_4(x_graph_4, pooled_edges)
        #     x_graph_4 = nn.SiLU()(x_graph_4)
        #     x_pooled_4 = self.pool_4(x_graph_4, pooled_edges)
        #     x_graph, pooled_edges = x_pooled_4[0], x_pooled_4[1]
        #     if self.use_delta_skin_weights:
        #         unpooled_2 = unpool.knn_interpolate(x_graph_3, graph_data.x[x_pooled_2[4]][:,:3], graph_data.x[x_pooled_1[4]][:,:3])
        #         x_up_graph_2 = self.graph_upconv_2(unpooled_2, x_pooled_1[1])
        #         x_up_graph_2 = nn.SiLU()(x_up_graph_2)
        #         unpooled_1 = unpool.knn_interpolate(x_up_graph_2, graph_data.x[x_pooled_1[4]][:,:3], graph_data.x[:,:3])
        #         x_up_graph_1 = self.graph_upconv_1(unpooled_1, graph_data.edge_index)
        #         x_up_graph_1 = nn.SiLU()(x_up_graph_1)
        #         x_verts_feats =  self.graph_upconv_0(x_up_graph_1, graph_data.edge_index)
        #         x_verts_feats = nn.Tanh()(x_verts_feats)

        #     x_graph = x_graph.view(1,-1).repeat(len(x),1)
        #     if self.use_pattern_encodings:
        #         x = torch.cat([x, x_graph], dim=-1)
        #     else:
        #         x = x_graph
        # x = self.input_fc(x)
        # x = self.hidden(x)
        # x = self.output_layer(x)
        # if self.softmax is not None:
        #     x = self.softmax(x)
        # if self.final_activation is not None:
        #     x = self.final_activation(x)

        # if self.use_delta_skin_weights:
        #     return x, x_verts_feats
        # return x


class GarmentGraphMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        input_channel_dim,
        batch_norm=False,
        softmax=True,
        final_activation=None,
        garment_embedding_dim=None,
        use_graph_conv=False,
        graph_in_feature_dim=None,
        graph_out_dim=None,
        graph_hidden_layers=2,
        graph_hidden_features=64,
        use_pattern_encodings=True,
        use_delta_skin_weights=True,
    ):
        super().__init__()

        self.use_graph_conv = use_graph_conv
        self.use_pattern_latents = use_pattern_encodings
        self.use_delta_skin_weights = use_delta_skin_weights
        self.verts_out_feat_dim = 139

        inp_fc_features = 0

        # define Graph encoder
        if self.use_pattern_encodings:
            print()
            print("USING SEWING PATTERN EMBEDDINGS.....")
            print()
            # for channelwise-reduction of garment embeddings
            self.projection_weights = nn.Parameter(
                torch.randn(in_features, input_channel_dim, 1), requires_grad=True
            )
            self.projection_activation = nn.SiLU()
            inp_fc_features += in_features

        if self.use_graph_conv:
            print()
            print("USING GCN.....")
            print()
            if graph_in_feature_dim is None:
                graph_in_feature_dim = 3
            if graph_out_dim is None:
                graph_out_dim = in_features
            inp_fc_features += graph_out_dim
            self.graph_conv_1 = GraphConv(graph_in_feature_dim, graph_hidden_features)
            self.pool_1 = pool.SAGPooling(graph_hidden_features, 0.5)
            self.graph_conv_2 = GraphConv(
                graph_hidden_features, graph_hidden_features // 2
            )
            self.pool_2 = pool.SAGPooling(graph_hidden_features // 2, 0.5)
            self.graph_conv_3 = GraphConv(
                graph_hidden_features // 2, graph_hidden_features // 4
            )
            self.pool_3 = pool.SAGPooling(graph_hidden_features // 4, 0.5)
            self.graph_conv_4 = GraphConv(graph_hidden_features // 4, 1)
            self.pool_4 = pool.SAGPooling(1, graph_out_dim)

        if self.use_delta_skin_weights:
            self.graph_upconv_2 = GraphConv(
                graph_hidden_features // 4, graph_hidden_features // 2
            )
            self.graph_upconv_1 = GraphConv(
                graph_hidden_features // 2, graph_hidden_features
            )
            self.graph_upconv_0 = GraphConv(
                graph_hidden_features, self.verts_out_feat_dim
            )

        self.input_fc = nn.Linear(inp_fc_features, hidden_features, bias=True)

        hidden = []
        for _i in range(hidden_layers):
            hidden.append(nn.SiLU())
            hidden.append(nn.Linear(hidden_features, hidden_features, bias=True))
            if batch_norm:
                hidden.append(nn.LayerNorm(hidden_features))
        hidden.append(nn.SiLU())
        self.hidden = nn.Sequential(*hidden)
        self.output_layer = nn.Linear(hidden_features, out_features, bias=True)
        if softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = None

        self.final_activation = None
        if final_activation == "SiLU":
            self.final_activation = nn.SiLU()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()

    def forward(self, x, graph_data=None):
        _batch_size = x.shape[0]  # noqa: F841
        if self.use_pattern_encodings:
            x = torch.einsum("bsi,sio->bso", x, self.projection_weights).squeeze(-1)
            x = self.projection_activation(x)
        if self.use_graph_conv and graph_data is not None:
            x_graph_1 = self.graph_conv_1(graph_data.x, graph_data.edge_index)
            x_graph_1 = nn.SiLU()(x_graph_1)
            x_pooled_1 = self.pool_1(x_graph_1, graph_data.edge_index)
            x_graph_2, pooled_edges = x_pooled_1[0], x_pooled_1[1]
            x_graph_2 = self.graph_conv_2(x_graph_2, pooled_edges)
            x_graph_2 = nn.SiLU()(x_graph_2)
            x_pooled_2 = self.pool_2(x_graph_2, pooled_edges)
            x_graph_3, pooled_edges = x_pooled_2[0], x_pooled_2[1]
            x_graph_3 = self.graph_conv_3(x_graph_3, pooled_edges)
            x_graph_3 = nn.SiLU()(x_graph_3)
            x_pooled_3 = self.pool_3(x_graph_3, pooled_edges)
            x_graph_4, pooled_edges = x_pooled_3[0], x_pooled_3[1]
            x_graph_4 = self.graph_conv_4(x_graph_4, pooled_edges)
            x_graph_4 = nn.SiLU()(x_graph_4)
            x_pooled_4 = self.pool_4(x_graph_4, pooled_edges)
            x_graph, pooled_edges = x_pooled_4[0], x_pooled_4[1]
            if self.use_delta_skin_weights:
                unpooled_2 = unpool.knn_interpolate(
                    x_graph_3,
                    graph_data.x[x_pooled_2[4]][:, :3],
                    graph_data.x[x_pooled_1[4]][:, :3],
                )
                x_up_graph_2 = self.graph_upconv_2(unpooled_2, x_pooled_1[1])
                x_up_graph_2 = nn.SiLU()(x_up_graph_2)
                unpooled_1 = unpool.knn_interpolate(
                    x_up_graph_2,
                    graph_data.x[x_pooled_1[4]][:, :3],
                    graph_data.x[:, :3],
                )
                x_up_graph_1 = self.graph_upconv_1(unpooled_1, graph_data.edge_index)
                x_up_graph_1 = nn.SiLU()(x_up_graph_1)
                x_verts_feats = self.graph_upconv_0(x_up_graph_1, graph_data.edge_index)
                x_verts_feats = nn.Tanh()(x_verts_feats)

            x_graph = x_graph.view(1, -1).repeat(len(x), 1)
            if self.use_pattern_encodings:
                x = torch.cat([x, x_graph], dim=-1)
            else:
                x = x_graph
        x = self.input_fc(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        if self.softmax is not None:
            x = self.softmax(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        if self.use_delta_skin_weights:
            return x, x_verts_feats
        return x


# SIREN WITH BATCHED MODULATION


class ModulatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        softmax=False,
        final_activation=None,
        hidden_activation=None,
    ):
        super().__init__()
        if hidden_activation is None:
            hidden_activation = nn.SiLU()
        hidden = []
        self.input_layer = nn.Linear(in_features, hidden_features, bias=True)
        for _i in range(hidden_layers):
            hidden.append(hidden_activation)
            hidden.append(nn.Linear(hidden_features, hidden_features, bias=True))
        hidden.append(hidden_activation)
        self.hidden = nn.Sequential(*hidden)
        self.output_layer = nn.Linear(hidden_features, out_features, bias=True)
        if softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = None

        self.final_activation = final_activation

        # Store number of layers for modulation
        self.num_layers = hidden_layers

    def forward(self, coords, mods=None, shifts=None):
        # Process input layer
        x = self.input_layer(coords)

        # Process hidden layers with optional modulation
        if mods is not None and shifts is None:
            mods = tuple(
                mods.transpose(0, 1)
            )  # [batch x num_layer x num_hidden_features] --> num_layer x [batch x num_hidden_features]
            mod_idx = 0
            for _i, layer in enumerate(self.hidden):
                x = layer(x)
                # Apply modulation after each linear layer (skip SiLU layers)
                if isinstance(layer, nn.Linear) and mod_idx < len(mods):
                    x = x * mods[mod_idx]
                    mod_idx += 1
        elif mods is not None and shifts is not None:
            mods = tuple(
                mods.transpose(0, 1)
            )  # [batch x num_layer x num_hidden_features] --> num_layer x [batch x num_hidden_features]
            shifts = tuple(
                shifts.transpose(0, 1)
            )  # [batch x num_layer x num_hidden_features] --> num_layer x [batch x num_hidden_features]
            mod_idx = 0
            for _i, layer in enumerate(self.hidden):
                x = layer(x)
                # Apply modulation and shift after each linear layer (skip SiLU layers)
                if isinstance(layer, nn.Linear) and mod_idx < len(mods):
                    x = x * mods[mod_idx] + shifts[mod_idx]
                    mod_idx += 1
        else:
            x = self.hidden(x)

        # Process output layer
        output = self.output_layer(x)

        if self.softmax is not None:
            output = self.softmax(output)
        if self.final_activation is not None:
            output = self.final_activation(output)

        return output


# Alias for backward compatibility
GarmentMLP = GarmentGraphMLP
