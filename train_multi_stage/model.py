import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GeoGNNConv(MessagePassing):
    def __init__(self, hidden_dim, edge_in_dim, dropout=0.1, use_bn=True):
        super().__init__(aggr="mean")
        self.hidden_dim = hidden_dim

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.GELU(),
        )

        self.message_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        out = x + self.dropout(out)

        if self.use_bn and out.size(0) > 1:
            out = self.bn(out)

        out = self.norm(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        score = (x_i * x_j).sum(dim=-1, keepdim=True) / (self.hidden_dim ** 0.5)
        alpha = torch.softmax(score, dim=0)
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m = self.message_net(m)
        return alpha * m


class GraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        drop = config.gnn_dropout

        bond_dim = config.edge_feature_dim
        angle_dim = 1
        dihedral_dim = 1

        self.coord_linear = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.atom_gnn = nn.ModuleList([
            GeoGNNConv(self.hidden_dim, bond_dim, dropout=drop)
            for _ in range(config.num_atom_layers)
        ])

        self.angle_gnn = nn.ModuleList([
            GeoGNNConv(self.hidden_dim, angle_dim, dropout=drop)
            for _ in range(config.num_angle_layers)
        ])

        self.dihedral_gnn = nn.ModuleList([
            GeoGNNConv(self.hidden_dim, dihedral_dim, dropout=drop)
            for _ in range(config.num_dihedral_layers)
        ])

        self.fuse_linear = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.fuse_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(drop)

    def _run_gnn_stack(self, x, edge_index, edge_attr, gnn_stack):
        if edge_index is None or edge_attr is None:
            return x

        if edge_index.numel() == 0 or edge_attr.numel() == 0:
            return x

        out = x
        for conv in gnn_stack:
            out = conv(out, edge_index, edge_attr)

        return out

    def forward(
        self,
        pos,
        atom_edge_index,
        edge_attr,
        angle_edge_index,
        angle_attr,
        dihedral_edge_index,
        dihedral_attr,
    ):
        x = self.coord_linear(pos)

        atom_x = self._run_gnn_stack(
            x,
            atom_edge_index,
            edge_attr,
            self.atom_gnn,
        )

        angle_x = self._run_gnn_stack(
            x,
            angle_edge_index,
            angle_attr,
            self.angle_gnn,
        )

        dihedral_x = self._run_gnn_stack(
            x,
            dihedral_edge_index,
            dihedral_attr,
            self.dihedral_gnn,
        )

        fused = torch.cat([atom_x, angle_x, dihedral_x], dim=-1)
        fused = self.dropout(self.fuse_linear(fused))
        fused = self.fuse_norm(fused)
        return fused


class MultiGraphGeoGNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = GraphEncoder(config)

        self.n_domains = getattr(config, "n_domains", 3)
        self.dom_embed_dim = getattr(config, "domain_embed_dim", 32)
        self.infer_domain_idx = getattr(config, "infer_domain_idx", 2)

        self.dom_embed = nn.Embedding(self.n_domains, self.dom_embed_dim)
        self.film_gamma = nn.Linear(self.dom_embed_dim, config.hidden_dim)
        self.film_beta = nn.Linear(self.dom_embed_dim, config.hidden_dim)

        self.delta_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 3),
        )

        self.max_delta = getattr(config, "max_delta", None)

    @staticmethod
    def _get_batch_vector(data):
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch

        return torch.zeros(
            data.pos.size(0),
            dtype=torch.long,
            device=data.pos.device,
        )

    @staticmethod
    def _center_by_batch(pos, batch):
        pos_centered = pos.clone()

        for b in torch.unique(batch):
            mask = batch == b
            pos_centered[mask] = pos[mask] - pos[mask].mean(dim=0, keepdim=True)

        return pos_centered

    @staticmethod
    def _remove_delta_translation(delta, batch):
        delta_new = delta.clone()

        for b in torch.unique(batch):
            mask = batch == b
            delta_new[mask] = delta[mask] - delta[mask].mean(dim=0, keepdim=True)

        return delta_new

    def _get_domain_embedding_per_node(self, data, fused, batch):
        device = fused.device
        n_nodes = fused.size(0)

        if not hasattr(data, "domain_idx") or data.domain_idx is None:
            dom_node = torch.full(
                (n_nodes,),
                self.infer_domain_idx,
                device=device,
                dtype=torch.long,
            )
            return self.dom_embed(dom_node)

        dom = data.domain_idx.to(device)

        if dom.dim() == 0:
            dom_node = dom.expand(n_nodes)

        elif dom.numel() == n_nodes:
            dom_node = dom

        else:
            n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

            if dom.numel() == n_graphs:
                dom_node = dom[batch]
            else:
                dom_node = dom.reshape(-1)[0].expand(n_nodes)

        return self.dom_embed(dom_node.long())

    def forward(self, data):
        batch = self._get_batch_vector(data)

        pos_centered = self._center_by_batch(data.pos, batch)

        fused = self.encoder(
            pos_centered,
            data.atom_edge_index,
            data.edge_attr,
            data.angle_edge_index,
            data.angle_edge_attr,
            data.dihedral_edge_index,
            data.dihedral_edge_attr,
        )

        emb = self._get_domain_embedding_per_node(data, fused, batch)

        gamma = torch.tanh(self.film_gamma(emb))
        beta = torch.tanh(self.film_beta(emb))

        h_mod = fused * (1.0 + gamma) + beta

        delta_raw = self.delta_head(h_mod)

        if self.max_delta is not None:
            delta = self.max_delta * torch.tanh(delta_raw / self.max_delta)
        else:
            delta = delta_raw

        delta = self._remove_delta_translation(delta, batch)

        coords = data.pos + delta

        return coords