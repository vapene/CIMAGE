import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import math
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
)
import numpy as np
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling, degree
from torch_geometric.nn import knn_graph
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_cluster import random_walk
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
# custom modules
from loss import setup_loss_fn, ce_loss
# torch version
from CMI_HSIC.models import api
from utils import get_gpu_memory_usage, seed_worker

# from api import Proposed_HSIC_Lasso

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class MLPEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPEncoder, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(Linear(first_channels, second_channels))
            self.bns.append(bn(second_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, args, x, remaining_edges, train_neighbors, y, train_mask, train_B):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x)
            x = self.bns[i](x)
            # x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x)
        x = self.bns[-1](x)
        disen_embedding, score_list = self.disentangle(args, x, remaining_edges, train_neighbors, y, train_mask, train_B)
        return disen_embedding, score_list

    def pairwise_euclidean_distances(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf).sqrt()

    def cknn_graph(self, X, k, delta=1):
        assert k < X.shape[0]
        D = self.pairwise_euclidean_distances(X, X)
        D.fill_diagonal_(0.0)
        D_low_k, _ = torch.topk(D, k=k, largest=False, dim=-1)
        D_low_k = D_low_k[:, -1]
        adj = (D.square() < delta * delta * torch.matmul(D_low_k.view(-1, 1),
                                                         D_low_k.view(1, -1))).float().fill_diagonal_(0.0)
        adj = (adj + adj.T) / 2.0
        return adj

    def gcn_agg(self, adj, X):
        adjacency_matrix = adj.coalesce()
        adj_indices = adjacency_matrix.indices()
        adj_values = adjacency_matrix.values()
        num_nodes = X.shape[0]
        # add self-loop
        self_edge_indices = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1).to(X.device)
        self_edge_values = torch.ones(num_nodes).to(X.device)
        adj_indices = torch.cat([adj_indices, self_edge_indices], dim=1)
        adj_values = torch.cat([adj_values, self_edge_values])
        adjacency_matrix = torch.sparse_coo_tensor(adj_indices, adj_values, (X.shape[0],X.shape[0]))
        # calculate D
        adjacency_matrix = adjacency_matrix.coalesce()
        row_sum_inv = torch.sqrt(torch.sparse.sum(adjacency_matrix, dim=0).values()).pow(-1)
        row_sum_inv_diag = torch.sparse_coo_tensor(self_edge_indices, row_sum_inv, (X.shape[0],X.shape[0]))
        normalized_adjacency_matrix = row_sum_inv_diag @ adjacency_matrix @ row_sum_inv_diag
        # Compute the degree vector
        # identity = torch.eye(adj.shape[0]).to(adj.device)
        # adj_I = adj + identity
        # d = torch.sum(adj_I, axis=1)
        # d = 1 / torch.sqrt(d)
        # D = torch.diag(d)
        # normalized_adj = D.to_sparse() @ adj_I.to_sparse() @ D.to_sparse()
        return torch.sparse.mm(normalized_adjacency_matrix, X)

    def disentangle(self, args, z, remaining_edges, remaining_neighbors, y, train_mask, block):  # z shape 2708,256
        remain_data_adj_sp = torch.sparse_coo_tensor(remaining_edges, torch.ones(len(remaining_edges[0])).to(y.device),
                                                     [z.shape[0], z.shape[0]])
        ch_dim = z.shape[1] // args.ncaps
        hidden_xs = []
        x = routing_layer_32(z, args.ncaps, args.nlayer, args.max_iter, remaining_neighbors)
        # if args.dataset in ['Cora','Citeseer','Photo']:
        #     x = routing_layer_32(z, args.ncaps, args.nlayer, args.max_iter, remaining_neighbors)
        # else:
        #     x = routing_layer_16(z, args.ncaps, args.nlayer, args.max_iter, remaining_neighbors)


        hidden_xs.append(x.view(-1, z.shape[1]))  # 675 torch.Size([2708, 512])
        x = x.view(-1, args.ncaps, z.shape[1] // args.ncaps)  # 2708,16,32
        hidden_disen_x = x.detach().clone()
        result = []
        model_PH1 = api.Proposed_HSIC_Lasso(device=y.device, ch_dim=ch_dim, lam=[np.inf, args.hsic_lamb])  # 원래 tol 1e-5

        model_PH1.input(hidden_xs[0][train_mask], y[train_mask])
        model_PH1.classification_multi(B=block, kernels=['Gaussian'], n_jobs=8)
        _, scores = model_PH1.get_index_score()
        scores_list = torch.mean(scores.reshape(-1, ch_dim), axis=1)  # channel 8개
        # normalize_score_list = [float(i) / max(scores_list + 1e-10) for i in scores_list]  # 0~1사이

        min_value = min(scores_list)
        max_value = max(scores_list)
        normalize_score_list = [float(i) / max(scores_list + 1e-10) for i in scores_list]  # 0~1사이

    for idx_f in range(args.ncaps):
            cur_X = hidden_disen_x[:, idx_f, :]
            # if args.dataset in ['Cora','Citeseer','Photo']:
            new_adj = self.cknn_graph(X=cur_X, k=args.knn_nb)

            cur_adj_sp = normalize_score_list[idx_f] * args.new_lamb * new_adj.to_sparse() + (
                        (1 + 1e-10) - normalize_score_list[idx_f]) * args.original_lamb * remain_data_adj_sp

            # else:
            #     knn_edge = knn_graph(cur_X, k=args.knn_nb, loop=False)
            #     new_adj_sp = torch.sparse_coo_tensor(knn_edge, torch.ones(len(knn_edge[0])).to(cur_X.device),
            #                                            [cur_X.shape[0], cur_X.shape[0]])
            #     cur_adj_sp = normalize_score_list[idx_f] * new_adj_sp + (
            #                 (1 + 1e-10) - normalize_score_list[idx_f]) * args.original_lamb * remain_data_adj_sp


            cur_output = self.gcn_agg(adj=cur_adj_sp, X=x[:, idx_f, :])
            result.append(cur_output)
        x = torch.cat(result, dim=-1)
        return x, scores_list

    @torch.no_grad()
    def get_embedding(self, args, x, edge_index, train_neighbors, y, train_mask, l2_normalize, block):

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x)
        x = self.bns[-1](x)
        disen_embedding, score_list = self.disentangle(args, x, edge_index, train_neighbors, y, train_mask, block)
        if l2_normalize:
            disen_embedding = F.normalize(disen_embedding, p=2, dim=1)

        return disen_embedding, score_list


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 num_layers=2, dropout=0.5, activation='relu'):
        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = 1 if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))
        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        return x

class ChannelDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 num_layers=2, dropout=0.5, activation='elu', ncaps=16):
        super().__init__()
        self.ncaps = ncaps
        self.mlps = nn.ModuleList()
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = in_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))
        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)
    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()
    def forward(self, ch_init):
        x=ch_init
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        ch_rec = self.mlps[-1](x)
        return ch_rec.view(ch_rec.shape[0], self.ncaps, -1)


def routing_layer_32(x, num_caps, nlayer, max_iter, neighbors):
    batch_size=500
    dev = x.device
    n, d, k, m = x.shape[0], x.shape[1], num_caps, len(neighbors[0])

    delta_d = int(d // k)
    _cache_zero_d = torch.zeros(1, d).to(dev)
    _cache_zero_k = torch.zeros(1, k).to(dev)
    final_chunks = []

    for nl in range(nlayer):
        if nl > 0:
            x = final_chunks[-1]  # Reuse the last final_chunks as input
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)  # Normalize and convert to smaller data type
        temp_z = torch.cat([x, _cache_zero_d], dim=0)
        final_chunks_batch = []

        for idx in range(0, neighbors.shape[0], batch_size):
            torch.cuda.empty_cache()  # Clear intermediate tensors

            batch_end = min(idx + batch_size, neighbors.shape[0])
            neigh = neighbors[idx:batch_end, :]
            chunk_size = neigh.shape[0]
            z = temp_z[neigh].view(chunk_size, m, k, delta_d)

            u = None
            for clus_iter in range(max_iter):
                if clus_iter == 0:
                    p = _cache_zero_k.expand(chunk_size * m, k).view(chunk_size, m, k)
                else:
                    p = torch.sum(z * u.view(chunk_size, 1, k, delta_d), dim=3)

                p = F.softmax(p, dim=2)
                u = torch.sum(z * p.view(chunk_size, m, k, 1), dim=1)

                u += x[idx:batch_end, :].view(chunk_size, k, delta_d)
                if clus_iter < max_iter - 1:
                    u = F.normalize(u, dim=2)  # Normalize and convert to smaller data type

            final_chunks_batch.append(u.view(chunk_size, d))

        final_chunks_batch = torch.cat(final_chunks_batch, dim=0)  # Convert to original data type
        final_chunks.append(final_chunks_batch)

    return final_chunks[-1]  # Return the last final_chunks

def routing_layer_16(x, num_caps, nlayer, max_iter, neighbors):
    batch_size=500
    dev = x.device
    n, d, k, m = x.shape[0], x.shape[1], num_caps, len(neighbors[0])

    delta_d = int(d // k)
    _cache_zero_d = torch.zeros(1, d).to(dev)
    _cache_zero_k = torch.zeros(1, k).to(dev)
    final_chunks = []

    # Convert to smaller data type
    x = x.to(torch.float16)
    _cache_zero_d = _cache_zero_d.to(torch.float16)
    _cache_zero_k = _cache_zero_k.to(torch.float16)

    for nl in range(nlayer):
        if nl > 0:
            x = final_chunks[-1]  # Reuse the last final_chunks as input
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d).to(torch.float16)  # Normalize and convert to smaller data type
        temp_z = torch.cat([x, _cache_zero_d], dim=0)
        final_chunks_batch = []

        for idx in range(0, neighbors.shape[0], batch_size):
            torch.cuda.empty_cache()  # Clear intermediate tensors

            batch_end = min(idx + batch_size, neighbors.shape[0])
            neigh = neighbors[idx:batch_end, :]
            chunk_size = neigh.shape[0]
            z = temp_z[neigh].view(chunk_size, m, k, delta_d)

            u = None
            for clus_iter in range(max_iter):
                if clus_iter == 0:
                    p = _cache_zero_k.expand(chunk_size * m, k).view(chunk_size, m, k)
                else:
                    p = torch.sum(z * u.view(chunk_size, 1, k, delta_d), dim=3)

                p = F.softmax(p, dim=2)
                u = torch.sum(z * p.view(chunk_size, m, k, 1), dim=1)

                u += x[idx:batch_end, :].view(chunk_size, k, delta_d)
                if clus_iter < max_iter - 1:
                    u = F.normalize(u, dim=2).to(torch.float16)  # Normalize and convert to smaller data type

            final_chunks_batch.append(u.view(chunk_size, d))

        final_chunks_batch = torch.cat(final_chunks_batch, dim=0).to(torch.float32)  # Convert to original data type
        final_chunks.append(final_chunks_batch)

    return final_chunks[-1]  # Return the last final_chunks



class MaskGAE(nn.Module):
    def __init__(
            self,
            args,
            encoder,
            edge_decoder,
            ch_decoder,
            mask,
            torch_generator
    ):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.ch_decoder = ch_decoder
        self.mask = mask
        self.torch_generator = torch_generator
        self.edge_loss_fn = ce_loss
        self.ch_loss_fn = setup_loss_fn(args.recon_alpha)
        self.negative_sampler = negative_sampling

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()

    def forward(self, args, x, edge_index, neighbors, y, train_mask, B):
        embedding, test_score_list= self.encoder(args, x, edge_index, neighbors, y, train_mask, B)
        return embedding

    def neigh_sampler_torch(self, args, num_nodes, edge_index, epoch):
        random.seed(epoch)
        neighbors = torch.zeros(1, args.nb_size).to(edge_index.device)

        first = edge_index[0]
        second = edge_index[1]
        for v in range(num_nodes):
            temp = second[(first == v).nonzero(as_tuple=True)[0]]
            if temp.shape[0] <= args.nb_size:
                shortage = args.nb_size - temp.shape[0]
                sampled_values = torch.cat(
                    (temp.reshape(1, -1), torch.IntTensor([-1]).repeat(shortage).reshape(1, -1).to(edge_index.device)),
                    1)
                neighbors = torch.cat((neighbors, sampled_values), dim=0)
            else:
                indice = random.sample(range(temp.shape[0]), args.nb_size)
                indice = torch.tensor(indice)
                sampled_values = temp[indice].reshape(1, -1)
                neighbors = torch.cat((neighbors, sampled_values), dim=0)
        # del first
        # del second
        return neighbors[1:].long()

    def train_epoch(
            self, args, train_data, optimizer, scheduler, batch_size, grad_norm, train_B, epoch):
        optimizer.zero_grad()
        x, edge_index, y, train_mask = train_data.x, train_data.edge_index, train_data.y, train_data.train_mask
        remaining_edges, masked_edges = self.mask(edge_index)
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(aug_edge_index, num_nodes=train_data.num_nodes,
                                          num_neg_samples=masked_edges.view(2, -1).size(1), ).view_as(masked_edges)
        self.nhidden = args.encoder_out // args.ncaps
        self.disen_y = torch.arange(args.ncaps).long().unsqueeze(dim=0).repeat(train_data.num_nodes, 1).flatten().to(
            x.device)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,

        # remaining loss
        for perm in DataLoader(
                range(masked_edges.size(1)), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                generator=self.torch_generator):
            train_neighbors = self.neigh_sampler_torch(args, train_data.num_nodes, remaining_edges, epoch)
            z, score_list = self.encoder(args, x, remaining_edges, train_neighbors, y, train_mask, train_B)
            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            # ******************* loss for edge prediction *********************

            pos_out = self.edge_decoder(z, batch_masked_edges)
            neg_out = self.edge_decoder(z, batch_neg_edges)  # 계속 낮아짐

            edge_loss = self.edge_loss_fn(pos_out, neg_out)

            # *****************************************************************

            ch_masking_temp = z.view(z.shape[0], args.ncaps, -1)

            non_zero_indices = torch.nonzero(score_list).squeeze().tolist()

            if isinstance(non_zero_indices, int):
                non_zero_indices = [non_zero_indices]

            zero_indices = list(set(range(args.ncaps)) - set(non_zero_indices))
            if isinstance(zero_indices, int):
                zero_indices = [zero_indices]

            mask_init = torch.cat([ch_masking_temp[:, i, :] if i in zero_indices else torch.zeros_like(ch_masking_temp[:, i, :]) for i in range(args.ncaps)],
                               dim=-1)

            mask_recon = self.ch_decoder(mask_init)


            ch_init = ch_masking_temp[:, non_zero_indices, :]
            ch_rec = mask_recon[:, non_zero_indices, :]

            ch_loss = args.recon_alpha * self.ch_loss_fn(ch_rec, ch_init)
            loss = edge_loss + ch_loss
            # ******************************************************************

            loss.backward()

            if grad_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()
            scheduler.step()
        return loss.item(), len(torch.nonzero(score_list))

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size, worker_init_fn=seed_worker,
                               generator=self.torch_generator):  # 한번 돌음
            edge = edges[:, perm]

            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
            # preds += [self.edge_decoder.forward_disen(z, edge, disen_adj_list).squeeze().detach().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):

        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)  # 계속 낮아짐

        pred = torch.cat([pos_pred, neg_pred], dim=0)  # pred torch.Size([526])
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

