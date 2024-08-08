import os
import torch
import random
import numpy as np
import yaml
import torch_geometric.transforms as T
import os.path as osp
import os
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS, NELL, WebKB, CoraFull, WikipediaNetwork
import subprocess
import time
# import dgl
# from sklearn.preprocessing import StandardScaler


from scipy.sparse.linalg import norm as sparse_norm


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def save_best_model(args, model, optimizer, node_test, unique_id, cur_best, cur_best_txt):

    try:
        os.remove(cur_best)
        os.remove(cur_best_txt)
    except:
        pass
    best_time = str(time.strftime("%Y%m%d-%H%M%S"))
    filename = './pretrained/bestresults/' + str(args.dataset) + '_' + str(args.shot) + '_' + str(
        node_test) + '_' + unique_id + '_' + best_time + args.save_path
    filename_txt = './pretrained/bestresults/' + str(
        args.dataset) + '_' + str(args.shot)+ '_' + str(node_test) + '_' + unique_id + '_' + best_time + '.txt'

    with open(f'{filename_txt}', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')
    # Save model and optimizer state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'previous_unconflicted': model.previous_unconflicted,
        'cluster_pred': model.cluster_pred,
    }, filename)
    return filename, filename_txt

class Sampler:
    def __init__(self, features, adj, **kwargs):
        allowed_kwargs = {'input_dim', 'layer_sizes', 'device'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, \
                'Invalid keyword argument: ' + kwarg
        self.input_dim = kwargs.get('input_dim', 1)
        self.layer_sizes = kwargs.get('layer_sizes', [1])
        self.scope = kwargs.get('scope', 'test_graph')
        self.device = kwargs.get('device', torch.device("cpu"))
        self.num_layers = len(self.layer_sizes)
        self.adj = adj
        self.features = features
        self.train_nodes_number = self.adj.shape[0]

    def sampling(self, v_indices):
        raise NotImplementedError("sampling is not implimented")

    def _change_sparse_to_tensor(self, adjs):
        new_adjs = []
        for adj in adjs:
            new_adjs.append(
                sparse_mx_to_torch_sparse_tensor(adj).to(self.device))
        return new_adjs

import scipy.sparse as sp
class Sampler_FastGCN(Sampler):
    def __init__(self, features, adj, **kwargs):
        super().__init__(features, adj, **kwargs)
        col_norm = sparse_norm(adj, axis=0)
        self.probs = col_norm / np.sum(col_norm)

    def sampling(self, v):
        all_support = [[]] * self.num_layers

        cur_out_nodes = v
        for layer_index in range(self.num_layers-1, -1, -1):
            cur_sampled, cur_support = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_support[layer_index] = cur_support
            cur_out_nodes = cur_sampled

        all_support = self._change_sparse_to_tensor(all_support)
        sampled_X0 = self.features[cur_out_nodes]
        return sampled_X0, all_support, 0

    def _one_layer_sampling(self, v_indices, output_size):

        support = self.adj.tocsr()[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   output_size, True, p1)

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support



def find_number_with_no_remainder(N):
    min_remainder = float('inf')
    number_with_no_remainder = None
    for divisor in range(20, 200):
        remainder = N % divisor
        if remainder == 0:
            return torch.tensor(divisor)  # Found a number that divides N evenly with no remainder
        elif remainder < min_remainder:
            min_remainder = remainder
            number_with_no_remainder = divisor

    return torch.tensor(number_with_no_remainder)



def set_args_node(args, cnt):
    np.random.seed(cnt+int(time.time())% 10000 )
    random.seed(cnt+int(time.time())% 10000 )
    if args.dataset == 'Arxiv':
        args.encoder_out = 512
        args.link_lr_max = np.random.choice([0.0005, 0.005], 1)[0]
        args.encoder_dropout = round(np.random.uniform(0.1, 0.3, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.0, 0.2, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.6, 1)[0], 2)
        args.nb_size = 50 #round(np.random.choice(range(15, 30)), -1)
        args.knn_nb = 50 # round(np.random.choice(range(15, 30)), -1)
        args.nlayer = random.choice(np.arange(3, 5))
        args.max_iter = random.choice(np.arange(5, 8))
        if cnt <= 1000:
            args.hsic_lamb = round(np.random.uniform(1.0e-05, 1.5e-05, 1)[0], 7)
        elif 1000 < cnt <= 10000:
            args.hsic_lamb = round(np.random.uniform(8.0e-06, 9.0e-06, 1)[0], 7)
        else:
            args.hsic_lamb = round(np.random.uniform(8.0e-06, 1.5e-05, 1)[0], 7)
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.2, 0.9, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.10, 1.0, 1)[0], 2)
        args.weight_decay = 0#round(np.random.uniform(0.00001, 0.001, 1)[0], 6)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00001, 0.0001, 1)[0], 5)
        args.alpha_l = 1
        args.epochs = np.random.choice([100, 150], 1)[0]
    elif args.dataset in ['Chameleon','Texas','Wisconsin','Cornell']:
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.6, 0.9, 1)[0], 2)
        args.decoder_hidden = 32  # np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.1, 0.4, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.6, 1)[0], 2)
        args.nlayer = random.choice(np.arange(5, 8))
        args.max_iter = random.choice(np.arange(6, 12))
        args.hsic_lamb = round(np.random.uniform(8.0e-04, 4.5e-03, 1)[0], 6)
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.2, 0.9, 1)[0], 1)
        args.new_lamb = 1.0
        if cnt % 2 == 0:
            args.recon_alpha = round(np.random.uniform(0.0001, 0.01, 1)[0], 5)
        else:
            args.recon_alpha = round(np.random.uniform(0.01, 1.0, 1)[0], 5)
        args.weight_decay = round(np.random.uniform(0.00001, 0.0002, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.0001, 0.01, 1)[0], 4)
        args.alpha_l = random.randint(1, 3)

    elif args.dataset == 'Cora':
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.65, 0.9, 1)[0], 2)
        args.decoder_hidden = 32# np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.2, 0.5, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.2, 0.8, 1)[0], 2)
        # args.nb_size = round(np.random.choice(range(30, 70)), -1)
        # args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(3, 8))
        args.max_iter = random.choice(np.arange(6, 12))
        args.hsic_lamb = round(np.random.uniform(8.0e-04, 4.0e-03, 1)[0],4)
        # args.hsic_lamb = round(np.random.uniform(9.0e-05, 7.5e-04, 1)[0], 6)
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.2, 0.9, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.0001, 0.009, 1)[0], 4)
        # args.weight_decay = round(np.random.uniform(0.00001, 0.0002, 1)[0], 4)
        args.nodeclas_weight_decay = round(np.random.uniform(0.002, 0.007, 1)[0], 4)
        args.alpha_l = 1 # random.randint(1, 3)
        # args.mask_ratio = round(np.random.uniform(0.6, 0.8, 1)[0], 1)
        # args.epoch = np.random.randint(40, 60)
    elif args.dataset == 'Citeseer':
        args.encoder_out = 512
        args.decoder_hidden = 32  # np.random.choice([64, 32], 1)[0]
        args.encoder_layers = 1 #round(np.random.choice(np.arange(1, 3)), 2)
        args.encoder_dropout = round(np.random.uniform(0.65, 0.85, 1)[0], 2)
        args.decoder_layers = round(np.random.choice(np.arange(3, 7)), 2)
        args.decoder_dropout = round(np.random.uniform(0.10, 0.3, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(1, 3))
        args.ch_decoder_dropout = round(np.random.uniform(0.2, 0.65, 1)[0], 2)
        # args.nb_size = round(np.random.choice(range(30, 70)), -1)
        # args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        # args.nlayer = random.choice(np.arange(4, 9))
        # args.max_iter = random.choice(np.arange(6, 11))
        args.hsic_lamb = round(np.random.uniform(9.5e-07, 4.0e-06, 1)[0], 7)
        args.grad_norm = 1  # np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.l2_normalize = True
        args.original_lamb = round(np.random.uniform(0.6, 0.8, 1)[0], 2)
        args.new_lamb = 1.0
        # if cnt%2==0:
        #     args.recon_alpha = round(np.random.uniform(0.00001, 0.001, 1)[0], 5)
        # else:
        if cnt % 2 == 0:
            args.recon_alpha = round(np.random.uniform(0.0001, 0.01, 1)[0], 5)
        else:
            args.recon_alpha = round(np.random.uniform(0.01, 1.0, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.00001, 0.00040, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.05, 0.7, 1)[0], 2)
        args.alpha_l = random.randint(1, 3)
        # args.epoch = np.random.randint(80, 120)  # Random value between 20 and 300, inclusive

    elif args.dataset == 'Pubmed':
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.4, 0.6, 1)[0], 2)
        args.decoder_hidden = 32#np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.35, 0.6, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(3, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.3, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(40, 80)), -1)
        args.knn_nb = round(np.random.choice(range(40, 80)), -1)
        args.nlayer = random.choice(np.arange(4, 9))
        args.max_iter = random.choice(np.arange(6, 11))
        args.hsic_lamb = round(np.random.uniform(2.0e-05,3.0e-05, 1)[0], 6)
        args.grad_norm = 1#np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.3, 0.5, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.5, 0.8, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.0008, 0.0015, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.0005, 0.0015, 1)[0], 5)
    elif args.dataset == 'Photo':
        args.ncaps == 32
        args.hsic_lamb = round(np.random.uniform(1.0e-04, 2.0e-03, 1)[0], 5) 
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.5, 0.75, 1)[0], 2)
        args.decoder_hidden = 64 
        args.decoder_layers = random.choice(np.arange(3, 6))
        args.decoder_dropout = round(np.random.uniform(0.2, 0.5, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(3, 6))
        args.ch_decoder_dropout = round(np.random.uniform(0.25, 0.5, 1)[0], 2)
        args.link_lr_max = 0.005 # 0.01
        args.link_lr_min = 0.005 # 0.0005
        args.node_lr_max = 0.1
        args.node_lr_min = 0.001
        args.nlayer = random.choice(np.arange(5, 8))
        args.max_iter = random.choice(np.arange(6, 10))
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.3, 0.6, 1)[0], 2)
        args.new_lamb = 1.0
        if cnt % 2 == 0:
            args.recon_alpha = round(np.random.uniform(0.0001, 0.01, 1)[0], 5)
        else:
            args.recon_alpha = round(np.random.uniform(0.01, 1.0, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(8.0e-06, 6.0e-05, 1)[0], 7)
        args.nodeclas_weight_decay = round(np.random.uniform(8.0e-06, 5.0e-05, 1)[0], 7)
        args.cluster_emb = round(np.random.uniform(0.900, 0.999, 1)[0], 3)
        args.tace = round(np.random.uniform(0.001, 1.0, 1)[0], 3)
    elif args.dataset == 'Computers':
        args.ncpas = 32
        args.hsic_lamb = round(np.random.uniform(9.7e-5, 2.0e-03, 1)[0], 6) # 1.7e-04:27, 5.7e-04: 18, 1.7e-03:5
        args.encoder_out = 512 # np.random.choice([256, 512], 1, p=[0.2, 0.8])[0]
        args.encoder_dropout = round(np.random.uniform(0.4, 0.7, 1)[0], 2)
        args.decoder_hidden = 64 # np.random.choice([64, 32], 1, p=[0.8, 0.2])[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.15, 0.5, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 4))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.2, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(4, 7))
        args.max_iter = random.choice(np.arange(6, 9))
        args.link_lr_max= 0.01
        args.link_lr_min= 0.001
        args.node_lr_max= 0.1
        args.node_lr_min= 0.001
        args.grad_norm = 1.0
        args.alpha_l = np.random.choice([1, 2, 3], 1, p=[1/3, 1/3, 1/3])[0]
        args.original_lamb = round(np.random.uniform(0.4, 0.90, 1)[0], 1)
        args.new_lamb = 1.0
        if cnt%2==0:
            args.recon_alpha = round(np.random.uniform(0.2, 0.4, 1)[0], 2)
        else:
            args.recon_alpha = round(np.random.uniform(0.0001, 0.01, 1)[0], 4)
        args.weight_decay = round(np.random.uniform(7.0e-05, 1.0e-04, 1)[0], 6)
        args.nodeclas_weight_decay = round(np.random.uniform(6.0e-05, 1.0e-04, 1)[0], 6)
        args.cluster_emb = round(np.random.uniform(0.900, 0.999, 1)[0], 3)
        args.tace = round(np.random.uniform(0.001, 1.0, 1)[0], 3)

    elif args.dataset == 'WikiCS':
        args.ncaps==32
        args.hsic_lamb = round(np.random.uniform(3.0e-05, 6.0e-05, 1)[0], 6)
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.4, 0.7, 1)[0], 2)
        args.decoder_hidden = 32 # np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.6, 0.85, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(3, 6))
        args.ch_decoder_dropout = round(np.random.uniform(0.15, 0.5, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(4, 8))
        args.max_iter = random.choice(np.arange(4, 10))
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.2, 0.7, 1)[0], 1)
        args.new_lamb = 1.0
        if cnt % 2 == 0:
            args.recon_alpha = round(np.random.uniform(0.1, 0.3, 1)[0], 2)
        else:
            args.recon_alpha = round(np.random.uniform(0.0001, 0.01, 1)[0], 4)
        args.weight_decay = round(np.random.uniform(6.0e-05, 2.0e-04, 1)[0], 6)
        args.nodeclas_weight_decay = round(np.random.uniform(1.0e-05, 8.0e-05, 1)[0], 6)
    elif args.dataset == 'CoraFull':
        args.encoder_out = 256
        args.encoder_dropout = round(np.random.uniform(0.6, 0.9, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.3, 0.9, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 4))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.9, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(40, 70)), -1)
        args.nlayer = random.choice(np.arange(4, 10))
        args.max_iter = random.choice(np.arange(6, 11))
        args.hsic_lamb = round(np.random.uniform(4.9e-07, 9.9e-07, 1)[0], 8)
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.1, 0.9, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.1, 1.0, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.00001, 0.1, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00001, 0.01, 1)[0], 5)
    elif args.dataset == 'CS':
        args.encoder_out = np.random.choice([256, 512], 1, p=[0.2, 0.8])[0]
        args.encoder_layers = 1
        args.encoder_dropout = round(np.random.uniform(0.2, 0.7, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(1, 4))
        args.decoder_dropout = round(np.random.uniform(0.1, 0.6, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(2, 5))
        args.max_iter = random.choice(np.arange(3, 7))
        args.hsic_lamb = round(np.random.uniform(1.0e-05, 5.0e-05, 1)[0], 6)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.1, 1.0, 1)[0], 1)
        args.new_lamb = 1.0
        args.weight_decay = round(np.random.uniform(0.00001, 0.001, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00001, 0.1, 1)[0], 3)
    elif args.dataset == 'Physics':
        args.encoder_out = 512
        args.encoder_layers = random.choice(np.arange(1, 3))
        args.encoder_dropout = round(np.random.uniform(0.2, 0.9, 1)[0], 2)
        args.decoder_hidden = np.random.choice([128, 64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(1, 5))
        args.decoder_dropout = round(np.random.uniform(0.1, 0.9, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(2, 5))
        args.max_iter = random.choice(np.arange(3, 7))
        args.hsic_lamb = round(np.random.uniform(1.9e-06, 4.9e-06, 1)[0], 5)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.1, 1.0, 1)[0], 1)
        args.weight_decay = round(np.random.uniform(0.00001, 0.001, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00057, 0.057, 1)[0], 2)
    else:
        print('check dataset again')
    return args

def set_args_link(args, cnt):
    np.random.seed(cnt+int(time.time())% 10000 )
    random.seed(cnt+int(time.time())% 10000 )
    if args.dataset == 'Pubmed':
        args.encoder_out = np.random.choice([256, 512], 1, p=[0.5, 0.5])[0]
        args.encoder_dropout = round(np.random.uniform(0.2, 0.6, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.3, 0.6, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.7, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(40, 80)), -1)
        args.knn_nb = round(np.random.choice(range(40, 80)), -1)
        args.nlayer = random.choice(np.arange(4, 9))
        args.max_iter = random.choice(np.arange(6, 11))
        args.hsic_lamb = round(np.random.uniform(4.0e-05,1.0e-04, 1)[0], 6)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.4, 1.0, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.2, 0.8, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.00001, 0.0001, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.001, 0.01, 1)[0], 3)
    elif args.dataset == 'Photo':
        args.encoder_out = np.random.choice([256, 512], 1, p=[0.1, 0.9])[0]
        args.encoder_dropout = round(np.random.uniform(0.4, 0.9, 1)[0], 2)
        args.decoder_hidden = np.random.choice([128, 64], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.1, 0.6, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.5, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.knn_nb = round(np.random.choice(range(40, 80)), -1)
        args.nlayer = random.choice(np.arange(6, 10))
        args.max_iter = random.choice(np.arange(8, 13))
        args.hsic_lamb = round(np.random.uniform(2.0e-04, 9.0e-04, 1)[0], 4)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.1, 1.0, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.1, 1.0, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.000005, 0.0005, 1)[0], 6)
        args.nodeclas_weight_decay = round(np.random.uniform(0.0005, 0.05, 1)[0], 4)
    elif args.dataset == 'Computers':
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.3, 0.8, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.2, 0.7, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.5, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(5, 9))
        args.max_iter = random.choice(np.arange(7, 13))
        args.hsic_lamb = round(np.random.uniform(9.0e-05, 3.0e-04, 1)[0], 5)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.1, 1.0, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.1, 1.0, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.00001, 0.001, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.0001, 0.1, 1)[0], 4)
    elif args.dataset == 'WikiCS':
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.2, 0.8, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.2, 0.8, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 4))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.9, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(4, 10))
        args.max_iter = random.choice(np.arange(6, 13))
        args.hsic_lamb = round(np.random.uniform(3.5e-04, 9.5e-04, 1)[0], 5)
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.1, 1.0, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.1, 1.0, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.00001, 0.1, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00001, 0.01, 1)[0], 5)
    elif args.dataset == 'CoraFull':
        args.encoder_out = 256
        args.encoder_dropout = round(np.random.uniform(0.6, 0.9, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.3, 0.9, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 4))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.9, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(40, 70)), -1)
        args.nlayer = random.choice(np.arange(4, 10))
        args.max_iter = random.choice(np.arange(6, 11))
        args.hsic_lamb = round(np.random.uniform(5.9e-07, 9.9e-07, 1)[0], 8)
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.1, 0.9, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.1, 1.0, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.00001, 0.1, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00001, 0.01, 1)[0], 5)
    elif args.dataset == 'CS':
        args.encoder_out = np.random.choice([256, 512], 1, p=[0.2, 0.8])[0]
        args.encoder_layers = 1
        args.encoder_dropout = round(np.random.uniform(0.2, 0.7, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(1, 4))
        args.decoder_dropout = round(np.random.uniform(0.1, 0.6, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(2, 5))
        args.max_iter = random.choice(np.arange(3, 7))
        args.hsic_lamb = round(np.random.uniform(1.0e-05, 5.0e-05, 1)[0], 6)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.1, 1.0, 1)[0], 1)
        args.new_lamb = 1.0
        args.weight_decay = round(np.random.uniform(0.00001, 0.001, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00001, 0.1, 1)[0], 3)
    elif args.dataset == 'Physics':
        args.encoder_out = 512
        args.encoder_layers = random.choice(np.arange(1, 3))
        args.encoder_dropout = round(np.random.uniform(0.2, 0.9, 1)[0], 2)
        args.decoder_hidden = np.random.choice([128, 64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(1, 5))
        args.decoder_dropout = round(np.random.uniform(0.1, 0.9, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(2, 5))
        args.max_iter = random.choice(np.arange(3, 7))
        args.hsic_lamb = round(np.random.uniform(1.9e-06, 4.9e-06, 1)[0], 5)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.1, 1.0, 1)[0], 1)
        args.weight_decay = round(np.random.uniform(0.00001, 0.001, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.00057, 0.057, 1)[0], 2)
    else:
        print('check dataset again')
    np.random.seed(0)
    random.seed(0)
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataset.lower()]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True) # 아마 해보자

# def scale_feats(x):
#     scaler = StandardScaler()
#     feats = x.numpy()
#     scaler.fit(feats)
#     feats = torch.from_numpy(scaler.transform(feats)).float()
#     return feats


def load_dataset(dataset_name, device):
    transform = T.Compose([T.ToUndirected(), T.ToDevice(device)])
    root = osp.join('~/public_data/pyg_data')
    if dataset_name in {'Arxiv'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-arxiv')
        data = transform(dataset[0])
        split_idx = dataset.get_idx_split()
        data.train_mask = split_idx['train']
        data.val_mask = split_idx['valid']
        data.test_mask = split_idx['test']

    elif dataset_name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, dataset_name, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
    elif dataset_name in {'Photo', 'Computers'}:
        dataset = Amazon(root, dataset_name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif dataset_name in {'CS', 'Physics'}:
        dataset = Coauthor(root, dataset_name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif dataset_name in {'WikiCS'}:
        dataset = WikiCS(root=root)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif dataset_name in {'NELL'}:
        dataset = NELL(root=root)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        data.x=data.x.to_dense()
    elif dataset_name in {'CoraFull'}:
        dataset = CoraFull(root="/home/jongwon208/MaskGAE/mine_encoder_list/data", transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        # dataset = CitationFull(root="/home/jongwon208/MaskGAE/mine_encoder_list/data",name= "cora", transform=T.NormalizeFeatures())
        # data = transform(dataset[0])
    elif dataset_name in {'Chameleon'}:
        print('chameleon dataaset load')
        dataset = WikipediaNetwork(root, dataset_name, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    elif dataset_name in {'Texas','Cornell','Wisconsin'}:
        print('texas')
        dataset = WebKB(root, dataset_name, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data.train_mask = data.train_mask[:,0]
        data.val_mask = data.val_mask[:,0]
        data.test_mask = data.test_mask[:,0]

    else:
        raise ValueError(dataset_name)
        print('check dataset name')
    return data

