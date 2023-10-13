import os
import torch
import random
import numpy as np
import yaml
import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS, NELL, WebKB, CoraFull
import subprocess
import time

def get_gpu_memory_usage(line, device):
    device = int(device)
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], capture_output=True, text=True)
    output = result.stdout.strip().split('\n')
    memory_usages = [int(x) for x in output]
    return print(f"{line} GPU:{device} memory usage: {memory_usages[device]} MB")


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
    np.random.seed(cnt+int(time.time()))
    random.seed(cnt+int(time.time()))
    if args.dataset == 'Pubmed':
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.4, 0.7, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.3, 0.6, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.4, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(40, 80)), -1)
        args.knn_nb = round(np.random.choice(range(40, 80)), -1)
        args.nlayer = random.choice(np.arange(4, 9))
        args.max_iter = random.choice(np.arange(6, 11))
        args.hsic_lamb = round(np.random.uniform(2.0e-05,9.0e-05, 1)[0], 6)
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.3, 0.7, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.3, 0.8, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(0.00001, 0.001, 1)[0], 5)
        args.nodeclas_weight_decay = round(np.random.uniform(0.001, 0.1, 1)[0], 3)
    elif args.dataset == 'Photo':
        args.ncaps == 32
        args.hsic_lamb = round(np.random.uniform(3.0e-04, 7.5e-04, 1)[0], 6)
        args.encoder_out = 512 # np.random.choice([256, 512], 1, p=[0.1, 0.9])[0]
        args.encoder_dropout = round(np.random.uniform(0.5, 0.8, 1)[0], 2)
        args.decoder_hidden = 64 #np.random.choice([32, 64], 1, p=[0.1, 0.9])[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.3, 0.8, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.4, 1)[0], 2)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.knn_nb = round(np.random.choice(range(40, 80)), -1)
        args.link_lr_max = 0.005 # 0.01
        args.link_lr_min = 0.005 # 0.0005
        args.node_lr_max = 0.1
        args.node_lr_min = 0.001
        args.nlayer = random.choice(np.arange(5, 8))
        args.max_iter = random.choice(np.arange(6, 9))
        args.grad_norm = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
        args.original_lamb = round(np.random.uniform(0.3, 0.7, 1)[0], 2)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.2, 0.7, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(1.0e-06, 9.9e-05, 1)[0], 7)
        args.nodeclas_weight_decay = round(np.random.uniform(1.0e-06, 5.9e-05, 1)[0], 7)
    elif args.dataset == 'Computers':
        # if args.trial ==1:
        #     args.ncaps =16
        #     args.hsic_lamb = round(np.random.uniform(9.0e-05, 4.0e-04, 1)[0], 6)
        # else:
        #     args.ncpas=32
        #     args.hsic_lamb = round(np.random.uniform(6.5e-05, 8.5e-05, 1)[0], 5)
        args.ncpas = 32
        # args.hsic_lamb = round(np.random.uniform(1.5e-04, 3.5e-04, 1)[0], 5) # 256
        args.hsic_lamb = round(np.random.uniform(7.0e-05, 2.7e-04, 1)[0], 5) # 256
        args.encoder_out = 512 # np.random.choice([256, 512], 1, p=[0.2, 0.8])[0]
        args.encoder_dropout = round(np.random.uniform(0.4, 0.9, 1)[0], 2)
        args.decoder_hidden = 64 # np.random.choice([64, 32], 1, p=[0.8, 0.2])[0]
        args.decoder_layers = random.choice(np.arange(2, 5))
        args.decoder_dropout = round(np.random.uniform(0.15, 0.5, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 5))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.3, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(5, 8))
        args.max_iter = random.choice(np.arange(7, 11))
        args.link_lr_max= 0.01
        args.link_lr_min= 0.01
        args.node_lr_max= 0.1
        args.node_lr_min= 0.001
        args.grad_norm = np.random.choice([1.0, 0.0], 1, p=[0.8, 0.2])[0]
        args.alpha_l = np.random.choice([1, 3], 1, p=[0.5, 0.5])[0]
        args.original_lamb = round(np.random.uniform(0.4, 0.90, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.2, 0.6, 1)[0], 2)
        args.weight_decay = round(np.random.uniform(1.0e-06, 9.0e-05, 1)[0], 7)
        args.nodeclas_weight_decay = round(np.random.uniform(1.0e-06, 9.0e-05, 1)[0], 7)
    elif args.dataset == 'WikiCS':
        if args.trial==1:
            args.ncaps==16
            args.hsic_lamb = round(np.random.uniform(3.5e-04, 9.5e-04, 1)[0], 5)
        else:
            args.ncaps==32
            args.hsic_lamb = round(np.random.uniform(5.5e-05, 8.0e-05, 1)[0], 5)
        args.encoder_out = 512
        args.encoder_dropout = round(np.random.uniform(0.4, 0.8, 1)[0], 2)
        args.decoder_hidden = np.random.choice([64, 32], 1)[0]
        args.decoder_layers = random.choice(np.arange(2, 4))
        args.decoder_dropout = round(np.random.uniform(0.2, 0.6, 1)[0], 2)
        args.ch_decoder_layers = random.choice(np.arange(2, 4))
        args.ch_decoder_dropout = round(np.random.uniform(0.1, 0.5, 1)[0], 2)
        args.knn_nb = round(np.random.choice(range(30, 70)), -1)
        args.nb_size = round(np.random.choice(range(30, 70)), -1)
        args.nlayer = random.choice(np.arange(4, 10))
        args.max_iter = random.choice(np.arange(6, 13))
        args.grad_norm = 1.0
        args.original_lamb = round(np.random.uniform(0.2, 0.7, 1)[0], 1)
        args.new_lamb = 1.0
        args.recon_alpha = round(np.random.uniform(0.1, 0.3, 1)[0], 2)
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
    np.random.seed(cnt+int(time.time()))
    random.seed(cnt+int(time.time()))
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
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)
    configs = configs[args.dataset.lower()+str(args.trial)]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True) # 아마 해보자

#
# def load_dataset_large(dataset_name):
#     transform = T.Compose([T.ToUndirected()])
#     root = osp.join('~/public_data/pyg_data')
#
#     if dataset_name in ['arxiv']:
#         data = torch.load("/home/jongwon208/public_data/pyg_data/ogbn_arxiv/processed_data.pt").detach().cpu() # Data(num_nodes=169343, edge_index=[2, 2315598], x=[169343, 128], node_year=[169343, 1], y=[169343, 1], train_mask=[169343], val_mask=[169343], test_mask=[169343])
#         data = T.NormalizeFeatures()(data)
#         data.y = data.y.view(-1)
#         # from ogb.nodeproppred import PygNodePropPredDataset
#         # dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{dataset_name}')
#         # split_idx = dataset.get_idx_split()
#         # data = transform(dataset[0])
#         # data.train_mask = torch.tensor(
#         #     [True if i in split_idx['train'] else False for i in range(data.num_nodes)]).to(device)
#         # data.val_mask = torch.tensor(
#         #     [True if i in split_idx['valid'] else False for i in range(data.num_nodes)]).to(device)
#         # data.test_mask = torch.tensor(
#         #     [True if i in split_idx['test'] else False for i in range(data.num_nodes)]).to(device)
#     elif dataset_name in {'Cora', 'Citeseer', 'Pubmed'}:
#         dataset = Planetoid(root, dataset_name, transform=T.NormalizeFeatures())
#         data = transform(dataset[0])
#     elif dataset_name in {'Photo', 'Computers'}:
#         dataset = Amazon(root, dataset_name)
#         data = transform(dataset[0])
#         data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
#     elif dataset_name in {'CS', 'Physics'}:
#         dataset = Coauthor(root, dataset_name)
#         data = transform(dataset[0])
#         data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
#     elif dataset_name in {'WikiCS'}:
#         dataset = WikiCS(root=root)
#         data = transform(dataset[0])
#         data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
#     elif dataset_name in {'CoraFull'}:
#         dataset = CoraFull(root="/home/jongwon208/MaskGAE/mine_encoder_list/data", transform=T.NormalizeFeatures())
#         data = transform(dataset[0])
#         data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
#     else:
#         print('only arxiv in load_dataset_large')
#     return data

def load_dataset(dataset_name, device):
    transform = T.Compose([T.ToUndirected(), T.ToDevice(device)])
    root = osp.join('~/public_data/pyg_data')

    if dataset_name in ['arxiv']:
        data = torch.load("/home/jongwon208/public_data/pyg_data/ogbn_arxiv/processed_data.pt") # Data(num_nodes=169343, edge_index=[2, 2315598], x=[169343, 128], node_year=[169343, 1], y=[169343, 1], train_mask=[169343], val_mask=[169343], test_mask=[169343])
        # from ogb.nodeproppred import PygNodePropPredDataset
        # dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{dataset_name}')
        # split_idx = dataset.get_idx_split()
        # data = transform(dataset[0])
        # data.train_mask = torch.tensor(
        #     [True if i in split_idx['train'] else False for i in range(data.num_nodes)]).to(device)
        # data.val_mask = torch.tensor(
        #     [True if i in split_idx['valid'] else False for i in range(data.num_nodes)]).to(device)
        # data.test_mask = torch.tensor(
        #     [True if i in split_idx['test'] else False for i in range(data.num_nodes)]).to(device)
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
    elif dataset_name in {'Texas','Cornell','Wisconsin'}:
        dataset = WebKB(root, dataset_name, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data.train_mask = data.train_mask[:,0]
        data.val_mask = data.val_mask[:,0]
        data.test_mask = data.test_mask[:,0]

    else:
        raise ValueError(dataset_name)
        print('check dataset name')
    return data

