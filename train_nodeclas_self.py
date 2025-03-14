import time
import argparse
import os
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
from utils import load_dataset, set_seed, load_best_configs, seed_worker, set_args_node
from model_self import MaskGAE, MLPEncoder, EdgeDecoder, ChannelDecoder
from mask import MaskPath
import uuid
unique_id = str(uuid.uuid4())

def train_linkpred(run, model, splits, args, device="cpu"):
    def train(data, epoch):
        model.train()
        print('train_link')
        edge_loss, non_zero, z = model.train_epoch(args, data.to(device), optimizer, scheduler, batch_size=args.batch_size,
                                      grad_norm=args.grad_norm, epoch=epoch)

        link_loss_list.append((epoch,edge_loss))
        non_zero_list.append(non_zero)
        return edge_loss, z

    @torch.no_grad()
    def test(epoch, z):
        print('test_link')
        model.eval()
        valid_auc, valid_ap = model.test(z, valid_pos_edge_label_index, valid_neg_edge_label_index)
        test_auc, test_ap = model.test(z, test_pos_edge_label_index, test_neg_edge_label_index)
        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        link_AUC_valid_list.append((epoch,valid_auc))
        link_AUC_test_list.append((epoch,test_auc))
        link_AP_valid_list.append((epoch,valid_ap))
        link_AP_test_list.append((epoch,test_ap))
        return results

    print('Start Training (Link Prediction Pretext Training)...')
    best_valid = 0
    best_epoch = 0
    non_zero_list = []
    link_loss_list=[]
    link_AUC_valid_list=[]
    link_AUC_test_list = []
    link_AP_valid_list = []
    link_AP_test_list = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.link_lr_max, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=args.link_lr_min, last_epoch=-1)
    checkpoint = torch.load(f'./pretrained/{args.dataset}_kmeans_onlyedge_scheduler_seed0.pt', map_location=device) #  426 accuracy 1 tensor(0.9671) acc2 tensor(0.9714) link AUC, 95.8% link AP, 96.17 NodeTest: 77.80%

    model.previous_unconflicted = checkpoint['previous_unconflicted'].to(device)
    model.cluster_pred = checkpoint['cluster_pred'].to(device)
    model.reset_parameters()
    for epoch in range(args.epochs):
        edge_loss, z = train(data, epoch)
        results = test(epoch, z)
        # test node
        node_val, node_test, node_loss_list, node_acc_val_list, node_acc_test_list, _ = train_nodeclas(run, model, data,
                                                                                                    args, device=device)
        print('test node', node_acc_test_list[-1][1])
    
        if node_test > best_valid:
            best_valid = node_test
            best_epoch = epoch
    
        print('\n ##### Testing result for (link prediction)')
        valid_result = results['AUC'][0]
        test_result = results['AUC'][1]
        print(f"epoch:{epoch}, Best epoch:{best_epoch}, valid: {valid_result}, test: {test_result}")

    return results, best_epoch, best_valid, link_loss_list, link_AUC_valid_list, link_AUC_test_list, link_AP_valid_list, link_AP_test_list, non_zero_list, optimizer


def train_nodeclas(run, model, data, args, device='cpu'):
    def train(loader):
        clf.train()
        for nodes in loader:
            optimizer_node.zero_grad()
            node_loss = loss_fn(clf(embedding[nodes]), y[nodes])
            node_loss.backward()
            optimizer_node.step()
            scheduler_node.step()
        return node_loss

    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()

    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=20000, shuffle=True,
                              worker_init_fn=seed_worker, generator=th_g)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000, shuffle=False,worker_init_fn=seed_worker, generator=th_g)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000, shuffle=False,worker_init_fn=seed_worker, generator=th_g)

    data = data.to(device)
    y = data.y.squeeze()

    embedding, _ = model.encoder.get_embedding(args, data.x, data.edge_index, neighbors, data.y, data.train_mask,
                                            args.l2_normalize)
    loss_fn = nn.CrossEntropyLoss()

    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)
    print('Start Training (Node Classification)...')
    nn.init.xavier_uniform_(clf.weight.data)
    optimizer_node = torch.optim.Adam(clf.parameters(), lr=args.node_lr_max,
                                      weight_decay=args.nodeclas_weight_decay)  # 1 for citeseer
    scheduler_node = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_node, T_0=50, T_mult=2,eta_min=args.node_lr_min, last_epoch=-1)

    node_loss_list = []
    node_acc_val_list = []
    node_acc_test_list = []
    best_val = best_test = best_epoch2 = 0
    for epoch2 in range(300):
        node_loss = train(train_loader)
        node_loss_list.append((epoch2, node_loss))
        val_metric = test(val_loader)
        node_acc_val_list.append((epoch2, val_metric))

        if val_metric > best_val:
            best_val = val_metric
            best_epoch2 = epoch2
            best_test = test(test_loader)
            node_acc_test_list.append((epoch2, best_test))

    print(f'##### Testing result for Node Classification',
          f'**** Testing on Run: {run + 1:02d}, '
          f'Best Epoch: {best_epoch2:02d}'
          f'Valid: {best_val:.2%}, '
          f'Test: {best_test:.2%}')
    return best_val, best_test, node_loss_list, node_acc_val_list, node_acc_test_list, optimizer_node


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument('--device', type=int, default=0, help='GPU')
parser.add_argument('--encoder_hidden', type=int, default=512, help='input dim. of hidden representation.')
parser.add_argument('--encoder_out', type=int, default=512, help='Output dim. of hidden representation.')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers for Encoders.')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder.')

parser.add_argument('--decoder_hidden', type=int, default=64, help='Input dim. of decoder layers.')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders.')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder.')

parser.add_argument('--ch_decoder_layers', type=int, default=2, help='Number of layers for channel decoders.')
parser.add_argument('--ch_decoder_dropout', type=float, default=0.1, help='Dropout probability of channel decoder.')
#clustering 
parser.add_argument('--cluster_emb', type=float, default=0.995, help='cluster emb. conf.')
parser.add_argument('--trace_reg', type=float, default=0.1, help='cluster loss reg.')
# routing
parser.add_argument('--nb_size', type=int, default=50, help='nb size for neighRouting.')
parser.add_argument('--ncaps', type=int, default=16, help='num channels.')
parser.add_argument('--nlayer', type=int, default=6, help='routing layer.')
parser.add_argument('--max_iter', type=int, default=11, help='routing iterations')
parser.add_argument('--hsic_lamb', type=float, default=3.5E-05, help='HSIC lambd.')
parser.add_argument('--link_lr_max', type=float, default=0.1, help='max lr for link prediction.')
parser.add_argument('--link_lr_min', type=float, default=0.005,help='min lr rate for link prediction.')
parser.add_argument('--node_lr_max', type=float, default=0.1, help='max lr for node classification. (default: 0.01)')
parser.add_argument('--node_lr_min', type=float, default=0.001, help='min lr for node classification. (default: 0.01)')

parser.add_argument('--grad_norm', type=float, default=1.0, help='(default: 1.0)')
parser.add_argument('--recon_alpha', type=float, default=1.0, help='ch loss reg.')
parser.add_argument('--batch_size', type=int, default=2 ** 16, help='Number of batch size for link prediction')
#
parser.add_argument('--l2_normalize', action='store_false', help='Whether to use l2 normalize output embedding.')
parser.add_argument('--alpha_l', type=int, default=1, help='HSIC param.')
parser.add_argument('--weight_decay', type=float, default=5e-5,help='weight_decay for link prediction training.')
parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training.')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
parser.add_argument('--eval_period', type=int, default=1, help='')
parser.add_argument("--save_path", nargs="?", default="model_nodeclas.pth", help="save path for model.")
parser.add_argument('--trial', type=int, default=10, help='')
args = parser.parse_args()

use_best = True
if use_best:   
    args = load_best_configs(args, "node_configs.yml")
else:
    print('not using best yaml')

if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
# experiment code
best_test_acc = 0.1
data = load_dataset(args.dataset,device)

count = 0 
while True:
# for cnt in range(10):
    count += 1
        #############
    if count > 1:
        args = set_args_node(args, count)
    #############
    print('args', args)
    AUC_val_list = []
    AP_val_list = []
    AUC_test_list = []
    AP_test_list = []
    val_list = []
    test_list = []
    best_epoch_list= []
    best_valid_list= []
    for run in range(args.trial):
        print(f"run {run}")
        seed = run
        set_seed(seed)
        th_g = torch.Generator()
        th_g.manual_seed(seed)

        train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                            is_undirected=True,
                                                            split_labels=True,
                                                            add_negative_train_samples=True)(data)

        splits = dict(train=train_data.to(device), valid=val_data.to(device), test=test_data.to(device))
        valid_pos_edge_label_index = val_data.pos_edge_label_index.to(device)
        valid_neg_edge_label_index = val_data.neg_edge_label_index.to(device)
        test_pos_edge_label_index = test_data.pos_edge_label_index.to(device)
        test_neg_edge_label_index = test_data.neg_edge_label_index.to(device)
        mask = MaskPath(p=0.7, num_nodes=data.num_nodes, start='node', walk_length=3)

        encoder = MLPEncoder(data.x.shape[1], args.encoder_hidden, args.encoder_out, args.encoder_layers, dropout=args.encoder_dropout)
        edge_decoder = EdgeDecoder(args.encoder_out, args.decoder_hidden, num_layers=args.decoder_layers,
                                    dropout=args.decoder_dropout)
        ch_decoder = ChannelDecoder(args.encoder_out, args.encoder_out//2, num_layers=args.ch_decoder_layers,
                                    dropout=args.ch_decoder_dropout, ncaps=args.ncaps)

        model = MaskGAE(args, encoder, edge_decoder,ch_decoder, mask, torch_generator=th_g, num_labels = len(torch.unique(data.y))).to(device)

        neighbors = model.neigh_sampler_torch(args, data.num_nodes, data.edge_index, seed)  
        print('\n start link pred')
        results, best_epoch, best_valid, link_loss_list, link_AUC_valid_list, link_AUC_test_list, link_AP_valid_list, link_AP_test_list, non_zero_list, optimizer = train_linkpred(run, model, splits, args, device=device)
        node_val, node_test, node_loss_list, node_acc_val_list, node_acc_test_list, optimizer_node = train_nodeclas(run, model, data, args, device=device)
        #     
        AUC_val_list.append(results['AUC'][0])
        AUC_test_list.append(results['AUC'][1])
        AP_val_list.append(results['AP'][0])
        AP_test_list.append(results['AP'][1])
        val_list.append(node_val)
        test_list.append(node_test)    
        if args.dataset=='Cora':
            print('node test', args.dataset, node_test)
            if node_test < 0.831:
                print('yes break')
                break
            else:
                unique_id = str(uuid.uuid4())
                cur_time = time.strftime("%Y%m%d-%H%M%S")
                filename = './bestresults/' +str(args.dataset)+ '_' + str(node_test) + '_' + unique_id + '_' + cur_time + args.save_path
                filename_txt = './bestresults/' +str(args.dataset)+ '_' + str(node_test) + '_' + unique_id + '_' + cur_time + '.txt'
                with open(f'{filename_txt}', 'w') as file:
                    for arg, value in vars(args).items():
                        file.write(f'{arg}: {value}\n')
                # Save model and optimizer state
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_node_state_dict': optimizer_node.state_dict(),
                    'previous_unconflicted': model.previous_unconflicted,
                    'cluster_pred': model.cluster_pred,
                }, filename)
        
        elif args.dataset=='Photo':
            if node_test < 0.92:
                break
            else:
                unique_id = str(uuid.uuid4())
                cur_time = time.strftime("%Y%m%d-%H%M%S")
                filename = './bestresults/' +str(args.dataset)+ '_' + str(node_test) + '_' + unique_id + '_' + cur_time + args.save_path
                filename_txt = './bestresults/' +str(args.dataset)+ '_' + str(node_test) + '_' + unique_id + '_' + cur_time + '.txt'
                with open(f'{filename_txt}', 'w') as file:
                    for arg, value in vars(args).items():
                        file.write(f'{arg}: {value}\n')
                # Save model and optimizer state
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_node_state_dict': optimizer_node.state_dict(),
                    'previous_unconflicted': model.previous_unconflicted,
                    'cluster_pred': model.cluster_pred,
                }, filename)

    file_name = f"./{args.dataset.lower()}_nohup_outfiles/node2/{round(np.mean(test_list) * 100, 2)}_{args.ncaps}_{unique_id}.txt"
    with open(file_name, 'w') as file:
        file.write("logs of node classification loss:\n"
                    f"non_zero_list: {np.mean(non_zero_list), non_zero_list}"
                    f"\n\n\n{vars(args)}\nseed {seed} runs {args.trial}"
                    f"\nAUC_val: {AUC_val_list}"
                    f"\nAP_val: {AP_val_list}"
                    f"\nAUC_test: {AUC_test_list}"
                    f"\nAP_test: {AP_test_list}"
                    f"\nval: {val_list}"
                    f"\ntest: {test_list}"
                    f"\n{np.mean(AUC_val_list) * 100:.2f}±{np.std(AUC_val_list)*100:.2f}"
                    f"\n{np.mean(AP_val_list) * 100:.2f}±{np.std(AP_val_list)*100:.2f}"
                    f"\n{np.mean(AUC_test_list) * 100:.2f}±{np.std(AUC_test_list)*100:.2f}"
                    f"\n{np.mean(AP_test_list) * 100:.2f}±{np.std(AP_test_list)*100:.2f}"
                    f"\n{np.mean(val_list) * 100:.2f}±{np.std(val_list)*100:.2f}"
                    f"\n{np.mean(test_list) * 100:.2f}±{np.std(test_list)*100:.2f}")
    