import argparse
import torch
import torch_geometric.transforms as T
import warnings
warnings.filterwarnings(action='ignore')
from utils import load_dataset, set_seed, load_best_configs
from model_pretrain_edge import MaskGAE_pretrain, MLPEncoder, EdgeDecoder
from mask import MaskPath
import uuid
unique_id = str(uuid.uuid4())

def train_linkpred(model, splits, args, device="cpu"):
    def train(data, epoch):
        model.train()
        print('train_link')
        edge_loss, non_zero, z = model.train_epoch(args, data.to(device), optimizer, scheduler, batch_size=args.batch_size,
                                      grad_norm=args.grad_norm, epoch=epoch)
        return edge_loss, z

    @torch.no_grad()
    def test(epoch, splits, z):
        print('test_link')
        model.eval()
        # z = model(args, splits['train'].x, splits['train'].edge_index, train_neighbors, splits['train'].y, splits['train'].train_mask, train_B)
        valid_auc, valid_ap = model.test(z, valid_pos_edge_label_index, valid_neg_edge_label_index)
        test_auc, test_ap = model.test(z, test_pos_edge_label_index, test_neg_edge_label_index)
        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    save_path = args.save_path

    print('Start Training (Link Prediction Pretext Training)...')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.link_lr_max, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=args.link_lr_min, last_epoch=-1)
    model.reset_parameters()

    best_epoch = 0 
    cnt_wait = 0 
    for epoch in range(args.epochs):
        edge_loss, z = train(data, epoch)
        results = test(epoch, splits, z)

        valid_result = results['AUC'][0] 
        test_result = results['AUC'][1]
        print('\n ##### Testing result for (link prediction)')
        print(f"epoch:{epoch}, Best epoch:{best_epoch}, valid: {valid_result}, test: {test_result}")

        for key, result in results.items():
            valid_result, test_result = result
        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        cnt_wait+=1
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'previous_unconflicted': model.previous_unconflicted,
        'cluster_pred': model.cluster_pred,
    # }, f'./pretrained/{args.dataset}_kmeans_onlyedge_scheduler_seed0.pt')
    }, f'./pretrained/{args.dataset}_visang.pt')
    return results


parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, default=0, help='hyper search')
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument('--device', type=int, default=0, help='GPU . (default: 1)')
parser.add_argument('--encoder_hidden', type=int, default=512, help='Channels of hidden representation. (default: 64)')
parser.add_argument('--encoder_out', type=int, default=512, help='Channels of hidden representation. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers for decoders. (default: 1)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')

parser.add_argument('--decoder_hidden', type=int, default=64, help='Channels of decoder layers. (default: 128)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')

parser.add_argument('--nb_size', type=int, default=50, help='nb size for neighRouting. (default: 50)')
parser.add_argument('--ncaps', type=int, default=16, help='num channels. (default: 4)')
parser.add_argument('--nlayer', type=int, default=6, help='routing layer. (default: 6)')
parser.add_argument('--max_iter', type=int, default=11, help='routing iterations. (default: 12)')
parser.add_argument('--link_lr_max', type=float, default=0.1, help='Learning rate for link prediction. (default: 0.01)')
parser.add_argument('--link_lr_min', type=float, default=0.005,
                    help='Learning rate for link prediction. (default: 0.01)')

parser.add_argument('--grad_norm', type=float, default=1.0, help='(default: 1.0)')
parser.add_argument('--batch_size', type=int, default=2 ** 16,
                    help='Number of batch size for link prediction training. (default: 2**16)')
##
parser.add_argument('--l2_normalize', action='store_false',
                    help='Whether to use l2 normalize output embedding. (default: True)')
parser.add_argument('--alpha_l', type=int, default=1, help='(pubmed 3)')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs. (default: 450가 disen에서 잘 돼)')
parser.add_argument('--eval_period', type=int, default=1, help='(default: 10)')
parser.add_argument('--patience', type=int, default=150, help='(default: 4)')
parser.add_argument("--save_path", nargs="?", default="model_nodeclas.pth",
                    help="save path for model. (default: model_nodeclas)")
args = parser.parse_args()

if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

data = load_dataset(args.dataset,device)
args = load_best_configs(args, "node_configs.yml")
seed =0
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
edge_decoder = EdgeDecoder(args.encoder_out, args.decoder_hidden, num_layers=args.decoder_layers, dropout=args.decoder_dropout)

model = MaskGAE_pretrain(args, encoder, edge_decoder, mask, torch_generator=th_g, num_labels = len(torch.unique(data.y))).to(device)
print('\n start link pred')
results = train_linkpred(model, splits, args, device=device)

