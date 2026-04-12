from datasets import dataloader_generator, multidataloader_generator, CUSTOM_ROI_ROOT_DEFAULTS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models.heads import Classifier, BNDecoder
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os, importlib, subprocess
import numpy as np
from datetime import datetime

MODEL_SPECS = {
    'neurodetour': ('models.neuro_detour', 'DetourTransformer'),
    'neurodetourSingleFC': ('models.neuro_detour', 'DetourTransformerSingleFC'),
    'neurodetourSingleSC': ('models.neuro_detour', 'DetourTransformerSingleSC'),
    'bnt': ('models.brain_net_transformer', 'BrainNetworkTransformer'),
    'braingnn': ('models.brain_gnn', 'Network'),
    'bolt': ('models.bolt', 'get_BolT'),
    'graphormer': ('models.graphormer', 'Graphormer'),
    'nagphormer': ('models.nagphormer', 'TransformerModel'),
    'transformer': ('models.vanilla_model', 'Transformer'),
    'gcn': ('models.vanilla_model', 'GCN'),
    'sage': ('models.vanilla_model', 'SAGE'),
    'sgc': ('models.vanilla_model', 'SGC'),
    'none': ('models.brain_identity', 'Identity'),
}
CLASSIFIER_BANK = {
    'mlp': nn.Linear,
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'sgc': SGConv
}
TRANSFORM_SPECS = {
    'graphormer': ('models.graphormer', 'ShortestDistance'),
    'nagphormer': ('models.nagphormer', 'NAGdataTransform'),
}
ATLAS_ROI_N = {
    'AAL_116': 116,
    'Gordon_333': 333,
    'Shaefer_100': 100,
    'Shaefer_200': 200,
    'Shaefer_400': 400,
    'D_160': 160
}
DATA_CLASS_N = {
    'ukb': 2,
    'hcpa': 4,
    'hcpya': 7,
    'adni': 2,
    'oasis': 2,
    'oasis': 2,
    'ppmi': 4,
    'abide': 2,
    'neurocon': 2,
    'taowu': 2,
    'adni_our_2cls': 2,
    'ppmi_our_2cls': 2,
    'ppmi_our_3cls': 3,
}
LOSS_FUNCS = {
    'y': nn.CrossEntropyLoss(),
    # 'sex': nn.CrossEntropyLoss(),
    # 'age': nn.MSELoss(), 
}
LOSS_W = {
    'y': 1,
    'sex': 1,
    'age': 1e-4,
}

CUSTOM_DOWNSTREAM_DATASETS = set(CUSTOM_ROI_ROOT_DEFAULTS)


def get_model_class(model_name):
    module_name, attr_name = MODEL_SPECS[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_data_transform(model_name):
    if model_name not in TRANSFORM_SPECS:
        return None
    module_name, attr_name = TRANSFORM_SPECS[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)()


def resolve_weights_dir(args, save_mn):
    if args.weights_dir is not None:
        return args.weights_dir
    return f'model_weights/{save_mn}_{"-".join(args.pretrained_datanames)}_boldwin{args.bold_winsize}_{args.adj_type}{args.node_attr}'


def pick_least_used_cuda_device():
    if not torch.cuda.is_available():
        print('CUDA is not available, fallback to cpu')
        return 'cpu'
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            check=True,
            capture_output=True,
            text=True
        )
        candidates = []
        for line in result.stdout.strip().splitlines():
            idx, mem_used, mem_total, gpu_util = [item.strip() for item in line.split(',')]
            candidates.append((int(idx), int(mem_used), int(mem_total), int(gpu_util)))
        best_idx, best_used, best_total, best_util = min(candidates, key=lambda item: (item[1], item[3], item[0]))
        device = f'cuda:{best_idx}'
        print(f'Auto-selected device {device} (memory_used={best_used}MiB/{best_total}MiB, gpu_util={best_util}%)')
        return device
    except Exception as exc:
        print(f'Auto device selection failed ({exc}), fallback to cuda:0')
        return 'cuda:0'


def resolve_device(device_arg):
    if device_arg is None:
        return 'cpu'
    if str(device_arg).lower() in ['auto', 'cuda:auto', 'gpu:auto']:
        return pick_least_used_cuda_device()
    return device_arg


def find_checkpoint_pair(weights_dir, fold_idx, load_dname, checkpoint_fold=None):
    load_fold = fold_idx if checkpoint_fold is None else checkpoint_fold
    bb_prefix = f'bb_fold{load_fold}_{load_dname}Best_'
    head_prefix = f'head_fold{load_fold}_{load_dname}Best_'
    bb_path, head_path = None, None
    for fn in os.listdir(weights_dir):
        if fn.startswith(bb_prefix):
            bb_path = os.path.join(weights_dir, fn)
        if fn.startswith(head_prefix):
            head_path = os.path.join(weights_dir, fn)
        if bb_path is not None and head_path is not None:
            break
    if bb_path is None or head_path is None:
        raise FileNotFoundError(f'Cannot find checkpoint pair under {weights_dir} with prefixes {bb_prefix} / {head_prefix}')
    return bb_path, head_path


def infer_pretrained_nclass(head_state):
    if 'object_query.weight' not in head_state:
        raise KeyError('Checkpoint missing object_query.weight, cannot infer pre-trained class token count')
    return head_state['object_query.weight'].shape[0] - 3


def build_custom_finetune_tokenid(pretrained_nclass, task_nclass):
    return torch.LongTensor(list(range(pretrained_nclass, pretrained_nclass + task_nclass)) + [-3, -2, -1])


def safe_load_partial_state_dict(model, state_dict):
    model_state = model.state_dict()
    matched_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched_state[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(matched_state, strict=False)
    print(f'Loaded {len(matched_state)} matching tensors into {type(model).__name__}; skipped {len(skipped)} tensors')


def build_custom_dataloaders(args, dataset_cache, transform, fold_idx):
    defaults = CUSTOM_ROI_ROOT_DEFAULTS[args.dataset_name]
    roi_root = args.roi_root or defaults['roi_root']
    label_csv = args.label_csv or defaults['label_csv']
    train_loader, val_loader, dataset_cache = dataloader_generator(
        batch_size=args.batch_size,
        nfold=fold_idx,
        total_fold=args.cv_fold_n,
        dataset=dataset_cache,
        dname=args.dataset_name,
        node_attr=args.node_attr,
        adj_type=args.adj_type,
        transform=transform,
        fc_winsize=args.bold_winsize,
        atlas_name=args.atlas,
        fc_th=args.fc_th,
        sc_th=args.sc_th,
        roi_root=roi_root,
        label_csv=label_csv,
    )
    print(
        f'Custom downstream dataset={args.dataset_name}, roi_root={roi_root}, label_csv={label_csv}, '
        f'nclass={dataset_cache.nclass_list[0]}, label_names={dataset_cache.label_names}'
    )
    return train_loader, {args.dataset_name: val_loader}, dataset_cache, dataset_cache

def main():
    parser = argparse.ArgumentParser(description='NeuroDetour')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default = 200)
    parser.add_argument('--models', type=str, default = 'none')
    parser.add_argument('--classifier', type=str, default = 'mlp')
    parser.add_argument('--max_patience', type=int, default = 50)
    parser.add_argument('--hiddim', type=int, default = 2048)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--atlas', type=str, default = 'AAL_116')
    # parser.add_argument('--dataname', type=str, default = 'ppmi')
    # parser.add_argument('--testname', type=str, default = 'None')
    parser.add_argument('--node_attr', type=str, default = 'FC')
    parser.add_argument('--adj_type', type=str, default = 'FC')
    parser.add_argument('--bold_winsize', type=int, default = 500)
    parser.add_argument('--nlayer', type=int, default = 4)
    parser.add_argument('--nhead', type=int, default = 8)
    parser.add_argument('--classifier_aggr', type=str, default = 'learn')
    parser.add_argument('--savemodel', action='store_true')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--device', type=str, default = 'cuda:2')
    parser.add_argument('--fc_th', type=float, default = 0.5)
    parser.add_argument('--sc_th', type=float, default = 0.1)
    parser.add_argument('--only_dataload', action='store_true')
    parser.add_argument('--cv_fold_n', type=int, default = 10)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--decoder_layer', type=int, default = 32)
    parser.add_argument('--datanames', nargs='+', default = ['adni','abide','ppmi','taowu','neurocon'], required=False)
    parser.add_argument('--pretrained_datanames', nargs='+', default = ['ppmi','abide','taowu','neurocon','hcpa','hcpya'], required=False)
    parser.add_argument('--load_dname', type=str, default = 'hcpa', required=False)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--roi_root', type=str, default=None)
    parser.add_argument('--label_csv', type=str, default=None)
    parser.add_argument('--weights_dir', type=str, default=None)
    parser.add_argument('--fold_ids', nargs='+', type=int, default=None)
    parser.add_argument('--checkpoint_fold', type=int, default=None)

    args = parser.parse_args()
    # args.decoder = True
    print(args)
    # expdate = str(datetime.now())
    # expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    load_dname = args.load_dname
    use_custom_downstream = args.dataset_name in CUSTOM_DOWNSTREAM_DATASETS if args.dataset_name is not None else False
    device = resolve_device(args.device)
    args.device = device
    hiddim = args.hiddim
    # nclass = DATA_CLASS_N[args.dataname]
    if use_custom_downstream:
        dataset = None
    else:
        dataset = {'adni': None,'hcpa': None,'hcpya': None,'abide': None,'ppmi': None,'taowu': None,'neurocon': None}
    # Initialize lists to store evaluation metrics
    accuracies_dict = {}
    f1_scores_dict = {}
    prec_scores_dict = {}
    rec_scores_dict = {}
    # taccuracies = []
    # tf1_scores = []
    # tprec_scores = []
    # trec_scores = []
    node_sz = ATLAS_ROI_N[args.atlas]
    # if args.models != 'neurodetour':
    transform = None
    # dek, pek = 0, 0
    if args.node_attr != 'BOLD':
        input_dim = node_sz
    else:
        input_dim = args.bold_winsize
    transform = get_data_transform(args.models)
    # testset = args.testname
    
    
    if args.decoder:
        save_mn = f'{args.models}_decoder{args.decoder_layer}'
    else:
        save_mn = f'{args.models}_mlp{args.decoder_layer}'
    
    mweight_fn = resolve_weights_dir(args, save_mn)
    assert os.path.exists(mweight_fn), mweight_fn
    print(f'Using checkpoint directory: {mweight_fn}')
    fold_ids = args.fold_ids if args.fold_ids is not None else list(range(args.cv_fold_n))
    for i in fold_ids:
        if use_custom_downstream:
            train_loader, val_loader, merged_dataset, dataset = build_custom_dataloaders(args, dataset, transform, i)
        else:
            dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=dataset, dname_list=args.datanames,
                                                                     node_attr=args.node_attr, adj_type=args.adj_type, transform=transform,
                                                                     fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
            train_loader, val_loader, merged_dataset, dataset = dataloaders
        if args.only_dataload:
            exit()

        model_cls = get_model_class(args.models)
        model = model_cls(node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=args.batch_size, device=device, nlayer=args.nlayer, heads=args.nhead).to(device)
        bb_ckpt, head_ckpt = find_checkpoint_pair(mweight_fn, i, load_dname, checkpoint_fold=args.checkpoint_fold)
        print(f'{datetime.now()} Load pre-trained model: bb={bb_ckpt}, head={head_ckpt}')
        model.load_state_dict(torch.load(bb_ckpt, map_location='cpu'))
        head_state = torch.load(head_ckpt, map_location='cpu')

        if use_custom_downstream:
            target_nclass = sum(merged_dataset.nclass_list)
            if not args.decoder:
                classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nlayer=args.decoder_layer, nclass=target_nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), aggr=args.classifier_aggr).to(device)
                safe_load_partial_state_dict(classifier, head_state)
            else:
                pretrained_nclass = infer_pretrained_nclass(head_state)
                finetune_tokenid = build_custom_finetune_tokenid(pretrained_nclass, target_nclass)
                print(f'Custom task token setup: pretrained_nclass={pretrained_nclass}, target_nclass={target_nclass}, finetune_tokenid={finetune_tokenid.tolist()}')
                classifier = BNDecoder(
                    hiddim,
                    nclass=pretrained_nclass,
                    node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio),
                    nlayer=args.decoder_layer,
                    head_num=8,
                    finetune=True,
                    finetune_nclass=target_nclass,
                    finetune_tokenid=finetune_tokenid
                ).to(device)
                classifier.load_state_dict(head_state, strict=False)
        else:
            pretrain_dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=dataset, dname_list=args.pretrained_datanames,
                                                                     node_attr=args.node_attr, adj_type=args.adj_type, transform=transform,
                                                                     fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
            pretrain_merged_dataset = pretrain_dataloaders[2]
            nclass = sum(pretrain_merged_dataset.nclass_list)
            overlap_dnames = list(np.intersect1d(args.pretrained_datanames, merged_dataset.dnames))
            if 'ppmi' in merged_dataset.dnames and 'ppmi' not in overlap_dnames:
                if 'taowu' in overlap_dnames: del overlap_dnames[overlap_dnames.index('taowu')]
                if 'neurocon' in overlap_dnames: del overlap_dnames[overlap_dnames.index('neurocon')]
            overlap_dtid = list(set(pretrain_merged_dataset.dname2tokenid[d] for d in overlap_dnames))
            overlap_dtoken = sum([pretrain_merged_dataset.nclass_list[tid] for tid in overlap_dtid])
            assert sum(merged_dataset.nclass_list) - overlap_dtoken >= 0, f'{sum(merged_dataset.nclass_list)} - {overlap_dtoken}'
            tokenid2dname = {}
            for d in merged_dataset.dname2tokenid:
                tid = merged_dataset.dname2tokenid[d]
                if tid not in tokenid2dname: tokenid2dname[tid] = []
                tokenid2dname[tid].append(d)
            print(merged_dataset.dname2tokenid)
            assert max(tokenid2dname.keys()) == len(tokenid2dname.keys())-1, tokenid2dname
            finetune_tokenid = []
            new_token_nclass = 0
            for tid in range(max(tokenid2dname.keys())+1):
                overlap_d = False
                for d in tokenid2dname[tid]:
                    if d in overlap_dnames:
                        start_ti = sum([pretrain_merged_dataset.nclass_list[nclass_i] for nclass_i in range(pretrain_merged_dataset.dname2tokenid[d])])
                        end_ti = pretrain_merged_dataset.nclass_list[pretrain_merged_dataset.dname2tokenid[d]] + start_ti
                        f_tid = list(range(start_ti, end_ti))
                        overlap_d = True
                        break
                if not overlap_d:
                    f_tid = list(range(nclass+new_token_nclass, nclass+new_token_nclass+merged_dataset.nclass_list[tid]))
                    new_token_nclass += merged_dataset.nclass_list[tid]
                finetune_tokenid.extend(f_tid)
            finetune_tokenid = torch.LongTensor(finetune_tokenid+[-3,-2,-1])
            print(finetune_tokenid)
            if not args.decoder:
                classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nlayer=args.decoder_layer, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), aggr=args.classifier_aggr).to(device)
            else:
                classifier = BNDecoder(hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), nlayer=args.decoder_layer, head_num=8, finetune=True, finetune_nclass=sum(merged_dataset.nclass_list) - overlap_dtoken, finetune_tokenid=finetune_tokenid).to(device)
            classifier.load_state_dict(head_state, strict=False)
        print(datetime.now(), 'Done')
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # print(optimizer)
        
        best_f1 = {}
        best_acc = {}
        best_prec = {}
        best_rec = {}
        # patience = {}
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            print(datetime.now(), 'train start')
            train(model, classifier, device, train_loader, optimizer, epoch)
            print(datetime.now(), 'train done, test start')
            for dname in val_loader:
                if dname not in best_f1:
                    best_f1[dname] = {}
                    best_acc[dname] = {}
                    best_prec[dname] = {}
                    best_rec[dname] = {}
                one_val_loader = val_loader[dname]
                # acc, prec, rec, f1 = eval(model, classifier, device, one_val_loader, dname=dname)
                scores = eval(model, classifier, device, one_val_loader, dname=dname)
                print(datetime.now(), 'test done')
                log = f'Dataset: {dname} [Accuracy, F1 Score]:'
                for k in scores:
                    acc, prec, rec, f1 = scores[k]
                    if scores[k][0] == -1:
                        f1 = -1*f1

                    log += f'({k}) [{acc:.6f},  {f1:.6f}], \t'
                    if k not in best_f1[dname]:
                        best_f1[dname][k] = -torch.inf
                        best_acc[dname][k] = -torch.inf
                        best_prec[dname][k] = -torch.inf
                        best_rec[dname][k] = -torch.inf
                    
                    if f1 >= best_f1[dname][k]:
                        best_f1[dname][k] = f1
                        best_acc[dname][k] = acc
                        best_prec[dname][k] = prec
                        best_rec[dname][k] = rec
                        # if args.savemodel:
                        #     torch.save(model.state_dict(), f'{mweight_fn}/bb_fold{i}_{dname}Best-{k}_{expdate}.pt')
                        #     torch.save(classifier.state_dict(), f'{mweight_fn}/head_fold{i}_{dname}Best-{k}_{expdate}.pt')
                
                print(log)
        
        for dname in best_acc:
            log = f'Dataset: {dname} [Accuracy, F1 Score, Prec, Rec]:'
            for k in best_acc[dname]:
                if dname not in accuracies_dict: accuracies_dict[dname] = {k: []}
                if dname not in f1_scores_dict: f1_scores_dict[dname] = {k: []}
                if dname not in prec_scores_dict: prec_scores_dict[dname] = {k: []}
                if dname not in rec_scores_dict: rec_scores_dict[dname] = {k: []}
                if k not in accuracies_dict[dname]:
                    accuracies_dict[dname][k] = []
                    f1_scores_dict[dname][k] = []
                    prec_scores_dict[dname][k] = []
                    rec_scores_dict[dname][k] = []
                accuracies_dict[dname][k].append(best_acc[dname][k])
                f1_scores_dict[dname][k].append(best_f1[dname][k])
                prec_scores_dict[dname][k].append(best_prec[dname][k])
                rec_scores_dict[dname][k].append(best_rec[dname][k])
                log += f'({k}) [{best_acc[dname][k]}, {best_f1[dname][k]}, {best_prec[dname][k]}, {best_rec[dname][k]}], \t'
            print(log)

    # Calculate mean and standard deviation of evaluation metrics
    for dname in accuracies_dict:
        for k in accuracies_dict[dname]:
            accuracies = accuracies_dict[dname][k]
            f1_scores = f1_scores_dict[dname][k]
            prec_scores = prec_scores_dict[dname][k]
            rec_scores = rec_scores_dict[dname][k]
            mean_accuracy = sum(accuracies) / len(accuracies)
            std_accuracy = torch.std(torch.tensor(accuracies))
            mean_f1_score = sum(f1_scores) / len(f1_scores)
            std_f1_score = torch.std(torch.tensor(f1_scores))
            mean_prec_score = sum(prec_scores) / len(prec_scores)
            std_prec_score = torch.std(torch.tensor(prec_scores))
            mean_rec_score = sum(rec_scores) / len(rec_scores)
            std_rec_score = torch.std(torch.tensor(rec_scores))
            print(f'Dataset: {dname} ({k})')
            print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
            print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
            print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
            print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')

        
def train(model, classifier, device, loader, optimizer, epoch):
    model.train()
    classifier.train()
    losses = []
    y_true_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    y_scores_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    # loss_fn = nn.CrossEntropyLoss()
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        feat = model(batch)
        edge_index = batch.edge_index
        batchid = batch.batch
        if len(feat) == 3:  # brainGnn Selected Topk nodes
            feat, edge_index, batchid = feat
        y = classifier(feat, edge_index, batchid)
        loss = 0
        for k in LOSS_FUNCS:
            if k == 'y':
                pre_nclass_i = 0
                for nclassi, nclass in enumerate(loader.dataset.dataset.nclass_list):
                    one_gt = batch[k][:, nclassi]
                    one_y = y[k][..., pre_nclass_i:pre_nclass_i+nclass]
                    if len(one_y.shape) == 3:
                        one_y = one_y[:, one_gt != -1]                    
                        one_gt = one_gt[one_gt != -1]
                        if epoch > 5:
                            one_yi = torch.arange(one_y.shape[1])
                            layeri = one_y[:, one_yi, one_gt].argmax(0)
                            one_y = one_y[layeri, one_yi]
                        else:
                            one_y = one_y.mean(0)
                    else:
                        one_y = one_y[one_gt != -1]
                        one_gt = one_gt[one_gt != -1]
                    loss += LOSS_W[k]*LOSS_FUNCS[k](one_y, one_gt)
                    pre_nclass_i += nclass
            else: # sex or age
                one_gt = batch[k]
                one_y = y[k]
                if len(one_y.shape) == 3:
                    one_y = one_y[:, one_gt != -1]                    
                    one_gt = one_gt[one_gt != -1]
                    if epoch > 5 and k != 'age':
                        one_yi = torch.arange(one_y.shape[1])
                        layeri = one_y[:, one_yi, one_gt].argmax(0)
                        one_y = one_y[layeri, one_yi]
                    else:
                        one_y = one_y.mean(0)
                else:
                    one_y = one_y[one_gt != -1]
                    one_gt = one_gt[one_gt != -1]

                loss += LOSS_W[k]*LOSS_FUNCS[k](one_y, one_gt)
            # print(k, y[k].shape, loss)
        
        # exit()
        if hasattr(model, 'loss'):
            loss = loss + model.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        pre_nclass_i = 0
        for nclassi, nclass in enumerate(loader.dataset.dataset.nclass_list):            
            for k in LOSS_FUNCS:
                if k != 'y': continue
                one_gt = batch[k][:, nclassi]
                one_y = y[k][..., pre_nclass_i:pre_nclass_i+nclass]
                if len(one_y.shape) == 3:
                    one_y = one_y[:, one_gt != -1].max(0)[0]
                else:
                    one_y = one_y[one_gt != -1]
                one_gt = one_gt[one_gt != -1]
                if k not in y_true_dict[nclassi]: 
                    y_true_dict[nclassi][k] = []
                    y_scores_dict[nclassi][k] = []
                y_true_dict[nclassi][k].append(one_gt.detach().cpu())
                y_scores_dict[nclassi][k].append(one_y.detach().cpu())
            pre_nclass_i += nclass
        
        nclassi = len(loader.dataset.dataset.nclass_list)
        for k in LOSS_FUNCS:
            if k == 'y': continue
            one_gt = batch[k]
            one_y = y[k]
            if len(one_y.shape) == 3:
                if k == 'age':
                    one_y = one_y[:, one_gt != -1].mean(0)
                else:
                    one_y = one_y[:, one_gt != -1].max(0)[0]
            else:
                one_y = one_y[one_gt != -1]
            one_gt = one_gt[one_gt != -1]
            if k not in y_true_dict[nclassi]: 
                y_true_dict[nclassi][k] = []
                y_scores_dict[nclassi][k] = []
            y_true_dict[nclassi][k].append(one_gt.detach().cpu())
            y_scores_dict[nclassi][k].append(one_y.detach().cpu())
            nclassi += 1
    # print([_y_true_dict.keys() for _y_true_dict in y_true_dict])
    logs = [f'Train loss: {np.mean(losses):.6f}']
    for di, dname in enumerate(loader.dataset.dataset.dnames):
        di = loader.dataset.dataset.dname2tokenid[dname]
        for k in y_true_dict[di]:
            assert k == 'y', f'{k},{di},{dname},{len(y_true_dict)}'
            y_true = torch.cat(y_true_dict[di][k], dim = 0).detach().cpu()
            y_scores = torch.cat(y_scores_dict[di][k], dim = 0).detach().cpu()
            y_true = y_true.numpy()
            y_scores = y_scores.numpy().argmax(1)
            acc = accuracy_score(y_true, y_scores)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
            logs.append(f'{dname}-{k}-Accuracy: {acc:.6f}')
            
    for di in range(len(loader.dataset.dataset.nclass_list), len(y_true_dict)):
        dname = 'All'
        for k in y_true_dict[di]:
            assert k != 'y', k
            y_true = torch.cat(y_true_dict[di][k], dim = 0).detach().cpu()
            y_scores = torch.cat(y_scores_dict[di][k], dim = 0).detach().cpu()
            if k != 'age':
                y_true = y_true.numpy()
                y_scores = y_scores.numpy().argmax(1)
                acc = accuracy_score(y_true, y_scores)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
                logs.append(f'{dname}-{k}-Accuracy: {acc:.6f}')
            else:
                logs.append(f'{dname}-{k}-MSE: {torch.nn.functional.mse_loss(y_scores, y_true):.6f}')
    
    print(', '.join(logs))

def eval(model, classifier, device, loader, dname=None):
    model.eval()
    classifier.eval()
    y_true = [[]]
    y_scores = [[]]
    y_true_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    y_scores_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            feat = model(batch)
            edge_index = batch.edge_index
            batchid = batch.batch
            if len(feat) == 3:  # brainGnn Selected Topk nodes
                feat, edge_index, batchid = feat
            y = classifier(feat, edge_index, batchid)

        pre_nclass_i = 0
        for nclassi, nclass in enumerate(loader.dataset.dataset.nclass_list):
            for k in LOSS_FUNCS:
                if k != 'y': continue
                one_gt = batch[k][:, nclassi]
                one_y = y[k][..., pre_nclass_i:pre_nclass_i+nclass]
                if len(one_y.shape) == 3:
                    one_y = one_y[:, one_gt != -1].max(0)[0]
                else:
                    one_y = one_y[one_gt != -1]
                one_gt = one_gt[one_gt != -1]
                if k not in y_true_dict[nclassi]:
                    y_true_dict[nclassi][k] = []
                    y_scores_dict[nclassi][k] = []
                y_true_dict[nclassi][k].append(one_gt.detach().cpu())
                y_scores_dict[nclassi][k].append(one_y.detach().cpu())
            pre_nclass_i += nclass
        
        nclassi = len(loader.dataset.dataset.nclass_list)
        for k in LOSS_FUNCS:
            if k == 'y': continue
            one_gt = batch[k]
            one_y = y[k]
            if len(one_y.shape) == 3:
                if k == 'age':
                    one_y = one_y[:, one_gt != -1].mean(0)
                else:
                    one_y = one_y[:, one_gt != -1].max(0)[0]
            else:
                one_y = one_y[one_gt != -1]
            one_gt = one_gt[one_gt != -1]
            if k not in y_true_dict[nclassi]:
                y_true_dict[nclassi][k] = []
                y_scores_dict[nclassi][k] = []
            y_true_dict[nclassi][k].append(one_gt.detach().cpu())
            y_scores_dict[nclassi][k].append(one_y.detach().cpu())
            nclassi += 1
     
    val_di = loader.dataset.dataset.dname2tokenid[dname] 
    scores = {}
    if 'y' in LOSS_FUNCS:
        y_true = torch.cat(y_true_dict[val_di]['y'], dim = 0).numpy()
        y_scores = torch.cat(y_scores_dict[val_di]['y'], dim = 0).numpy().argmax(1)
        acc = accuracy_score(y_true, y_scores)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
        scores['y'] = [acc, prec, rec, f1]
    for di in range(len(loader.dataset.dataset.nclass_list), len(y_true_dict)):
        for k in y_true_dict[di]:
            assert k != 'y', k
            y_true = torch.cat(y_true_dict[di][k], dim = 0).detach().cpu()
            y_scores = torch.cat(y_scores_dict[di][k], dim = 0).detach().cpu()
            if k != 'age':
                y_true = y_true.numpy()
                y_scores = y_scores.numpy().argmax(1)
                acc = accuracy_score(y_true, y_scores)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
                scores[k] = [acc, prec, rec, f1]
            else:
                scores[k] = [-1, -1, -1, torch.nn.functional.mse_loss(y_scores, y_true)]

    return scores

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()
