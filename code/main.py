import torch
import argparse
from packages.utils import label2index
from packages import trainer
from packages.LoadData import DepolymeraseDataset, DposFinderDataset
from torch.utils.data import DataLoader
import os
from esm import Alphabet


parser = argparse.ArgumentParser(
    description='Phage Host Interaction Prediction')
parser.add_argument('-f', default='', type=str)

parser.add_argument('--mode', type=str, default='final_train',
                    help='mode to run(train, test, final_train or predict)')

parser.add_argument('--model', type=str, default='DposFinder',
                    help='name of the model to use')

parser.add_argument('--data_path', type=str, default='./data/',
                    help='path to dataset (e.g. ./data/)')

parser.add_argument('--test_data', type=str, default='',
                    help='name of test set file (e.g. test_set.fasta)')

# Dropout
parser.add_argument('--attn_dropout', type=float, default=0.05,
                    help='attention dropout (default: 0.05)')
parser.add_argument('--embed_dropout', type=float, default=0.4,
                    help='embedding dropout (default: 0.4)')

# Architecture

parser.add_argument('--n_layers', type=int, default=1,
                    help='number of layers in the network (default: 1)')
parser.add_argument('--num_heads', type=int, default=4,
                    help='number of heads for the transformer network (default: 4)')
parser.add_argument('--hid_dim', type=int, default=256,
                    help='number of hidden units in the network (default: 256)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for transformer (default: True)')

# Tuning
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate (default: 5e-5)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=15,
                    help='number of epochs (default: 15)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay rate (default: 1e-4)')
parser.add_argument('--when', type=int, default=5,
                    help='when to decay learning rate (default: 5)')


# Logistics
parser.add_argument('--log_interval', type=int, default=5,
                    help='frequency of result logging (default: 5)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='Final',
                    help='name of the trial (default: "Final")')
parser.add_argument('--return_embedding', action='store_true',
                    help='return hidden embeddings instead of logits (default: False)')
parser.add_argument('--return_attn', action='store_true',
                    help='return attention weights (default: False)')
parser.add_argument('--kfold', default=0, type=int,
                        help="num. of CV folds. (default: 0)")
parser.add_argument('--fold_num', default=0, type=int,
                    help="fold number (default: 0)")
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = False
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run without --no_cuda")
        device = torch.device('cpu')
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

alphabet = Alphabet.from_architecture("roberta_large")

if args.mode == 'train':
    ####################################################################
    #
    # Load the dataset
    #
    ####################################################################

    print("Start loading the data...")
    train_set = DepolymeraseDataset(path=os.path.join(args.data_path,'train_set.fasta'), transform=label2index, mode='train', kfold=args.kfold, fold_num=args.fold_num)
    valid_set = DepolymeraseDataset(path=os.path.join(args.data_path,'train_set.fasta'), transform=label2index, mode='valid', kfold=args.kfold, fold_num=args.fold_num)
    # train_set = BertDataset(path=os.path.join(args.data_path,'train_set_bert.pt'), transform=label2index,  mode='train', kfold=args.kfold, fold_num=args.fold_num)
    # valid_set = BertDataset(path=os.path.join(args.data_path,'train_set_bert.pt'), transform=label2index,  mode='valid', kfold=args.kfold, fold_num=args.fold_num)
    test_set = DepolymeraseDataset(path=os.path.join(args.data_path,'test_set.fasta'), transform=label2index, mode='test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=alphabet.get_batch_converter(), generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        collate_fn=alphabet.get_batch_converter(), shuffle=False, generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=padtoks, generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))
    # valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=padtoks, generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        collate_fn=alphabet.get_batch_converter(), shuffle=False, generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))

    print("Finish loading the data...")
    ####################################################################
    #
    # Hyperparameters
    #
    ####################################################################
    hyp_params = args
    hyp_params.emb_dim = 1280
    hyp_params.layers = args.n_layers
    hyp_params.use_cuda = use_cuda
    hyp_params.weight_decay = args.weight_decay
    # hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(
        train_set), len(valid_set), len(test_set)
    hyp_params.model = str.upper(args.model.strip())
    hyp_params.output_dim = 1
    hyp_params.criterion = 'BCEWithLogitsLoss'

elif args.mode == 'test':
    ####################################################################
    #
    # Load the dataset
    #
    ####################################################################
    print("Start loading the data...")

    path = os.path.join(args.data_path, args.test_data)
    test_set = DepolymeraseDataset(path=path, transform=label2index, mode='test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             collate_fn=alphabet.get_batch_converter(), shuffle=False, generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))

    print("Finish loading the data...")
    ####################################################################
    #
    # Hyperparameters
    #
    ####################################################################
    hyp_params = args
    hyp_params.emb_dim = 1280
    hyp_params.layers = args.n_layers
    hyp_params.use_cuda = use_cuda
    hyp_params.weight_decay = args.weight_decay
    # hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_test = len(test_set)
    hyp_params.model = str.upper(args.model.strip())
    hyp_params.output_dim = 1
    hyp_params.criterion = 'BCEWithLogitsLoss'

elif args.mode == 'predict':
    ####################################################################
    #
    # Load the dataset
    #
    ####################################################################

    path = os.path.join(args.data_path, args.test_data)
    test_set = DposFinderDataset(path=path)
    test_loader = DataLoader(test_set, batch_size=1,
                             collate_fn=alphabet.get_batch_converter(), shuffle=False, generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))

    ####################################################################
    #
    # Hyperparameters
    #
    ####################################################################
    hyp_params = args
    hyp_params.emb_dim = 1280
    hyp_params.layers = args.n_layers
    hyp_params.use_cuda = use_cuda
    hyp_params.weight_decay = args.weight_decay
    # hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_test = len(test_set)
    hyp_params.model = str.upper(args.model.strip())
    hyp_params.output_dim = 1
    hyp_params.criterion = 'BCEWithLogitsLoss'

elif args.mode == 'final_train':
    print("Start loading the data...")
    train_set = DepolymeraseDataset(path=os.path.join(args.data_path,'train_set.fasta'), transform=label2index, mode='test')
    test_set = DepolymeraseDataset(path=os.path.join(args.data_path,'test_set.fasta'), transform=label2index, mode='test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=alphabet.get_batch_converter(), generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        collate_fn=alphabet.get_batch_converter(), shuffle=False, generator=torch.Generator(device='cuda' if use_cuda else 'cpu'))
    
    print("Finish loading the data...")
    ####################################################################
    # Hyperparameters for final training on the whole dataset 
    ####################################################################
    hyp_params = args
    hyp_params.emb_dim = 1280
    hyp_params.layers = args.n_layers
    hyp_params.use_cuda = use_cuda
    hyp_params.weight_decay = args.weight_decay
    # hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_train, hyp_params.n_test = len(
        train_set), len(test_set)
    hyp_params.model = str.upper(args.model.strip())
    hyp_params.output_dim = 1
    hyp_params.criterion = 'BCEWithLogitsLoss'

if __name__ == '__main__':
    if args.mode == 'train':
        test_loss = trainer.initiate(
            hyp_params, train_loader, valid_loader, test_loader)
    elif args.mode == 'test':
        test_loss = trainer.test_case(hyp_params, test_loader)
    elif args.mode == 'predict':
        trainer.predict(hyp_params, test_loader)
    elif args.mode == 'final_train':
        test_loss = trainer.final_train(
            hyp_params, train_loader, test_loader)
