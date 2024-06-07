import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--dataset', nargs='?', default='Cdataset', help='Choose a dataset. [Fdataset/Cdataset/LRSSL]')
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--epoch', default=200, type=float, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument('--disease_TopK', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--drug_TopK', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024 * 5, help='Batch size.')
    parser.add_argument('--wd', type=float, default=0.3, help='the coefficient of feature fusion ')
    parser.add_argument('--wr', type=float, default=0.3, help='the coefficient of feature fusion ')
    return parser.parse_args()

