import argparse
from typing import Dict
import time
import torch
from torch import optim
from datasets import Dataset
from models import CP, KDGene, ComplEx, N3
from optimizers import KBCOptimizer


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Knowledge Graph Completion")

    parser.add_argument('--dataset', default='DisGeNet_cv')
    optimizers = ['Adagrad', 'Adam', 'SGD']
    parser.add_argument('--optimizer', choices=optimizers, default='Adagrad', help="Optimizer in {}".format(optimizers))
    parser.add_argument('--valid', default=5, type=float, help="Number of epochs before valid.")
    parser.add_argument('--init', default=1e-3, type=float, help="Initial scale")
    parser.add_argument('--max_epochs', default=200, type=int, help="Number of epochs.")
    parser.add_argument('--early_stop', default=4, type=int, help="Early stop")

    models = ['CP', 'KDGene', 'ComplEx']
    gates = ['RNNCell', 'LSTMCell', 'GRUCell']
    parser.add_argument('--model', choices=models, default='KDGene', help="Model in {}".format(models))
    parser.add_argument('--gate', choices=gates, default='LSTMCell', help='Gating mechanism in {}'.format(gates))
    parser.add_argument('--exp_name', default='1124', type=str, help="Experiment name")
    parser.add_argument('--batch_size', default=1024, type=int, help="Batch size.")
    parser.add_argument('--learning_rate', default=0.05, type=float, help="Learning rate")
    parser.add_argument('--edim', default=2000, type=int, help="Entity embedding dimensionality.")
    parser.add_argument('--rdim', default=1500, type=int, help="Relation embedding dimensionality.")
    parser.add_argument('--regFlag', default=True, help='Whether to use N3 regularizer')
    parser.add_argument('--reg', default=0.1, type=float, help="Regularization weight")
    parser.add_argument('--device', default="cuda:0", type=str, help="cup or gpu")
    parser.add_argument('--fold', default=1, help="fold")
    return parser.parse_args(args)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


def main(args):
    fold = args.fold
    print("***************fold {}***************".format(str(fold)))
    dataset = Dataset(args.dataset, str(fold))
    examples = torch.from_numpy(dataset.get_train().astype('int64'))
    print("model: ", args.model)

    model = {
        'CP': lambda: CP(dataset.get_shape(), args.edim, args.init),
        'KDGene': lambda: KDGene(dataset.get_shape(), args.edim, args.rdim, args.gate, args.init),
        'ComplEx': lambda: ComplEx(dataset.get_shape(), args.edim, args.init),
    }[args.model]()

    model.to(args.device)

    optim_method = {
        'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)),
        'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate),
        'SparseAdam': lambda: optim.SparseAdam(model.parameters(), lr=args.learning_rate)
    }[args.optimizer]()

    optimizer = KBCOptimizer(model, args.regFlag, args.reg, optim_method, args.batch_size, args.device)
    curve = {'train': [], 'valid': [], 'test': []}
    best_mrr = -1e6
    stop_flag = 0
    final_epoch = 0
    for e in range(args.max_epochs):
        optimizer.epoch(e, examples)
        if (e + 1) % args.valid == 0:
            test = [dataset.eval(model, split, -1 if split != 'train' else 50000, args.device) for split in ['test']]
            curve['test'].append(test)
            print("* epoch[", e + 1, "], TEST : ", test)
            if list(test[0][0].values())[0] > best_mrr:
                best_mrr = list(test[0][0].values())[0]
                stop_flag = 0
            else:
                stop_flag += 1
                if stop_flag >= args.early_stop:
                    final_epoch = e + 1
                    break
    if final_epoch == 0:
        final_epoch = args.max_epochs

    print("final epoch: ", final_epoch)
    print("best MRR: ", best_mrr)
    print("***************fold{}*Disease Gene Prediction***************".format(str(fold)))
    dataset.predict(model, args.device, args.exp_name)
    torch.cuda.empty_cache()
    # results = dataset.eval(model, 'test', -1, args.device)
    # print("\n\nTEST : ", results)


if __name__ == '__main__':
    args = parse_args()
    main(args)
