import tqdm
import torch
from torch import nn
from torch import optim
from models import KBCModel, N3
import numpy as np


class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regFlag: bool, reg: float, optimizer: optim.Optimizer, batch_size: int = 256,
            device: str = 'cpu', verbose: bool = True
    ):
        self.model = model
        self.regFlag = regFlag
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device
        self.regularizer = N3(reg)

    def epoch(self, e, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        losses = []
        b_begin = 0
        while b_begin < examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + self.batch_size].to(self.device)
            predictions, factors = self.model.forward(input_batch)
            truth = input_batch[:, 2]
            l = loss(predictions, truth)

            if self.regFlag:
                l_reg = self.regularizer.forward(factors)
                l = l + l_reg

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            b_begin += self.batch_size
            losses.append(l.item())
        print('train[{}]  loss:{:.5f}'.format(e, np.float(np.mean(losses))))

