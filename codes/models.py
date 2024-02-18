from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch


class KBCModel(torch.nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor, filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < chunk_size:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)
                    scores = q @ rhs
                    targets = self.score(these_queries)
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        # print(queries[b_begin + i, 2].item() == query[2].item())  True
                        filter_out += [queries[b_begin + i, 2].item()]

                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6  # filter
                    ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).float(), dim=1).cpu()
                    b_begin += batch_size  # next batch

                c_begin += chunk_size
        return ranks

    def get_predicted_gene(self, queries: torch.Tensor, filters: Dict[Tuple[int, int], List[int]], gene_num: int,
                           device: str = 'cpu'):
        with torch.no_grad():
            disease = torch.tensor([]).to(device)
            gene = torch.tensor([]).to(device)
            score = torch.tensor([]).to(device)
            rhs = self.get_rhs(0, gene_num)
            q = self.get_queries(queries)
            scores = q @ rhs
            for i, query in enumerate(queries):
                if (query[0].item(), query[1].item()) in filters:
                    in_train_gene = filters[(query[0].item(), query[1].item())]
                    all_gene_score = scores[i]
                    all_gene_score[torch.LongTensor(in_train_gene)] = -1e6

                    gene_rank = torch.argsort(all_gene_score, descending=True)[:(gene_num - len(in_train_gene))]
                else:
                    all_gene_score = scores[i]
                    gene_rank = torch.argsort(scores[i], descending=True)[:gene_num]
                gene_rank_scores = all_gene_score[gene_rank].view(-1, 1)

                disease = torch.cat((disease, torch.full(size=(len(gene_rank), 1), fill_value=query[0].item(),
                                                         dtype=torch.long).to(device)), 0)
                gene = torch.cat((gene, gene_rank.view(-1, 1)), 0)
                score = torch.cat((score, gene_rank_scores.view(-1, 1)), 0)
        return disease, gene, score


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = torch.nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = torch.nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = torch.nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class KDGene(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], edim: int, rdim: int, gatecell: str,
            init_size: float = 1e-3,
    ):
        super(KDGene, self).__init__()
        self.sizes = sizes
        self.edim = edim
        self.rdim = rdim
        self.gatecell = gatecell

        self.lhs = torch.nn.Embedding(sizes[0], edim, sparse=True)
        self.rel = torch.nn.Embedding(sizes[1], rdim, sparse=True)
        self.rhs = torch.nn.Embedding(sizes[2], edim, sparse=True)

        self.gate = {
            'RNNCell': lambda: torch.nn.RNNCell(rdim, edim),
            'LSTMCell': lambda: torch.nn.LSTMCell(rdim, edim),
            'GRUCell': lambda: torch.nn.GRUCell(rdim, edim)
        }[gatecell]()

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)

        return torch.sum(lhs * rel_update * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)
        output = lhs * rel_update
        pred = output @ self.rhs.weight.t()
        return pred, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)
        return lhs * rel_update


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)


class N3(torch.nn.Module):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight  # 0.01

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]
