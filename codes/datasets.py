import os
from typing import Tuple
import numpy as np
import torch
from collections import defaultdict
from models import KBCModel


class Dataset(object):
    def __init__(self, dataset: str, fold: str):
        self.root = '../' + dataset + '/'
        if dataset == 'DisGeNet_cv':
            self.gene_num = 8947	# candidate gene num
        # load entities and relations files
        with open(os.path.join(self.root, 'ent_id'), "r") as f:
            entities_to_id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
            ids_to_entities = {v: k for k, v in entities_to_id.items()}
        with open(os.path.join(self.root, 'rel_id'), "r") as f:
            relations_to_id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
        print("{} entities and {} relations".format(len(entities_to_id), len(relations_to_id)))
        self.n_entities = len(entities_to_id)
        self.n_relations = len(relations_to_id)
        self.n_predicates = self.n_relations * 2
        self.relations_to_id = relations_to_id
        self.ids_to_entities = ids_to_entities
        self.fold = fold

        files = ['train', 'test']
        self.data = {}
        for f in files:
            path = self.root + 'fold_' + fold + '_' + f + '.txt'
            to_read = open(path, 'r')
            examples = []
            for line in to_read.readlines():
                lhs, rel, rhs = line.strip().split('\t')
                if rel == 'rel':
                    continue
                try:
                    examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
                except ValueError:
                    continue
            self.data[f] = np.array(examples).astype('uint64')

        to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
        for f in files:
            for lhs, rel, rhs in self.data[f]:
                to_skip['lhs'][(rhs, rel + self.n_relations)].add(lhs)  # reciprocals
                to_skip['rhs'][(lhs, rel)].add(rhs)

        self.to_skip = {'lhs': {}, 'rhs': {}}
        for kk, skip in to_skip.items():
            for k, v in skip.items():
                self.to_skip[kk][k] = sorted(list(v))

        gene_in_train = {'rhs': defaultdict(set)}
        for lhs, rel, rhs in self.data['train']:
            gene_in_train['rhs'][(lhs, rel)].add(rhs)

        self.gene_in_train = {'rhs': {}}
        for kk, skip in gene_in_train.items():
            for k, v in skip.items():
                self.gene_in_train[kk][k] = sorted(list(v))

        print("* {} triples in train".format(len(self.data['train'])))
        # [disease-symptom, protein-protein(-900/950/850), GO-protein, pathway-protein, disease-drug, chemical-protein, demo]
        base_kg = ['disease-symptom']
        for kg in base_kg:
            examples = []
            with open("../src_data/" + kg + '.txt', "r") as f:
                for line in f.readlines():
                    lhs, rel, rhs = line.strip().split('\t')
                    try:
                        examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
                    except ValueError:
                        continue
            print("* {} triples in {}".format(len(examples), kg))
            self.data['train'] = np.vstack((self.data['train'], np.array(examples).astype('uint64')))
        print("* train+base_kg: {}".format(len(self.data['train'])))

    def get_examples(self, split):
        return self.data[split]

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, device: str = 'cpu', missing_eval: str = 'rhs',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).to(device)
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]

            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2

            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500, chunk_size=self.gene_num)
            mean_reciprocal_rank[m] = round(torch.mean(1. / ranks).item(), 3)
            hits_at[m] = torch.FloatTensor((list(map(lambda x: torch.mean((ranks <= x).float()).item(), at))))
        return mean_reciprocal_rank, hits_at

    def predict(self, model: KBCModel, device: str = 'cpu', exp_name: str = 'exp'):
        test = self.get_examples('test')
        disease = np.unique(test[:, 0])
        dis_tensor = torch.from_numpy(disease.astype('int64'))
        dis_tensor = dis_tensor.view(-1, 1)
        rel_tensor = torch.full(size=(len(disease), 1), fill_value=self.relations_to_id['disease_gene'])
        try:
            dis_rel = torch.cat((dis_tensor, rel_tensor), 1).to(device)
        except ValueError:
            print(dis_tensor.shape, rel_tensor.shape)

        disease, gene, score = model.get_predicted_gene(dis_rel, self.gene_in_train['rhs'], self.gene_num, device)
        assert len(disease) == len(gene) == len(score)
        w_disease = disease[0]
        index = 0
        with open(exp_name + "_fold" + self.fold + ".txt", 'w') as f:
            for i in range(len(disease)):
                if disease[i] != w_disease:
                    index = 0
                    w_disease = disease[i]
                if index < 100:
                    f.write(
                        str(self.ids_to_entities[disease[i].item()]) + '\t' + str(self.ids_to_entities[gene[i].item()])
                        + '\t' + str(score[i].item()) + '\n')
                    index += 1
        print("fold {} predicted genes saved".format(self.fold))
