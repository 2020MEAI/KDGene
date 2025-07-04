import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pickle as pk
import ast
import warnings

warnings.filterwarnings('ignore')


class Evaluation(object):
    def __init__(self):

        self.result = list()
        self._evaluation_main()


    def _evaluation_main(self):

        # cv_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        cv_list = ['1', '2']
        self.ks = [1, 3, 10, 50]

        self.path = ''
        tmp = 'eval'
        self.cv_file = 'DisGeNet_cv.txt'
        self.out_file = self.path + tmp + '_metrics.txt'
        for self.cv_i in cv_list:
            self.pre_file = self.path + tmp + '_fold' + self.cv_i + '.txt'
            self._cal_eva()
        self.print_result()

    def _cal_eva(self):
        self.fpr_dict = dict()
        self.tpr_dict = dict()
        self.auc_counter = 0
        self.cv_dict = self.load_cv_data()
        self.pre_dict = self.load_pre_data()
        test_list = self.cv_dict.get(str(self.cv_i)).get('test')
        train_list = self.cv_dict.get(str(self.cv_i)).get('train')
        test_gene_list = list(set([x for _, x in test_list]))
        train_dict = dict()
        for d, g in train_list:
            train_dict.setdefault(d, set())
            train_dict[d].add(g)

        test_dict = dict()
        for d, g in test_list:
            test_dict.setdefault(d, set())
            test_dict[d].add(g)

        test_metric_dict = dict()
        test_metric_list = list()
        for test_d, pre_gene_dict in self.pre_dict.items():
            # 1. 排除test_d在训练集中的基因
            glist = train_dict.get(test_d)
            if glist is not None:
                test_gene_list_selected = list(set(test_gene_list) - set(glist))
            else:
                test_gene_list_selected = test_gene_list
            n_test_gene = len(test_gene_list_selected)

            # 2. 获得test_d的测试基因
            d_test_gene_set = test_dict.get(test_d)
            if d_test_gene_set is None: continue

            # 3. 构建score_list，label_list
            score_list = [0] * n_test_gene
            label_list = [0] * n_test_gene

            j = 0
            for i in range(n_test_gene):
                g = test_gene_list_selected[i]
                if g not in d_test_gene_set and g in pre_gene_dict:
                    score_list[j] = pre_gene_dict.get(g)
                    j += 1

            n_true_gene = len(d_test_gene_set)
            i = 0
            for g in d_test_gene_set:
                index_g = n_test_gene - n_true_gene + i
                if g in pre_gene_dict:
                    score_list[index_g] = pre_gene_dict.get(g)
                label_list[index_g] = 1
                i += 1

            n_d_test_gene = len(d_test_gene_set)
            auc_list1, n_hit_list, ap_list = self.cal_metrics(score_list, label_list, n_d_test_gene)
            test_metric_dict.setdefault(test_d, dict())
            test_metric_dict[test_d]['n_hit'] = n_hit_list
            test_metric_dict[test_d]['n_test_gene'] = n_d_test_gene
            test_metric_dict[test_d]['ap_list'] = ap_list

            test_metric_list.append([auc_list1, n_hit_list, ap_list, n_d_test_gene])

        auc_list, HR_list, MAP_list = self.metric_avg(test_metric_list)
        n_test_dis = len(self.pre_dict)
        r_list = [len(test_dict), n_test_dis] + HR_list + MAP_list
        print('\t'.join([str(x) for x in r_list]))
        self.result.append(r_list + [test_metric_dict])


    def metric_avg1(self, metric_list):
        hit_list = [x[0] for x in metric_list]
        ap_list = [x[1] for x in metric_list]
        n_test_gene_all = sum([x[2] for x in metric_list])

        HR_list = list()
        MAP_list = list()
        for i in range(len(self.ks)):
            n_hit_all = sum([x[i] for x in hit_list])
            HR = n_hit_all / n_test_gene_all  # hit ratio
            MAP = sum([x[i] for x in ap_list]) / len(ap_list)
            HR_list.append(round(HR, 5))
            MAP_list.append(round(MAP, 5))
        return HR_list, MAP_list

    def metric(self, algo, result_list):
        ks_list = ['HR@' + str(k) for k in self.ks] + ['MAP@' + str(k) for k in self.ks]
        title = '\t'.join(['fold', 'n_test_dis'] + ks_list)
        print(title)
        n_metric = len(result_list[0])
        avg_list = list()
        std_list = list()
        self.fw.write('--------' + algo + '--------' + '\n')
        self.fw.write(title + '\n')
        for i in range(len(result_list)):
            context = '\t'.join([str(i + 1)] + [str(x) for x in result_list[i]])
            print(context)
            self.fw.write(context + '\n')

        for i in range(n_metric):
            avg_list.append(np.mean([x[i] for x in result_list]))
            std_list.append(np.std([x[i] for x in result_list]))

        avg = '\t'.join(['avg'] + [str(round(x, 5)) for x in avg_list])
        std = '\t'.join(['std'] + [str(round(x, 5)) for x in std_list])
        self.fw.write(avg + '\n')
        self.fw.write(std + '\n')
        print(avg)
        print(std)


    def AP(self, score_label, n_test_gene):
        # Compute the average precision (AP) of a list of ranked items
        pre_list = list()
        hit_counter = 0
        k = len(score_label)
        for i in range(k):
            hit = score_label[i][1]
            if hit == 1:
                hit_counter += 1
                pre_list.append(hit_counter / (i + 1))

        ap = sum(pre_list) / min(k, n_test_gene)
        return ap

    def print_result(self):
        ks_list = ['HR@' + str(k) for k in self.ks] + ['MAP@' + str(k) for k in self.ks]
        title = '\t'.join(['fold', 'n_test_dis', 'n_pred_dis'] + ks_list)
        print(title)
        n_metric = len(self.result[0]) - 1
        avg_list = list()
        std_list = list()
        with open(self.out_file, 'w', encoding='utf8') as fw:
            fw.write(title + '\t' 'metric' + '\n')
            for i in range(len(self.result)):
                context = '\t'.join([str(i + 1)] + [str(x) for x in self.result[i][0:n_metric]])
                context1 = '\t'.join([str(i + 1)] + [str(x) for x in self.result[i]])
                print(context)
                fw.write(context1 + '\n')

            for i in range(n_metric):
                avg_list.append(np.mean([x[i] for x in self.result]))
                std_list.append(np.std([x[i] for x in self.result]))

            avg = '\t'.join(['avg'] + [str(round(x, 5)) for x in avg_list])
            std = '\t'.join(['std'] + [str(round(x, 5)) for x in std_list])
            fw.write(avg)
            fw.write(std)
            print(avg)
            print(std)


    def cal_metrics(self, score_list, label_list, n_d_test_gene):
        fpr, tpr, thresholds = roc_curve(label_list, score_list, pos_label=1)
        auc_val1 = auc(fpr, tpr)

        self.fpr_dict[self.auc_counter] = fpr
        self.tpr_dict[self.auc_counter] = tpr
        self.auc_counter += 1

        # considering the predicted candidate genes, calculate auc
        label_list_new = list()
        score_list_new = list()
        for i in range(len(label_list)):
            if score_list[i] != 0:
                score_list_new.append(score_list[i])
                label_list_new.append(label_list[i])
        if len(label_list_new) == 0:
            auc_val2 = 0
        else:
            fpr, tpr, thresholds = roc_curve(label_list_new, score_list_new, pos_label=1)
            auc_val2 = auc(fpr, tpr)

        # calculate HR@K
        score_label = [[score_list[x], label_list[x]] for x in range(len(label_list))]
        score_label.sort(key=lambda x: x[0], reverse=True)
        n_hit_list = list()
        for k in self.ks:
            n_hit = sum([x[1] for x in score_label[0:k]])  # 1:命中；0:未命中
            n_hit_list.append(n_hit)

        # calculate AP
        ap_list = list()
        for k in self.ks:
            ap = self.AP(score_label[0:k], n_d_test_gene)
            ap_list.append(ap)

        return [auc_val1, auc_val2], n_hit_list, ap_list

    def metric_avg(self, test_metric_list):

        auc_list = [x[0] for x in test_metric_list]
        hit_list = [x[1] for x in test_metric_list]
        ap_list = [x[2] for x in test_metric_list]
        n_test_gene_all = sum([x[3] for x in test_metric_list])

        HR_list = list()
        MAP_list = list()
        for i in range(len(self.ks)):
            n_hit_all = sum([x[i] for x in hit_list])
            HR = n_hit_all / n_test_gene_all  # hit ratio
            MAP = sum([x[i] for x in ap_list]) / len(ap_list)
            HR_list.append(round(HR, 5))
            MAP_list.append(round(MAP, 5))

        # calculate avg. of auc
        auc_avg1 = round(np.mean([x[0] for x in auc_list]), 5)
        auc_std1 = round(np.std([x[0] for x in auc_list]), 5)
        auc_avg2 = round(np.mean([x[1] for x in auc_list]), 5)
        auc_std2 = round(np.std([x[1] for x in auc_list]), 5)

        return [auc_avg1, auc_std1, auc_avg2, auc_std2], HR_list, MAP_list
        # return [str(auc_avg1) + '±' + str(auc_std1), str(auc_avg2) + '±' + str(auc_std2)] + hit_rate_list

    def load_pre_data(self):
        pre_dict = dict()
        with open(self.pre_file, 'r', encoding='utf8') as fr:
            for line in fr:
                dis, gene, score = line.strip().split('\t')
                pre_dict.setdefault(dis, dict())
                pre_dict[dis][gene] = float(score)
        return pre_dict

    def load_cv_data(self):
        cv_dict = dict()
        with open(self.cv_file, 'r', encoding='utf8') as fr:
            for line in fr:
                cv_i, type, d, g = line.strip().split('\t')
                cv_dict.setdefault(cv_i, dict())
                cv_dict[cv_i].setdefault(type, list())
                cv_dict[cv_i][type].append([d, g])
        return cv_dict


if __name__ == '__main__':
    e = Evaluation()
