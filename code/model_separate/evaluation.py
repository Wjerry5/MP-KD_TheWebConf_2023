import numpy as np
import torch
import torch.nn.functional as F
import tqdm

class Evaluation:
    def __init__(self, args):
        self.args = args
        self.train_time = args.train_time
        self.test_time = args.test_time
        self.best_val_metric = None
        self.best_ext_metric = None
        self.best_int_metric = None
        self.model_save = False

    def evaluate(self, val_data, test_data, ent_embedding, rel_embedding, hit_at_k = [1, 3, 10], mode = "ext"):
        
        if mode == "int":
            test_metrics = self.calc_metric(test_data, ent_embedding, rel_embedding, hit_at_k, mode = "int")
            self.best_int_metric = test_metrics
            return None, test_metrics, None
        
        val_metrics = self.calc_metric(val_data, ent_embedding, rel_embedding, hit_at_k, mode = "ext")
        test_metrics = self.calc_metric(test_data, ent_embedding, rel_embedding, hit_at_k, "ext")

        if self.best_val_metric is None or val_metrics['mrr'] > self.best_val_metric['mrr']:
            self.best_val_metric = val_metrics
            self.best_ext_metric = test_metrics
            self.model_save = True
        return val_metrics, test_metrics, self.model_save

    def calc_metric(self, data, ent_embedding, rel_embedding, hit_at_k = [1, 3, 10], mode = "ext"):
        ranks = []
        for i in tqdm.tqdm(range(0, len(data), self.args.test_batch), total = len(data) // self.args.test_batch):
            test_batch = np.array(data[i:i+self.args.test_batch]).astype(int).transpose()
            s, o, r, t = test_batch[0], test_batch[1], test_batch[2], test_batch[3]
            if mode == "ext":
                t = t - self.train_time
            elif mode == "int":
                t = t
            head_emb, tail_emb, rel_emb = ent_embedding[t,s], ent_embedding[t,o], rel_embedding[r]
            rank_s = self.calc_rank(ent_embedding[t].permute(1,0,2), rel_emb, tail_emb, s, filtered_node = None)
            rank_o = self.calc_rank(head_emb, rel_emb, ent_embedding[t].permute(1,0,2), o, filtered_node = None)
            ranks += rank_s
            ranks += rank_o
        metrics = self.metric_from_rank(ranks, hit_at_k)
        return metrics

    def calc_rank(self, head_embedding, relation_embedding, tail_embedding, target, filtered_node = None):   # input np.array

        score = head_embedding + relation_embedding - tail_embedding
        score = torch.norm(score, p = 2, dim = 2)
        score = F.softmax(score, dim=1)
        if filtered_node is not None:
            filtered_node = np.array(list(filtered_node - set([target])))
            score[filtered_node] = 10000.0
        _, idx = torch.sort(score, dim=1)
        _, rank = torch.sort(idx, dim=1)
        rank = rank[target,range(len(target))].cpu().numpy()
        return list(rank + 1)

    def metric_from_rank(self, ranks, hit_at_k):
        metrics = dict()
        ranks = np.array(ranks)
        metrics['mrr'] = np.mean(1.0 / ranks)
        for k in hit_at_k:
            metrics['hit@{}'.format(k)] = np.mean((ranks <= k))
        return metrics