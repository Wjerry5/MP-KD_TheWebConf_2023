import numpy as np
import torch
from torch.nn import functional as F
from evaluation import Evaluation
from module import *

class Model(torch.nn.Module):
    def __init__(self, args, ngh_finder, node_num, edge_num, n_feat = None, e_feat = None):
        super(Model, self).__init__()

        self.args = args
        self.use_cuda = args.use_cuda
        self.device = torch.device(args.device if args.use_cuda else "cpu")

        self.emb_encoder = TGAT_Encoder(args, ngh_finder, node_num, edge_num, n_feat, e_feat)

        self.num_neighbors = args.num_neighbors
        self.emb_dim = args.emb_dim

        self.tester = Evaluation(args)

    def forward(self, train_pos, train_neg):
        
        try:
            all_samples = np.concatenate([np.array(train_pos).astype(int), np.array(train_neg).astype(int)], axis = 0)
        except:
            print(np.array(train_pos).shape, np.array(train_neg).shape)
        src_idx_l = all_samples.transpose()[0]
        target_idx_l = all_samples.transpose()[1]
        rel_idx_l = all_samples.transpose()[2]
        cut_time_l = all_samples.transpose()[3]

        src_embed, target_embed, rel_embed = self.emb_encoder(src_idx_l, target_idx_l, rel_idx_l, cut_time_l, self.num_neighbors)
        kg_loss = self.kg_loss(src_embed, target_embed, rel_embed)

        # return kg_loss, len(src_embed), src_embed, target_embed, rel_embed
        return kg_loss

    def kg_loss(self, head_emb, tail_emb, rel_emb):
        len_positive_triplets = int(len(head_emb) / (self.args.neg_factor + 1))
        
        if self.args.score_function == 'TransE':

            score = head_emb + rel_emb - tail_emb
            score =  - torch.norm(score, p = 2, dim = 1)

            positive_score = score[:len_positive_triplets]
            negative_score = score[len_positive_triplets:]

        elif self.args.score_function == 'DistMult':

            score = head_emb * rel_emb * tail_emb
            score = torch.sum(score, dim = 1)

            positive_score = score[:len_positive_triplets]
            negative_score = score[len_positive_triplets:]

        y = torch.ones(len_positive_triplets * self.args.neg_factor)

        if self.args.use_cuda:
            y = y.to(head_emb.device)

        positive_score = positive_score.repeat(self.args.neg_factor)

        loss = F.margin_ranking_loss(positive_score, negative_score, y, margin = self.args.margin)
        return loss

    def evaluate(self, val_data, test_data, mode):
        ent_embedding, rel_embedding = self.update_embeddings(mode)
        val_metrics, test_metrics, model_save = self.tester.evaluate(val_data, test_data, ent_embedding, rel_embedding, [1,3,10], mode)
        return val_metrics, test_metrics, model_save

    def update_embeddings(self, mode):
        self.emb_encoder.eval()
        ent_embedding, rel_embedding = self.emb_encoder.update_embeddings(self.num_neighbors, mode = mode)
        return ent_embedding, rel_embedding

