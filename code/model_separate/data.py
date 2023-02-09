import numpy as np
import pandas as pd
import random
from utils import NeighborFinder
import os

class Dataloader:
    def __init__(self, args):
        
        self.args = args
        random.seed(args.seed)

        self.task = args.task
        
        self.neg_factor = args.neg_factor

        self.load_data()

    def load_data(self):

        # load data:
        # data.target_train_data: training data in target language;
        # data.target_val_data: validation data in target language;
        # data.target_int_data: masked training data in target language;
        # data.target_ext_data: testing data in target language;
        # data.source_train_data: training data in source language;
        # data.alignment_train_data: available alignment data during training;
        # data.source_ent_num: total entity amount on source language;
        # data.target_ent_num: total entity amoung on target language;

        data_files = ["source.csv", "target_train.csv", "target_val.csv", "target_int.csv", "target_ext.csv", "alignment_train.csv", "alignment_test.csv"]
        feat_files = ["ent_embedding.npy", "rel_embedding.npy"]
        self.data = {"source": None, "target_train": None, "target_val": None, "target_int": None, "target_ext": None, "alignment_train": None, "alignment_test": None, 
                        "source_ent_embedding": None, "target_ent_embedding": None, "rel_embedding": None}
        for file in data_files:
            file_path = os.path.join(self.args.data_dir, self.args.data_name, self.task, file)
            self.data[file.split(".")[0]] = pd.read_csv(file_path)
        
        self.source_ent_num = max([self.data["source"]["subject"].max(), self.data["source"]["object"].max()]) + 1
        self.target_ent_num = max([self.data["target_train"]["subject"].max(), self.data["target_train"]["object"].max(), 
                    self.data["target_int"]["subject"].max(), self.data["target_int"]["object"].max()]) + 1
        self.rel_num = max([self.data["source"]["relation"].max(), self.data["target_train"]["relation"].max()]) + 2 # last for alignment
        if self.args.use_pretrain:
            try:
                ent_emb, rel_emb = np.load(os.path.join(self.args.data_dir, self.args.data_name, self.task, feat_files[0])), np.load(os.path.join(self.args.data_dir, self.args.data_name, self.task, feat_files[1]))
            except:
                ent_emb, rel_emb = np.random.normal(0.5, 0.25, size=(self.source_ent_num + self.target_ent_num, self.args.node_dim)), np.random.normal(0.5, 0.25, size=(self.rel_num, self.args.edge_dim))
        else:
            ent_emb, rel_emb = np.random.normal(0.5, 0.25, size=(self.source_ent_num + self.target_ent_num, self.args.node_dim)), np.random.normal(0.5, 0.25, size=(self.rel_num, self.args.edge_dim))
        assert ent_emb.shape[0] == self.source_ent_num + self.target_ent_num
        assert rel_emb.shape[0] == self.rel_num

        self.data["source_ent_embedding"] = ent_emb[:self.source_ent_num]
        self.data["target_ent_embedding"] = ent_emb[self.source_ent_num:]
        self.data["rel_embedding"] = rel_emb

        self.source_train_data, self.source_ngh_finder = self.train_data_KG("source")
        self.target_train_data, self.target_ngh_finder = self.train_data_KG("target")
        self.target_val_data = self.test_data_KG("target_val")
        self.target_int_data = self.test_data_KG("target_int")
        self.target_ext_data = self.test_data_KG("target_ext")
        self.alignment_train_data = self.train_data_alignment("alignment_train")
        # self.alignment_test_data = self.test_data_alignment("alignment_test")

    def train_data_KG(self, language):
        
        node2hist, node_set = {}, set()

        if language == "source":
            data  = self.data["source"]
            node_set = set(range(self.source_ent_num))
        elif language == "target":
            data = self.data["target_train"]
            node_set = set(range(self.target_ent_num))

        # construct positive and negative samples for model training:

        if self.args.load_old_data and os.path.exists(os.path.join(self.args.data_dir, self.args.data_name, self.task, "train_data_{}.npy".format(language))):
            train_data = np.load(os.path.join(self.args.data_dir, self.args.data_name, self.task, "train_data_{}.npy".format(language)), allow_pickle=True)
            pos_all, neg_all = train_data[0], train_data[1]
        else:

            for src, dst, rel, ts in zip(data["subject"], data["object"], data["relation"], data["time"]):

                if src not in node2hist:
                    node2hist[src] = list()
                node2hist[src].append((src, dst, rel, ts))

                if dst not in node2hist:
                    node2hist[dst] = list()
                node2hist[dst].append((src, dst, rel, ts))

            pos_all, neg_all = [], []
            
            # the negative sampling is performed per each entity;
            for node in node2hist:

                pos =  np.array(sorted(node2hist[node], key=lambda x: x[3]))
                false_entities = np.array(list(node_set - set(pos[:,0]) - set(pos[:,1])))
                neg = np.tile(pos, (self.neg_factor, 1))
                neg[neg[:,0] == node,1] = np.random.choice(false_entities, int(sum(neg[:,0] == node)))
                neg[neg[:,1] == node,0] =  np.random.choice(false_entities, int(sum(neg[:,1] == node)))
                pos_all += list(pos)
                neg_all += list(neg)
            
            np.save(os.path.join(self.args.data_dir, self.args.data_name, self.task, "train_data_{}.npy".format(language)), [pos_all, neg_all])

        train_pos, train_neg = [], []
        for i in range(0, len(pos_all), self.args.train_batch):
            train_pos.append(np.array(pos_all[i:i+self.args.train_batch]))
            train_neg.append(np.array(neg_all[i * self.neg_factor:(i+self.args.train_batch) * self.neg_factor]))

        train_data = list(zip(train_pos, train_neg))
        
        # construct the neighbor finder for the representation module to integrate neighbor information:
        max_idx = max(node_set)

        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(data["subject"], data["object"], data["relation"], data["time"]):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))

        train_ngh_finder = NeighborFinder(adj_list, uniform=False)

        print("DONE: prepare {} training data set, total batch {}...".format(language, len(train_data)))
        return train_data, train_ngh_finder

    def test_data_KG(self, language):
        data = self.data[language]
        src, dst, rel, ts = data["subject"], data["object"], data["relation"], data["time"]
        test_data = list(zip(src, dst, rel, ts))
        # base, test_data_batch = 0, []
        # for i in range(0, len(test_data), self.args.test_batch):
        #     test_data_batch.append(test_data[i:i+self.args.test_batch])
        print("DONE: prepare {} data set, total triplets {}...".format(language, len(test_data)))
        return test_data

    
