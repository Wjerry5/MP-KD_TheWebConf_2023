import random
import time
import datetime
import numpy as np
import torch
from config import parse_args
from model import Model
from data import Dataloader 
from module import *
from evaluation import Evaluation

def load_emb(args, data, model, mode):
    if model == "TransE":
        ent_embedding = data.data["target_ent_embedding"]
        if mode == "ext":
            ent_embedding = torch.Tensor([ent_embedding for i in range(args.test_time)]).to(args.device)
        elif mode == "int":
            ent_embedding = torch.Tensor([ent_embedding for i in range(args.train_time)]).to(args.device)
        rel_embedding = torch.Tensor(data.data["rel_embedding"]).to(args.device)

    return ent_embedding, rel_embedding

def evaluate(args, data, model = "TransE", mode = "ext"):
    # testing
    print('evaluating model...')
    tester = Evaluation(args)
    ent_embedding, rel_embedding = load_emb(args, data, model, mode)
    if mode == "ext":
        test_data = data.target_ext_data
    elif mode == "int":
        test_data = data.target_int_data
    
    metrics = tester.evaluate(test_data, ent_embedding, rel_embedding, [1,3,10], mode)
    print(metrics)
    return metrics


if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_cuda:
        torch.cuda.set_device(args.device)

    data = Dataloader(args)
    evaluate(args, data)

    # print(datetime.datetime.now())
    # testing(model, test_data, args, val_time, test_time)
    # print(datetime.datetime.now())
    # print(datetime.datetime.now())
    # node_idx_batch, edge_index_batch, edge_time_batch, node_idx_new = full_ngh_finder.get_k_hop_neighbor(range(64), [1000000] * 64)
    # print(datetime.datetime.now())
    # print(len(node_idx_batch))
    # print(edge_index_batch)
    # print(len(edge_time_batch))
    # print(len(node_idx_new))
    # 

    # n, dim = 100, 128
    # node_feat = torch.Tensor(n, dim)
    # edge_feat = torch.Tensor(5, dim)
    # edge_index = torch.Tensor([[1,5,99,2, 2],[3,87,44,66,99]]).long()
    # edge_time = torch.Tensor([1,5,6,9,10])
    # t = 1
    # temporalGrpahAttn = TemporalGraphAttn(args)
    # out = temporalGrpahAttn.forward(node_feat, edge_feat, edge_index, edge_time, t)



    

