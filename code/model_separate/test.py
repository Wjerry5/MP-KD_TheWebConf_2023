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



    

