import random
import time
import datetime
import numpy as np
import torch
from config import parse_args
from model import Model
from data import Dataloader 
from module import *
import tqdm
import logging
import json

def training(args, model, data, language = "target"):
    logging.info('training model...')
    if args.use_cuda:
        model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)

    num_epoch = args.n_epoch
    if language == "target":
        train_data = data.target_train_data
        valid_data = data.target_val_data
        int_data = data.target_int_data
        ext_data = data.target_ext_data
    elif language == "source":
        train_data = data.source_train_data
        valid_data = None
        int_data = None
        ext_data = None

    for _ in range(1, num_epoch):  # 100
        loss = []
        start_time = time.time()

        random.shuffle(train_data)
        pbar = tqdm.tqdm(train_data, total=len(train_data))
        for i, data_ in enumerate(pbar):
            train_pos, train_neg = data_[0], data_[1]
            loss_ = model(train_pos, train_neg)

            loss_.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            loss.append(loss_.item())

            pbar.set_description('Epoch {}/{}, Loss {:.3f}, Time {:.1f}s'.format(_, num_epoch, np.mean(loss), time.time() - start_time))
        
        if _ % 5 == 0:
            val_metrics, test_metrics, model_save = evaluating(args, model, valid_data, ext_data, mode = "ext")
            logging.info('Epoch {}/{}, Loss {:.4f}.'.format(_, num_epoch, np.mean(loss)))
            logging.info('Epoch {}/{}, Val Metrics: {}.'.format(_, num_epoch, val_metrics))
            logging.info('Epoch {}/{}, Test Metrics: {}.'.format(_, num_epoch, test_metrics))
            if model_save:
                torch.save(model.state_dict(), args.model_file)
            
            model.train()

    testing(args, model, data)
    log_result(args, model)

def evaluating(args, model, val_data, test_data, mode = "ext"):
    # testing

    val_metrics, test_metrics, model_save = model.evaluate(val_data, test_data, mode)
    return val_metrics, test_metrics, model_save

def testing(args, model, data): 
    model.load_state_dict(torch.load(args.model_file))
    if args.use_cuda:
        model.cuda()
    val_metrics, ext_metrics, _ = evaluating(args, model, data.target_val_data, data.target_ext_data, mode = "ext")
    _, int_metrics, _ = evaluating(args, model, None, data.target_int_data, mode = "int")
    logging.info('Testing, Val Metrics: {}.'.format(val_metrics))
    logging.info('Testing, Ext Metrics: {}.'.format(ext_metrics))
    logging.info('Testing, Int Metrics: {}.'.format(int_metrics))

def log_result(args, model):
    result = vars(args)
    result['best_ext_metric'] = model.tester.best_ext_metric
    result['best_int_metric'] = model.tester.best_int_metric
    result['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(args.result_file, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.close()

if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])
    logging.info("---------------------Experiment Info---------------------")
    logging.info("Dataset: {}, Method: {}".format(args.task, args.method))

    if args.use_cuda:
        torch.cuda.set_device(args.device)

    data = Dataloader(args)
    model = Model(args, data.target_ngh_finder, data.target_ent_num, len(data.data["target_train"]), data.data["target_ent_embedding"], data.data["rel_embedding"])
    if args.load_model:
        logging.info("---------------------Loading Model---------------------")
        testing(args, model, data)
    else: 
        logging.info("---------------------Training Model---------------------")
        training(args, model, data)



    

