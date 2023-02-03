import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run DyRec.")

    # Setup parameters:
    parser.add_argument('--use_cuda', type=int, default=1,
                        help='GPU or CPU')
    parser.add_argument('--device', type=int, default=0,
                        help='device id')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed.')
    

    # Data loading parameters:
    parser.add_argument('--data_dir', type=str, default='../data/', help = 'data directory')
    parser.add_argument('--data_name', nargs='?', default='wiki',
                        help='Choose a dataset from {wiki, yago}')
    parser.add_argument('--task', nargs='?', default='en-it',
                        help='Source-Target.')
    parser.add_argument('--method', type=str, default="target")
    parser.add_argument('--load_old_data', type=int, default=1)
    parser.add_argument('--neg_factor', type=int, default=15,
                        help='negative sampling rate.')  
    parser.add_argument('--train_time', type=int, default=28)
    parser.add_argument('--test_time', type=int, default=12)
    parser.add_argument('--load_model', type=int, default=0)


    # Model hyperparameters:
    parser.add_argument('--score_function', type=str, default='TransE',
                        help='score function for KGC.') 
    parser.add_argument('--num_neighbors', type=int, default=8,
                        help='Number of neighbors for sampling.') 
    parser.add_argument('--attn_mode', type=str, default='multi',
                        help='Attention mode of encoder.') 
    parser.add_argument('--use_time', type=str, default='time',
                        help='Flag of whether use time encoding.') 
    parser.add_argument('--agg_method', type=str, default='attn',
                        help='Neborhood aggration model.') 
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers of encoder.') 
    parser.add_argument('--drop_out', type=float, default=0.5,
                        help='Dropout rate.') 
    parser.add_argument('--n_head', type=int, default=2,
                        help='Number of heads of attn.') 
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Dimension of embedding vectors.')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='Dimension of embedding vectors.')
    parser.add_argument('--edge_dim', type=int, default=128,
                        help='Dimension of embedding vectors.') 
    parser.add_argument('--time_dim', type=int, default=128,
                        help='Dimension of embedding vectors.')                   
    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='Load pretrained embeddings.')
    parser.add_argument('--residual', type=int, default=1,
                        help='Add residual cross layers.')  

    # Training hyperparameters:
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--n_epoch', type=int, default=50,
                        help='Number of epoch.')

    parser.add_argument('--margin', type=float, default=0.5,
                        help='margin value in ranking loss.')

    parser.add_argument('--test_epoch', type=int, default=10,
                        help='Number of epoch.')

    parser.add_argument('--train_batch', type=int, default=256,
                        help='training batch size.')
    parser.add_argument('--test_batch', type=int, default=256,
                        help='training batch size.')

    parser.add_argument('--l2_weight', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    
    args = parser.parse_args()

    if not os.path.exists(os.path.join("../log", args.data_name, args.task)):
        os.makedirs(os.path.join("../log", args.data_name, args.task))
    args.log_file = os.path.join("../log", args.data_name, args.task, 'log.txt')
    args.model_file = os.path.join("../log", args.data_name, args.task, 'model.pt')
    args.result_file = os.path.join("../log", args.data_name, args.task, 'result.json')

    return args