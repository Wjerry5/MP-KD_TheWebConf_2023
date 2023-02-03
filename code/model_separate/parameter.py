import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run DyRec.")

    # Setup parameters:
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='GPU or CPU')
    parser.add_argument('--device', type=int, default=0,
                        help='device id')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed.')
    

    # Data loading parameters:
    parser.add_argument('--dataset', nargs='?', default='wikipedia',
                        help='Choose a dataset from {wikipedia}')
    parser.add_argument('--data_dir', nargs='?', default='../Data',
                        help='Input data path.')

    parser.add_argument('--k_shots', type=int, default=1,
                        help='K-shot learning.')
    parser.add_argument('--neg_factor', type=int, default=1,
                        help='negative sampling rate.')  

    parser.add_argument('--val_ratio', type=float, default=0.5,
                        help='Val ratio of time split.') 
    parser.add_argument('--test_ratio', type=float, default=0.7,
                        help='Test ratio of time split.')   

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
    parser.add_argument('--n_head', type=int, default=1,
                        help='Number of heads of attn.') 
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Dimension of embedding vectors.')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='Dimension of embedding vectors.')
    parser.add_argument('--edge_dim', type=int, default=128,
                        help='Dimension of embedding vectors.') 
    parser.add_argument('--time_dim', type=int, default=128,
                        help='Dimension of embedding vectors.')                   
    parser.add_argument('--use_pretrain', type=bool, default=True,
                        help='Load pretrained embeddings.')
    parser.add_argument('--pretrain_embedding_rel', type=str, default='rel_emb_transe_YAGO_1.npy',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_embedding_ent', type=str, default='ent_emb_transe_YAGO_1.npy',
                        help='Path of stored model.')
    parser.add_argument('--residual', type=bool, default=True,
                        help='Add residual cross layers.')  

    # Training hyperparameters:
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--encoder_lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--local_lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=30,
                        help='Number of epoch for early stopping.')
    parser.add_argument('--inner_update', type=int, default=1,
                        help='Number of updates for inner optimization.')

    parser.add_argument('--margin', type=float, default=0.5,
                        help='margin value in ranking loss.')

    parser.add_argument('--test_epoch', type=int, default=10,
                        help='Number of epoch.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='recommendation batch size.')
    parser.add_argument('--query_batch', type=int, default=300,
                        help='recommendation batch size.')
    parser.add_argument('--test_batch', type=int, default=1000,
                        help='recommendation batch size.')

    parser.add_argument('--report_loss', type=int, default=1,
                        help='Iter interval of printing recommendation loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--l2_weight', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    
    args = parser.parse_args()


    save_dir = '../trained_model/{}/entitydim{}_lr{}_pretrain{}/'.format(
        args.dataset, args.emb_dim, args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args