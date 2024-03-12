import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(description='MTL BLF')

    parser.add_argument('--exp_kind', default='pure', type=str) # 'pure' first or 'mtl' (first and second)
    parser.add_argument('--exp_name', default='shared_bottom', type=str)
    parser.add_argument('--model', default='AB_Model', type=str)
    parser.add_argument('--cdf_predict', default=0, type=int)
    parser.add_argument('--predict_type', default='softmax', type=str)
    parser.add_argument('--random_bid', default=0, type=int)
    parser.add_argument('--dataset', default='ipinyou', type=str)
    parser.add_argument('--camp', default='2259', type=str)

    parser.add_argument('--win_lose_flag', default=0, type=int)

    parser.add_argument('--train_bs', default=10240, type=int)
    parser.add_argument('--predict_bs', default=10240, type=int)

    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--epoch', default=60, type=int)

    parser.add_argument('--z_size', default=301, type=int)

    parser.add_argument('--first_loss_weight', default=0.1, type=float)
    parser.add_argument('--second_loss_weight', default=0.9, type=float)

    parser.add_argument('--B_START', default=0, type=int)
    parser.add_argument('--B_LIMIT', default=10000, type=int)
    parser.add_argument('--B_DELTA', default=10, type=int)

    parser.add_argument('--bid_prop', default=0.01, type=float)

    parser.add_argument('--cdf_type', default='ab', type=str)


    return parser

