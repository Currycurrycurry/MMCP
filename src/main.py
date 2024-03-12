import os
import time
import datetime
import sys
sys.path.append('/root/MMCP/src')
import logging
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from config import base_config
from util.processor import Processor, EncodeData
from util.censored_prosessor import CensoredProcessor
from util.truthful_bidder import TruthfulBidder
from util.parser import get_base_parser
from common.callbacks import LogSDKCallback
from mt_model.mmcp import MMCP

# NOT CHANGE IT! set seeds for reproducible exps
np.random.seed(2023)
tf.random.set_seed(2023)


# # NOT CHANGE IT!
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = get_base_parser()
args = parser.parse_args()

exp_kind = args.exp_kind
exp_name = args.exp_name
model = args.model 
predict_type = args.predict_type

# bool type
random_bid = bool(args.random_bid)
cdf_predict = bool(args.cdf_predict)
win_lose_flag = bool(args.win_lose_flag) # 表示按照win和lose进行两路数据模型的split
mixed_win_lose = not win_lose_flag


dataset = args.dataset
camp = args.camp
train_bs = args.train_bs
predict_bs = args.predict_bs
lr = args.lr
epoch = args.epoch
predict_z_size = args.z_size
first_loss_weight = args.first_loss_weight
second_loss_weight = args.second_loss_weight
B_START = args.B_START
B_LIMIT = args.B_LIMIT
B_DELTA = args.B_DELTA
bid_prop = args.bid_prop
cdf_type = args.cdf_type

# 注意：load bid需要代码手动设置！

# data settings
data_path = base_config.data_root_path
output_dir_name = '{}/{}_{}_first{}_second{}/bs{}_lr{}_epoch{}'.format(exp_name, exp_name, datetime.datetime.now(), first_loss_weight, second_loss_weight, train_bs, lr, epoch)


result_path = base_config.log_path + '{}/'.format(output_dir_name)
figure_path = result_path + 'figure/{}/'.format(camp)
pdf_path = result_path + 'pdf/{}/'.format(camp)

# logger
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

# log redirection
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = base_config.log_path + output_dir_name
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file = '{}/{}_{}_{}'.format(log_path, dataset, camp, log_time)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info(args)

# load data
if dataset == 'ipinyou':
        processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
        truthful_bidder = TruthfulBidder(dataset=dataset, campaign=camp)
        censored_processor = CensoredProcessor(processor, truthful_bidder, bid_prop)
        train_first_data, val_data, train_second_data = censored_processor.load_encode_mixed(data_type='train', 
                                                                    load_bid=False, random_bid=random_bid, mixed_win_lose=mixed_win_lose)
        test_data = censored_processor.load_encode(data_type='test_first', load_bid=False, random_bid=random_bid ,mixed=mixed_win_lose)
        X_train_first, B_train, Z_train_first = train_first_data.X, train_first_data.B, train_first_data.Z # (376693, 15) (376693,) (376693,)
        X_train_second, Z_train_second = train_second_data.X,  train_second_data.Z # (458863, 15) 
        logger.info('fpa winning rate: {}/{}={}'.format(np.count_nonzero(train_first_data.B>=train_first_data.Z), len(train_first_data.Z), np.count_nonzero(train_first_data.B>=train_first_data.Z)/len(train_first_data.Z)))
        logger.info('spa winning rate: {}/{}={}'.format(np.count_nonzero(train_second_data.B>=train_second_data.Z), len(train_second_data.Z), np.count_nonzero(train_second_data.B>=train_second_data.Z)/len(train_second_data.Z)))
        X_test, Z_test, B_test = test_data.X, test_data.Z, test_data.B # (417197, 15)
        X_val, Z_val, B_val = val_data.X, val_data.Z, val_data.B # (417197, 15)

        if exp_kind == 'pure':
            censored_dataset = censored_processor.load_dataset_mixed(train_bs, 'train', mixed_batch=mixed_win_lose, only_first=True)
            combine_data = pd.concat([X_train_first, X_val, X_test]) # (1252753, 15)
        else:
            censored_dataset = censored_processor.load_dataset_mixed(train_bs, 'train', mixed_batch=mixed_win_lose)
            combine_data = pd.concat([X_train_first, X_val, X_train_second, X_test]) # (1252753, 15)
        # feature vocabulary
        feature_vocab = dict(combine_data.nunique().items())
        del combine_data

        # transform to (feature_size, data_size) as input
        X_train_first = [X_train_first.values[:, k] for k in range(X_train_first.values.shape[1])] # 15 * (376693,)
        X_train_second = [X_train_second.values[:, k] for k in range(X_train_second.values.shape[1])] # 15 * (458863,)
        X_test = [X_test.values[:, k] for k in range(X_test.values.shape[1])] # 15 * (417197,)
        log_cb = LogSDKCallback(dataset=dataset, camp=camp, val_data=val_data, test_data=test_data, save_fig=True, figure_path=figure_path, pdf_path=pdf_path,predict_batch_size=predict_bs,cdf_flag=cdf_predict,predict_z_size=predict_z_size, B_START=B_START, B_DELTA=B_DELTA, B_LIMIT=B_LIMIT, cdf_type=cdf_type)


logger.info(feature_vocab)

# only for fpa 
if exp_kind == 'pure':
    sb = eval(model)(feature_vocab=feature_vocab,
            LEARNING_RATE=lr,
            EPOCH=epoch,
            TRAIN_BATCH_SIZE=train_bs,
            PREDICT_Z_SIZE=predict_z_size,
            log_cb=log_cb)
    sb.win_rate_model.summary(print_fn=logger.info)
    
else:
    sb = eval(model)(feature_vocab=feature_vocab,
                LEARNING_RATE=lr,
                EPOCH=epoch,
                TRAIN_BATCH_SIZE=train_bs,
                PREDICT_Z_SIZE=predict_z_size,
                PDF_LOSS_WEIGHT=second_loss_weight,
                WIN_RATE_LOSS_WEIGHT=first_loss_weight,
                log_cb=log_cb)
    sb.full_model.summary(print_fn=logger.info)

# start training
train_start_time = time.time()
logger.info('Start training. Time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start_time))))

# training
if win_lose_flag:
    sb.fit(censored_dataset)
elif model == 'MMCP':
    sb.fit_mmcp_val(censored_dataset)

    

train_end_time = time.time()
logger.info('End training. Time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_end_time))))
logger.info("Time cost of training: {:.0f}s".format(train_end_time-train_start_time))

# save the log results as xlsx file
log_cb.result_df.to_excel(result_path + '{}.xlsx'.format(camp))


