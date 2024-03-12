import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics


epsilon = sys.float_info.epsilon


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)


def anlp(z_true, z_pdf, B_START=0, B_DELTA=1):
    z_size = len(z_true)
    nlp = np.zeros(shape=(z_size, ))
    for i in range(z_size):
        nlp[i] = -np.log(z_pdf[i][int((int(z_true[i]) - B_START) / B_DELTA - 1)] + epsilon)
    return np.sum(nlp) / z_size


def anlp_fixed(z_true, z_pdf):
    total_num = len(z_true)
    return sum([-np.log(z_pdf[z] + epsilon) for z in z_true]) / total_num


def anlp_z_prob(z_prob):
    return sum(-np.log(z_prob + epsilon)) / len(z_prob)

def bid_shading(pdf, b, B_START=0, B_DELTA=1, B_LIMIT=300):
    bs = [] # 根据pdf进行出价
    MAX_IDX = int((B_LIMIT - B_START) / B_DELTA - 1)
    bs_idx = []
    cdf = []
    if isinstance(b, pd.core.series.Series):
        b = b.values
    bid_idx = np.linspace(B_START, MAX_IDX, MAX_IDX, endpoint=False, dtype='int32')
    for i in range(len(pdf)):
        cdf_tmp = pdf[i].cumsum()
        surplus = (b[i]-(B_START+bid_idx*B_DELTA))*cdf_tmp[bid_idx]
        bs_idx.append(np.argmax(surplus))
        bs.append(B_START + np.argmax(surplus)*B_DELTA)
        cdf.append(cdf_tmp)
    return bs, bs_idx, cdf 


def cal_test_metrics(label, pred, ecpm, B_START=0, B_DELTA=1, B_LIMIT=300):
    # params: label (Z_test) (B, 1) winning price
    #         pred (Z_pred) (B, price_bucket_num)  bucket prob
    #         ecpm (B_test) (B, 1) we use the original B_test as the ecpm value for bid shading
    bidding_price, bidding_index, _ = bid_shading(pred, ecpm, B_START=B_START, B_DELTA=B_DELTA, B_LIMIT=B_LIMIT)
    surplus = 0
    opt_surplus = 0
    win_rate_true = []
    win_rate_pred = []
    wins = []
    if isinstance(ecpm, pd.core.series.Series):
        ecpm = ecpm.values

    bs = len(pred)
    for i in range(bs):
        if bidding_price[i] >= label[i]:
            surplus += ecpm[i] - bidding_price[i]
            win_rate_true.append(1)
            wins.append(1)
        else:
            win_rate_true.append(0)
            wins.append(0)
        win_rate_pred.append(sum(pred[i][0:bidding_index[i]]))
        if ecpm[i] >= label[i]:
            opt_surplus += ecpm[i] - label[i]

    surplus_rate = surplus / opt_surplus
    # error
    if all(x == 0 for x in win_rate_true) or all(x == 1 for x in win_rate_true):
        test_auc = 1
        test_bce = 0
    else:
        test_auc = metrics.roc_auc_score(win_rate_true, win_rate_pred)
        test_bce = metrics.log_loss(win_rate_true, win_rate_pred)
    number_of_wins = sum(wins) / len(wins)
    return surplus, surplus_rate, test_auc, number_of_wins, test_bce




