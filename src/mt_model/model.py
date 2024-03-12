import os
import time
import sys
sys.path.append('../')
import logging
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from common.censored_loss import cross_entropy_loss

# logger
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

class Model():
    def __init__(self, feature_vocab, LEARNING_RATE=0.0005, EPOCH=20, TRAIN_BATCH_SIZE=10240, PREDICT_Z_SIZE=301,
        log_cb=None):
        self.feature_vocab = feature_vocab
        self.LEARNING_RATE = LEARNING_RATE
        self.EPOCH = EPOCH
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.PREDICT_Z_SIZE = PREDICT_Z_SIZE
        self.log_cb = log_cb
        self.win_rate_model = self.generate_model(feature_vocab=self.feature_vocab, PREDICT_Z_SIZE=self.PREDICT_Z_SIZE)
        self.log_cb.import_model = self.win_rate_model # inference by the cdf model
        self.target_model = self.win_rate_model
 
    def fit_mixed_val(self, censored_dataset):
        for epoch in range(self.EPOCH):
            print("Epoch {}/{}".format(epoch + 1, self.EPOCH))
            batch_size = len(censored_dataset)
            first_cross_entropy_loss = []
            second_cross_entropy_loss = []
            logs = defaultdict(list)
            for step, batch_data in enumerate(censored_dataset):
                batch_X, batch_Z, batch_B = batch_data.data.X, batch_data.data.Z, batch_data.data.B
                batch_X = [batch_X.values[:, k] for k in range(batch_X.values.shape[1])]
                batch_win_mask = batch_Z < batch_B
                w = np.reshape(batch_win_mask, (batch_win_mask.shape[0], 1))
                b = np.reshape(batch_B, (batch_B.shape[0], 1))
                z = np.reshape(batch_Z, (batch_Z.shape[0], 1))
                total_y = np.concatenate((w, b, z), axis=1)
                if batch_data.first_flag:
                    loss = self.win_rate_model.train_on_batch(x=[batch_X, batch_B],
                                                y=[total_y]) #
                    first_cross_entropy_loss.append(loss)
                else:
                    loss = self.pdf_model.train_on_batch(x=[batch_X, batch_B],
                                                y=[total_y]) #
                    second_cross_entropy_loss.append(loss)

        
                # ignore the warning of empty slice for np.mean()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_first_total_cross_entropy_loss = np.mean(first_cross_entropy_loss)
                    mean_second_total_cross_entropy_loss = np.mean(second_cross_entropy_loss)
                print("{:.2%}  Step[{}/{}].first_ce:{:.5f} second_ce:{:.5f} ".format((step + 1) / batch_size, step + 1, batch_size,
                                                                            mean_first_total_cross_entropy_loss, mean_second_total_cross_entropy_loss), end='\r')
                
            logs['train_first_cross_entropy'].append(mean_first_total_cross_entropy_loss)
            logs['train_second_cross_entropy'].append(mean_second_total_cross_entropy_loss)
            logger.info(' first_ce:{:.5f} second_ce:{:.5f} '.format(mean_first_total_cross_entropy_loss, mean_second_total_cross_entropy_loss))
            if not self.log_cb.on_epoch_end(epoch+1, logs):
                break

    def fit_mmcp_val(self, censored_dataset):
        for epoch in range(self.EPOCH):
            print("Epoch {}/{}".format(epoch + 1, self.EPOCH))
            batch_size = len(censored_dataset)
            first_cross_entropy_loss = []
            second_cross_entropy_loss = []
            full_losses = []
            logs = defaultdict(list)
            for step, batch_data in enumerate(censored_dataset):
                # print('batch data len: {}'.format(len(batch_data.data.X)))
                batch_X, batch_Z, batch_B = batch_data.data.X, batch_data.data.Z, batch_data.data.B
                batch_X = [batch_X.values[:, k] for k in range(batch_X.values.shape[1])]
                batch_win_mask = batch_Z < batch_B
                w = np.reshape(batch_win_mask, (batch_win_mask.shape[0], 1))
                b = np.reshape(batch_B, (batch_B.shape[0], 1))
                z = np.reshape(batch_Z, (batch_Z.shape[0], 1))
                total_y = np.concatenate((w, b, z), axis=1)
                if batch_data.first_flag:
                    loss = self.full_model.train_on_batch(x=[batch_X, batch_B],
                                y=[total_y, total_y]) # , total_y
                    first_cross_entropy_loss.append(loss)
                    # loss = self.win_rate_model.train_on_batch(x=[batch_X, batch_B],
                    #                             y=[total_y]) #
                    # first_cross_entropy_loss.append(loss)
                else:
                    loss = self.pdf_model.train_on_batch(x=[batch_X, batch_B],
                                                y=[total_y]) #
                    # loss = self.full_model.train_on_batch(x=[batch_X, batch_B],
                    #             y=[total_y, total_y]) #
                    second_cross_entropy_loss.append(loss)
                    # full_losses.append(full_loss)

                # ignore the warning of empty slice for np.mean()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_first_total_cross_entropy_loss = np.mean(first_cross_entropy_loss)
                    mean_second_total_cross_entropy_loss = np.mean(second_cross_entropy_loss)
               
                print("{:.2%}  Step[{}/{}].first_ce:{:.5f} second_ce:{:.5f} ".format((step + 1) / batch_size, step + 1, batch_size,
                                                                            mean_first_total_cross_entropy_loss, mean_second_total_cross_entropy_loss), end='\r')
                
            logs['train_first_cross_entropy'].append(mean_first_total_cross_entropy_loss)
            logs['train_second_cross_entropy'].append(mean_second_total_cross_entropy_loss)
            logger.info(' first_ce:{:.5f} second_ce:{:.5f} '.format(mean_first_total_cross_entropy_loss, mean_second_total_cross_entropy_loss))
            if not self.log_cb.on_epoch_end(epoch+1, logs):
                break