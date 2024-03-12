from common.censored_loss import cross_entropy_loss, cross_entropy_loss_second, cross_entropy_loss_delta, cross_entropy_loss_mmcp, wd_loss
from mt_model.model import Model
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
from tensorflow.keras.layers import DenseFeatures, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from collections import defaultdict
import warnings
import logging
import os
import time
import sys
sys.path.append('../')

# logger
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)


class DMoELayer(tf.keras.layers.Layer):
    '''
    @param n_experts: list,每个任务使用几个expert。[3,4]第一个任务使用3个expert，第二个任务使用4个expert。
    @param n_expert_share: int,共享的部分设置的expert个数。
    @param expert_dim: int,每个专家网络输出的向量维度。
    @param n_task: int,任务个数。
    '''

    def __init__(self, n_task, n_experts, expert_dim, n_expert_share, dnn_reg_l2=1e-5):
        super(DMoELayer, self).__init__()
        self.n_task = n_task
        # 定义多个任务特定网络和1个共享网络
        self.E_layer = []
        for i in range(n_task):
            sub_exp = [Dense(expert_dim, activation='relu')
                       for j in range(n_experts[i])]
            self.E_layer.append(sub_exp)
        self.share_layer = [Dense(expert_dim, activation='relu')
                            for j in range(n_expert_share)]
        # 定义门控网络
        self.gate_layers = [Dense(n_expert_share+n_experts[i], kernel_regularizer=regularizers.l2(dnn_reg_l2),
                                  activation='softmax') for i in range(n_task)]

    def call(self, x):
        # 特定网络和共享网络
        E_net = [[expert(x) for expert in sub_expert]
                 for sub_expert in self.E_layer]
        share_net = [expert(x) for expert in self.share_layer]
        # 【门权重】和【指定任务及共享任务输出】的乘法计算
        towers = []
        for i in range(self.n_task):
            g = self.gate_layers[i](x)
            # 维度 (bs,n_expert_share+n_experts[i],1)
            g = tf.expand_dims(g, axis=-1)
            _e = share_net+E_net[i]
            # 维度 (bs,n_expert_share+n_experts[i],expert_dim)
            _e = Concatenate(axis=1)(
                [expert[:, tf.newaxis, :] for expert in _e])
            _tower = tf.matmul(_e, g, transpose_a=True)
            towers.append(Flatten()(_tower))  # 维度 (bs,expert_dim)
        return towers


class MMCP(Model):
    def __init__(self, feature_vocab, LEARNING_RATE=0.0005, EPOCH=20, TRAIN_BATCH_SIZE=10240, PREDICT_Z_SIZE=301,
                 PDF_LOSS_WEIGHT=0.2, WIN_RATE_LOSS_WEIGHT=0.8, log_cb=None):
        super(MMCP, self).__init__(feature_vocab, LEARNING_RATE, EPOCH, TRAIN_BATCH_SIZE, PREDICT_Z_SIZE,
        log_cb)
        self.feature_vocab = feature_vocab
        self.LEARNING_RATE = LEARNING_RATE
        self.EPOCH = EPOCH
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.PREDICT_Z_SIZE = PREDICT_Z_SIZE
        self.PDF_LOSS_WEIGHT = PDF_LOSS_WEIGHT
        self.WIN_RATE_LOSS_WEIGHT = WIN_RATE_LOSS_WEIGHT
        self.log_cb = log_cb
        self.full_model, self.pdf_model, self.win_rate_model = self.generate_model(feature_vocab=self.feature_vocab,
                                                                                   PREDICT_Z_SIZE=self.PREDICT_Z_SIZE,
                                                                                   PDF_LOSS_WEIGHT=self.PDF_LOSS_WEIGHT,
                                                                                   WIN_RATE_LOSS_WEIGHT=self.WIN_RATE_LOSS_WEIGHT,
                                                                                   )
        self.log_cb.import_model = self.full_model

    def generate_model(self, feature_vocab, PREDICT_Z_SIZE=301,  PDF_LOSS_WEIGHT=0.1, WIN_RATE_LOSS_WEIGHT=0.9):
        """
        :param feature_vocab: vocabulary of each features: dict
        :return: Win Rate model:  tensorflow.python.keras.engine.training.Model
        """

        embeddings = []  # put embeddings
        inputs = []  # put inputs

        # 1. feature-wise embedding
        for feature in feature_vocab.keys():
            vocab = feature_vocab[feature]
            embed_dim = int(min(np.ceil(vocab / 2), 50))
            input_layer = layers.Input(shape=(1,),
                                       name='input_' + '_'.join(feature.split(' ')))
            embed_layer = layers.Embedding(vocab, embed_dim, trainable=True,
                                           embeddings_initializer=tf.keras.initializers.he_normal())(input_layer)
            embed_reshape_layer = layers.Reshape(
                target_shape=(embed_dim,))(embed_layer)
            embeddings.append(embed_reshape_layer)
            inputs.append(input_layer)

        bid_price_input = layers.Input(shape=(1,), name='bid_price_input')

        concated_embeddings = layers.concatenate(
            embeddings, name='input_merge')
        price_bucket_num = 301
        l2_reg = 0.001

        ### 1-Order Feature Extractor ###
        x_o1_tensor = concated_embeddings
        ### High-Order Feature Extractor ###
        from tensorflow.keras import regularizers
        x_oh_tensor = layers.Dense(price_bucket_num/4, activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_reg), name='oh_Dense_1')(x_o1_tensor)
        x_oh_tensor = layers.Dense(price_bucket_num/2, activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_reg), name='oh_Dense_2')(x_oh_tensor)

        # Shared-bottom
        shared_tensor = layers.Concatenate(axis=1)([x_o1_tensor, x_oh_tensor])

        dmoe_layers = DMoELayer(2, [4, 4], 500, 0)(shared_tensor)

        output_layers = []

        output_info = ['y0', 'y1']

        # Build tower layer from dmoe layer
        for index, task_layer in enumerate(dmoe_layers):
            tower_layer = Dense(
                units=8,
                activation='relu',
                kernel_initializer=VarianceScaling())(task_layer)
            output_layer = Dense(
                units=1,
                name=output_info[index],
                activation='linear',
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)

        ### CDF Output Layer ###
        output_tensor = layers.Dense(price_bucket_num, kernel_regularizer=regularizers.l2(
            l2_reg), name='concat_Dense_first')(output_layers[0])
        output_tensor = layers.Softmax(name='ground_truth')(output_tensor)

        ### PDF Output Layer ###
        mp_tensor = layers.Dense(price_bucket_num, kernel_regularizer=regularizers.l2(
            l2_reg), name='concat_Dense_second')(output_layers[1])
        mp_output = layers.Softmax(name='second_ground_truth')(mp_tensor)

        # PDF Model
        pdf_model = tf.keras.models.Model(
            inputs=[inputs, bid_price_input], outputs=[mp_output])
        pdf_model.compile(loss=[cross_entropy_loss_second],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))

        # CDF Model
        win_rate_model = tf.keras.models.Model(
            inputs=[inputs, bid_price_input], outputs=[output_tensor])
        
        full_model_output = layers.concatenate(
            [output_tensor, mp_output], name='full_model_y')
    
        # full_model_output2 = layers.concatenate(
        #     [output_tensor, mp_output], name='full_model_y2')

        # Mixed Full Model
        # full_model = tf.keras.models.Model(inputs=[inputs, bid_price_input], outputs=[
        #                                    output_tensor, full_model_output, full_model_output])
        full_model = tf.keras.models.Model(inputs=[inputs, bid_price_input], outputs=[
                                   output_tensor, full_model_output])

        win_rate_model.compile(loss=[cross_entropy_loss],
                               optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))

        full_model.compile(loss=[lambda y_true, y_pred: cross_entropy_loss(y_true, y_pred),lambda y_true, y_pred: cross_entropy_loss_mmcp(y_true, y_pred)], #lambda y_true, y_pred: wd_loss(y_true, y_pred)
                           loss_weights=[WIN_RATE_LOSS_WEIGHT,
                                         PDF_LOSS_WEIGHT], # , WIN_RATE_LOSS_WEIGHT
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))
        return full_model, pdf_model, win_rate_model

    def fit_mixed(self, censored_dataset):
        for epoch in range(self.EPOCH):
            print("Epoch {}/{}".format(epoch + 1, self.EPOCH))
            batch_size = len(censored_dataset)
            first_cross_entropy_loss = []
            second_cross_entropy_loss = []
            logs = defaultdict(list)
            for step, batch_data in enumerate(censored_dataset):
                batch_X, batch_Z, batch_B = batch_data.data.X, batch_data.data.Z, batch_data.data.B
                batch_X = [batch_X.values[:, k]
                           for k in range(batch_X.values.shape[1])]
                batch_win_mask = batch_Z < batch_B
                w = np.reshape(batch_win_mask, (batch_win_mask.shape[0], 1))
                b = np.reshape(batch_B, (batch_B.shape[0], 1))
                z = np.reshape(batch_Z, (batch_Z.shape[0], 1))
                total_y = np.concatenate((w, b, z), axis=1)
                if batch_data.first_flag:
                    loss = self.win_rate_model.train_on_batch(x=[batch_X, batch_B],
                                                              y=[total_y])
                    first_cross_entropy_loss.append(loss)
                else:
                    loss = self.pdf_model.train_on_batch(x=[batch_X, batch_B],
                                                         y=[total_y])
                    second_cross_entropy_loss.append(loss)

                # ignore the warning of empty slice for np.mean()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_first_total_cross_entropy_loss = np.mean(
                        first_cross_entropy_loss)
                    mean_second_total_cross_entropy_loss = np.mean(
                        second_cross_entropy_loss)
                print("{:.2%}  Step[{}/{}].first_ce:{:.5f} second_ce:{:.5f} ".format((step + 1) / batch_size, step + 1, batch_size,
                                                                                     mean_first_total_cross_entropy_loss, mean_second_total_cross_entropy_loss), end='\r')

            logs['train_first_cross_entropy'].append(
                mean_first_total_cross_entropy_loss)
            logs['train_second_cross_entropy'].append(
                mean_second_total_cross_entropy_loss)
            logger.info(' first_ce:{:.5f} second_ce:{:.5f} '.format(
                mean_first_total_cross_entropy_loss, mean_second_total_cross_entropy_loss))
            self.log_cb.on_epoch_end(epoch+1, logs)

    def fit(self, censored_dataset):
        for epoch in range(self.EPOCH):
            print("Epoch {}/{}".format(epoch + 1, self.EPOCH))
            batch_size = len(censored_dataset)
            anlp_loss = []
            win_cross_entropy_loss = []
            lose_cross_entropy_loss = []
            logs = defaultdict(list)
            for step, batch_data in enumerate(censored_dataset):
                # second win
                if batch_data.win_flag:
                    batch_X, batch_Z, batch_B = batch_data.data.X, batch_data.data.Z, batch_data.data.B
                    batch_X = [batch_X.values[:, k]
                               for k in range(batch_X.values.shape[1])]
                    batch_win = np.ones(batch_data.size)
                    win_loss = self.full_model.train_on_batch(x=[batch_X, batch_B],
                                                              y=[batch_Z, batch_win])
                    win_cross_entropy_loss.append(win_loss[2])
                    anlp_loss.append(win_loss[1])
                else:
                    # second lose + first win + first lose
                    batch_X, batch_B = batch_data.data.X, batch_data.data.B
                    batch_X = [batch_X.values[:, k]
                               for k in range(batch_X.values.shape[1])]
                    batch_win = np.zeros(batch_data.size)
                    lose_loss = self.win_rate_model.train_on_batch(x=[batch_X, batch_B],
                                                                   y=[batch_win])
                    lose_cross_entropy_loss.append(
                        lose_loss/self.WIN_RATE_LOSS_WEIGHT)

                # ignore the warning of empty slice for np.mean()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_anlp_loss = np.mean(anlp_loss)
                    mean_win_cross_entropy_loss = np.mean(
                        win_cross_entropy_loss)
                    mean_lose_cross_entropy_loss = np.mean(
                        lose_cross_entropy_loss)
                    mean_total_cross_entropy_loss = np.mean(
                        win_cross_entropy_loss + lose_cross_entropy_loss)
                print("{:.2%}  Step[{}/{}]. Win flag: {}.  anlp: {:.5f} win_ce: {:.5f} lose_ce: {:.5f} total_ce:{:.5f}".format((step + 1) / batch_size, step + 1, batch_size, batch_data.win_flag,
                                                                                                                               mean_anlp_loss,
                                                                                                                               mean_win_cross_entropy_loss, mean_lose_cross_entropy_loss, mean_total_cross_entropy_loss), end='\r')
            logs['train_mean_anlp_loss'].append(mean_anlp_loss)
            logs['train_win_cross_entropy'].append(mean_win_cross_entropy_loss)
            logs['train_lose_cross_entropy'].append(
                mean_lose_cross_entropy_loss)
            logs['train_total_cross_entropy'].append(
                mean_total_cross_entropy_loss)
            logger.info(' anlp: {:.5f} win_ce: {:.5f} lose_ce: {:.5f} total_ce:{:.5f}'.format(mean_anlp_loss,
                                                                                              mean_win_cross_entropy_loss, mean_lose_cross_entropy_loss, mean_total_cross_entropy_loss))

            self.log_cb.on_epoch_end(epoch+1, logs)