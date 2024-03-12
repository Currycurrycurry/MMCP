import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn import metrics
from scipy.stats import entropy, wasserstein_distance
from common.tool import count_pdf, count_cdf, avg_pdf, avg_cdf, plot_mp_result, plot_mp_result_regression, plot_mp_case, pdf2cdf
from common.metrics import anlp, anlp_fixed, cal_test_metrics
from common.batch_predict import mt_batch_predict2


logger = logging.getLogger('tensorflow')


def evalute_sdk_mp_metrics(camp='sdk', val_data=None, test_data=None, output_dim=None,
                                        Z_result_pred_val=None, Z_result_pred_test=None,
                                        save_fig='', figure_path='', 
                                        pdf_path='', epoch=0, B_START=0, B_DELTA=1, B_LIMIT=300):
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)

    predict_pdf_val = Z_result_pred_val # (376693, 301)
    predict_pdf_test = Z_result_pred_test # (417197, 301)

    predict_z_size = output_dim

    Z_val = (val_data.Z).astype(int)
    Z_test = (test_data.Z).astype(int)

    if camp == 'sdk':
        ECPM_val = val_data.B.values
        ECPM_test = test_data.B.values
    else:
        ECPM_val = val_data.B
        ECPM_test = test_data.B

    count_pdf_val = count_pdf(Z_val, predict_z_size, B_START=B_START, B_DELTA=B_DELTA) # 301
    count_pdf_test = count_pdf(Z_test, predict_z_size, B_START=B_START, B_DELTA=B_DELTA) # 301
    count_cdf_val = count_cdf(Z_val, predict_z_size, B_START=B_START, B_DELTA=B_DELTA) # 301
    count_cdf_test = count_cdf(Z_test, predict_z_size, B_START=B_START, B_DELTA=B_DELTA) # 301

    count_pdf_pred_val = avg_pdf(predict_pdf_val)
    count_pdf_pred_test = avg_pdf(predict_pdf_test)
    count_cdf_pred_val = avg_cdf(predict_pdf_val)
    count_cdf_pred_test = avg_cdf(predict_pdf_test)
    weights = np.array([i for i in range(predict_z_size)])
    Z_pred_val = predict_pdf_val.dot(weights)
    Z_pred_test = predict_pdf_test.dot(weights)

    val_mse = metrics.mean_squared_error(Z_val, Z_pred_val) # (376693,) (376693,)
    test_mse = metrics.mean_squared_error(Z_test, Z_pred_test) # (417197,) (417197,)

    val_mae = metrics.mean_absolute_error(Z_val, Z_pred_val) # (376693,) (376693,)
    test_mae = metrics.mean_absolute_error(Z_test, Z_pred_test) # (376693,) (376693,)

    val_kld = tf.keras.metrics.kullback_leibler_divergence(count_pdf_val, count_pdf_pred_val).numpy() # 301 301
    test_kld = tf.keras.metrics.kullback_leibler_divergence(count_pdf_test, count_pdf_pred_test).numpy() # 301 301

    val_re = entropy(count_pdf_val, count_pdf_pred_val) # 301 301
    test_re = entropy(count_pdf_test, count_pdf_pred_test) # 301 301

    val_wd = wasserstein_distance(count_pdf_val, count_pdf_pred_val)# 301 301
    test_wd = wasserstein_distance(count_pdf_test, count_pdf_pred_test)# 301 301


    val_anlp = anlp(Z_val, predict_pdf_val, B_START=B_START, B_DELTA=B_DELTA)
    test_anlp = anlp(Z_test, predict_pdf_test, B_START=B_START, B_DELTA=B_DELTA)

    val_surplus, val_surplus_rate, val_auc, val_number_of_wins, val_bce = cal_test_metrics(Z_val, Z_result_pred_val, ECPM_val, B_START=B_START, B_DELTA=B_DELTA, B_LIMIT=B_LIMIT)
    surplus, surplus_rate, test_auc, number_of_wins, test_bce = cal_test_metrics(Z_test, Z_result_pred_test, ECPM_test, B_START=B_START, B_DELTA=B_DELTA, B_LIMIT=B_LIMIT)

    
    mp_metrics = {
        'val_mse': val_mse,
        'test_mse': test_mse,

        'val_mae': val_mae,
        'test_mae': test_mae,

        'val_kld': val_kld,
        'test_kld': test_kld,

        'val_re': val_re,
        'test_re': test_re,

        'val_wd': val_wd,
        'test_wd': test_wd,

        'val_anlp': val_anlp,
        'test_anlp': test_anlp,

        # added for more detailed analysis 230525
        'surplus': surplus,
        'surplus_rate': surplus_rate,

        'val_auc': val_auc,
        'test_auc': test_auc,

        'number_of_wins': number_of_wins,

        'val_bce': val_bce,
        'test_bce': test_bce,
        # 'total_area': total_area, 
        # 'left_area': left_area,
        # 'right_area': right_area,
        # 'value_z': value_z, 
        # 'value_win_z': value_win_z, 
        # 'value_lose_z': value_lose_z,
        # 'value_b': value_b, 
        # 'value_win_b': value_win_b, 
        # 'value_lose_b': value_lose_b,
    }

    logger.info('Total Val ANLP:{:.6f}. Total Test ANLP:{:.6f}.'.format(val_anlp, test_anlp))
    logger.info('Total Val MSE:{:.6f}. Total TEST MSE:{:.6f}.'.format(val_mse, test_mse))
    logger.info('Total Val MAE:{:.6f}. Total Test MAE:{:.6f}.'.format(val_mae, test_mae))
    logger.info('Total Val KLD:{:.6f}. Total Test KLD:{:.6f}.'.format(val_kld, test_kld))
    logger.info('Total Val RE:{:.6f}. Total Test RE:{:.6f}.'.format(val_re, test_re))
    logger.info('Total Val WD:{:.6f}. Total Test WD:{:.6f}.'.format(val_wd, test_wd))
    logger.info('Total Surplus:{:.10f}. Total Surplus Rate:{:.10f}.'.format(surplus, surplus_rate))
    # logger.info('Total C-index:{:.6f}.'.format(c_index))
    logger.info('Total Val AUC:{:.6f} Total Test AUC:{:.6f}.'.format(val_auc, test_auc))
    logger.info('Total Val BCE:{:.6f} Total Test BCE:{:.6f}.'.format(val_bce, test_bce))
    logger.info('Total number of wins:{:.6f}.'.format(number_of_wins))
    # logger.info('Total Area:{:.6f}. Left Area:{:.6f}. Right Area:{:.6f}.'.format(total_area, left_area, right_area))
    # logger.info('Total value_z:{:.6f}. Total value_win_z:{:.6f}. Total value_lose_z:{:.6f}.'.format(value_z, value_win_z, value_lose_z))
    # logger.info('Total value_b:{:.6f}. Total value_win_b:{:.6f}. Total value_lose_b:{:.6f}.'.format(value_b, value_win_b, value_lose_b))

    
    if save_fig:
        train_pdf_title = 'Epoch {}: Camp {} train pdf'.format(epoch, camp)
        train_cdf_title = 'Epoch {}: Camp {} train cdf'.format(epoch, camp)
        test_pdf_title = 'Epoch {}: Camp {} test pdf'.format(epoch, camp)
        test_cdf_title = 'Epoch {}: Camp {} test cdf'.format(epoch, camp)
        case_study_pdf_title = 'Epoch {}: Camp {} test pdf case study'.format(epoch, camp)
        case_study_cdf_title = 'Epoch {}: Camp {} test cdf case study'.format(epoch, camp)
        plot_mp_result(count_pdf_val, count_pdf_pred_val, save_fig=True, figure_path=figure_path,
                        title=train_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        plot_mp_result(count_cdf_val, count_cdf_pred_val, save_fig=True, figure_path=figure_path,
                        title=train_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        plot_mp_result(count_pdf_test, count_pdf_pred_test, save_fig=True, figure_path=figure_path,
                        title=test_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        plot_mp_result(count_cdf_test, count_cdf_pred_test, save_fig=True, figure_path=figure_path,
                        title=test_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        case_index = np.random.randint(len(Z_test))
        case_z = Z_test[case_index]
        case_pdf = predict_pdf_test[case_index]
        case_cdf = pdf2cdf(case_pdf)
        pickle.dump(case_pdf, open(pdf_path + 'case_study_pdf_{}.pkl'.format(epoch), 'wb'))
        pickle.dump(case_cdf, open(pdf_path + 'case_study_cdf_{}.pkl'.format(epoch), 'wb'))
        plot_mp_case(case_z, case_pdf, save_fig=True, figure_path=figure_path,
                        title=case_study_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        plot_mp_case(case_z, case_cdf, save_fig=True, figure_path=figure_path,
                        title=case_study_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)

    pickle.dump(count_pdf_pred_val, open(pdf_path + 'val_pdf_{}.pkl'.format(epoch), 'wb'))
    pickle.dump(count_pdf_pred_test, open(pdf_path + 'test_pdf_{}.pkl'.format(epoch), 'wb'))

    return mp_metrics

def tf_summary_metrics(metrics, epoch):
    for key in metrics.keys():
        tf.summary.scalar(key, data=metrics[key], step=epoch)


class LogSDKCallback(tf.keras.callbacks.Callback):
    """
    LogSDKCallback class is used to log the information about the loss during the training of the multi-task model.
    """
    def __init__(self, dataset=None, camp='sdk', monitor='surplus_rate', patience=5, val_data=None, test_data=None, save_fig=True, figure_path=None, pdf_path=None, predict_batch_size=None, predict_z_size=None, cdf_flag=False, B_START=0, B_DELTA=1, B_LIMIT=300, cdf_type='ab', **kwargs):
        """
        Intialize LogMTLossCallback class with full-volume data and batch size.
        """
        super(LogSDKCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.predict_z_size = predict_z_size
        # self.predict_type = predict_type
        self.save_fig = save_fig
        self.figure_path = figure_path
        self.pdf_path = pdf_path
        self.camp = camp
        # self.import_model = import_model
        self.result_df = pd.DataFrame()
        # self.B_test = B_test
        # self.B_train = B_train
        self.cdf_flag = cdf_flag

        self.val_data = val_data
        self.test_data = test_data

        self.monitor = monitor
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = 0
        self.best_epoch = 0
        self.dataset = dataset
        self.B_START = B_START
        self.B_DELTA = B_DELTA
        self.B_LIMIT = B_LIMIT
        self.cdf_type = cdf_type


    def on_epoch_end(self, epoch, logs):
        """
        Log the information about the loss during the training of the multi-task model at the end of each epoch.
        Logs include AUC, LOG-LOSS, MSE, MAE, ANLP, KLD, RE, WD.
        :param epoch: epoch: int
        :param logs: tensorflow default logs: dict
        :return: None
        """
        print('\n')
        logger.info('======================Current epoch:{}======================'.format(epoch))
        logger.info('=====================Evaluating metrics=====================')
        model = self.import_model if self.import_model else self.model

        Z_result_pred_val = mt_batch_predict2(self.dataset, model, self.val_data.X, self.val_data.B, self.predict_batch_size, output_dim=self.predict_z_size, cdf_flag=self.cdf_flag, cdf_type=self.cdf_type)
        Z_result_pred_test = mt_batch_predict2(self.dataset, model, self.test_data.X, self.test_data.B, self.predict_batch_size, output_dim=self.predict_z_size, cdf_flag=self.cdf_flag, cdf_type=self.cdf_type)
        mp_metrics = evalute_sdk_mp_metrics(camp=self.camp, output_dim=self.predict_z_size, val_data=self.val_data, test_data=self.test_data,
                                        Z_result_pred_val=Z_result_pred_val, Z_result_pred_test=Z_result_pred_test,
                                        save_fig=self.save_fig, figure_path=self.figure_path, 
                                        pdf_path=self.pdf_path, epoch=epoch, B_START=self.B_START, B_DELTA=self.B_DELTA, B_LIMIT=self.B_LIMIT)
        mt_metrics = dict(**mp_metrics)
        tf_summary_metrics(mt_metrics, epoch)
        mt_metrics = dict(mt_metrics, **logs)
        mt_metrics['epoch'] = epoch
        self.result_df = self.result_df.append([mt_metrics], ignore_index=True)

        current = mt_metrics[self.monitor] # earlystop metric

        if current > self.best:
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if not os.path.exists(self.pdf_path):
                os.makedirs(self.pdf_path)
            model.save(self.pdf_path + 'model_weights.h5')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                # self.model.stop_training = True
                logger.info(f'Epoch {epoch + 1}: early stopping: {self.best}')
                logger.info(f'Best at epoch {self.best_epoch}')
                # return False

        model.Z_result_pred_val = Z_result_pred_val
        model.Z_result_pred_test = Z_result_pred_test
        return Z_result_pred_val, Z_result_pred_test
