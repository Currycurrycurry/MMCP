import sys
# sys.path.append('../')

import time
import pickle
import logging
import numpy as np
from config import base_config, feature_config
from util.encoder import Encoder
from util.data_loader import DataLoader
from util.data_format import EncodeData
from sklearn.model_selection import train_test_split

logger = logging.getLogger('tensorflow')


class FeatureProcessor:
    """
    Processor class is used to process data.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = []

    @staticmethod
    def useragent_process(content):
        """
        Process the useragent feature.
        :param content: useragent: str
        :return: 'operation_browser': str
        """
        content = content.lower()
        operation = "other"
        oses = ["windows", "ios", "mac", "android", "linux"]
        browsers = [
            "chrome",
            "sogou",
            "maxthon",
            "safari",
            "firefox",
            "theworld",
            "opera",
            "ie"]
        for o in oses:
            if o in content:
                operation = o
                break
        browser = "other"
        for b in browsers:
            if b in content:
                browser = b
                break
        return operation + "_" + browser

    @staticmethod
    def slot_price_process(content):
        """
        Transform the continuous slot price into several discrete box.
        :param content: slot price: int
        :return: slot price box: str
        """
        price = int(content)
        if price > 100:
            return "101+"
        elif price > 50:
            return "51-100"
        elif price > 10:
            return "11-50"
        elif price > 0:
            return "1-10"
        else:
            return "0"

    def process(self, data, dataset='ipinyou'):
        """
        Process data.
        :param data: original data: Dataframe
        :return: processed data: Dataframe
        """
        print("Start processing data")
        start_time = time.time()
        data = data.fillna("other")
        if self.dataset == 'ipinyou':
            data.loc[:, ['useragent']] = data['useragent'].apply(lambda x: self.useragent_process(x))
            data.loc[:, ["slotprice"]] = data["slotprice"].apply(lambda x: self.slot_price_process(x))
        end_time = time.time()
        print("Process data done. Time cost:{}".format(end_time - start_time))
        return data


class Processor:
    """
    Processor class is used to process data in advance for models.
    """
    def __init__(self, campaign, dataset='ipinyou', encode_type='label_encode'):
        """
        Initilize Preprocessor class with following parameters.
        :param campaign: campaign name: str
        :param dataset: dataset name: str
        :param encode_type: 'not_encode', 'label_encode', "onehot_encode'
        :param train_features: training features: list
        :param categorical_features: categorical features: list
        """
        self.dataset = dataset
        self.campaign = campaign
        self.encode_type = encode_type
        # train
        self.X_train = None
        self.Y_train = None
        self.Z_train = None
        self.B_train = None
        # test
        self.X_test = None
        self.Y_test = None
        self.Z_test = None
        self.B_test = None
        # val 
        self.X_val = None
        self.B_val = None
        self.Z_val = None

        self.train_features = feature_config.train_features[dataset]
        self.clk_label = base_config.clk_label[dataset]
        self.mp_label = base_config.mp_label[dataset]
        self.bid_label = base_config.bid_label[dataset]
        self.pickle_path = base_config.encode_path + self.encode_type + '/'

        self.X_train_first = None
        self.Y_train_first = None
        self.Z_train_first = None
        self.B_train_first = None

        self.X_train_second = None
        self.Y_train_second = None
        self.Z_train_second = None
        self.B_train_second = None
    
    def data_process_rtb(self, random_bid=False):
        """
        Process rtb data with Processor class.
        :return: None.
        """ 
        data_loader = DataLoader(self.campaign, dataset=self.dataset)
        # data loading
        train_data = data_loader.load_data("rtb")

        self.X_train = train_data[self.train_features]
        self.Z_train = train_data[self.mp_label].values
        self.B_train = train_data[self.bid_label].values if self.bid_label else None
        # truthful bidder

        print('winning rate: ', np.count_nonzero(self.Z_train) / len(self.Z_train))

        train_data = EncodeData(self.X_train, None, self.Z_train, self.B_train)

        return train_data

    def data_process_sdk(self, random_bid=False, stats_only=False):
        """
        Process sdk data with Processor class.
        :return: None.
        """ 
        data_loader = DataLoader(self.campaign, dataset=self.dataset)
        # data loading
        data = data_loader.load_data("sdk")
        train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=0.22, random_state=42)
        self.X_train = train_data[self.train_features]
        self.Z_train = train_data[self.mp_label].values
        self.B_train = train_data[self.bid_label].values if self.bid_label else None
        # truthful bidder
        if random_bid:
            self.B_train = np.random.randint(1, 100000, size=self.B_train.shape)
        else:

            self.B_train = (self.X_train['first_ad_total_ecpm'] * 0.18).astype(int) 
            print('winning rate: ', np.count_nonzero(self.B_train >= self.Z_train) / len(self.Z_train))

        del train_data

        self.X_val = val_data[self.train_features]
        self.Z_val = val_data[self.mp_label].values

        if stats_only:
            self.B_val = (self.X_val['first_ad_total_ecpm'] * 0.18).astype(int) 
        else:
            self.B_val = self.X_val["first_ad_total_ecpm"]
        del val_data

        self.X_test = test_data[self.train_features]
        self.Z_test = test_data[self.mp_label].values

        if stats_only:
            self.B_test = (self.X_test['first_ad_total_ecpm'] * 0.18).astype(int) 
        else:
            self.B_test = self.X_test["first_ad_total_ecpm"]
        del test_data
        train_data = EncodeData(self.X_train, None, self.Z_train, self.B_train)
        val_data = EncodeData(self.X_val, None, self.Z_val, self.B_val)
        test_data = EncodeData(self.X_test, None, self.Z_test, self.B_test)
        return train_data, val_data, test_data

    def data_process(self):
        """
        Process data with Processor class.
        :return: None.
        """
        data_loader = DataLoader(self.campaign, dataset=self.dataset)
        # data loading
        train = data_loader.load_data("train")
        test = data_loader.load_data("test")
        # data preprocessing
        feat_processor = FeatureProcessor(dataset=self.dataset)
        train = feat_processor.process(train)
        test = feat_processor.process(test)
        self.X_train = train[self.train_features]
        self.Y_train = train[self.clk_label].values
        self.Z_train = train[self.mp_label].values
        self.B_train = train[self.bid_label].values if self.bid_label else None
        del train
        self.X_test = test[self.train_features]
        self.Y_test = test[self.clk_label].values
        self.Z_test = test[self.mp_label].values
        self.B_test = test[self.bid_label].values if self.bid_label else None
        del test
        return

    def data_process_mixed(self, first_second_ratio=0.5):
        """
        Process data with Processor class.
        :return: None.
        """
        data_loader = DataLoader(self.campaign, dataset=self.dataset)
        # data loading
        train = data_loader.load_data("train")
        test = data_loader.load_data("test")
        # data preprocessing
        feat_processor = FeatureProcessor(dataset=self.dataset)
        train = feat_processor.process(train)
        test = feat_processor.process(test)

        test_first_url_mask = train['url'].isin(test['url'])


        train_first = train[test_first_url_mask]
        self.X_train_first = train_first[self.train_features]
        self.Y_train_first = train_first[self.clk_label].values
        self.Z_train_first = train_first[self.mp_label].values
        self.B_train_first = train_first[self.bid_label].values if self.bid_label else None
        del train_first

        train_second = train[~test_first_url_mask]
        self.X_train_second = train_second[self.train_features]
        self.Y_train_second = train_second[self.clk_label].values
        self.Z_train_second = train_second[self.mp_label].values
        self.B_train_second = train_second[self.bid_label].values if self.bid_label else None
        del train_second

        self.X_test = test[self.train_features]
        self.Y_test = test[self.clk_label].values
        self.Z_test = test[self.mp_label].values
        self.B_test = test[self.bid_label].values if self.bid_label else None
        del test
        return

    def encode(self, save_result=False):
        """
        Encode data with Encode class
        :param save_result: saving result indicicator parameter
        :return: X_train: optional, X_test: optional
        """
        print("start encoding for campaign {}".format(self.campaign))
        self.data_process()
        # data encoding
        encoder = Encoder()
        self.X_train, self.X_test = encoder.encode(self.X_train, self.X_test, self.train_features, self.encode_type)

        if save_result:
            self.dump_pickle('train')
            self.dump_pickle('test')
        return self.X_train, self.X_test

    def encode_mixed(self, save_result=False):
        """
        Encode data with Encode class
        :param save_result: saving result indicicator parameter
        :return: X_train: optional, X_test: optional
        """
        print("start encoding both fisrt and second data for campaign {}".format(self.campaign))
        self.data_process_mixed()
        # data encoding
        encoder = Encoder()
        self.X_train_first, self.X_test = encoder.encode(self.X_train_first, self.X_test, self.train_features, self.encode_type)
        self.X_train_second, self.X_test = encoder.encode(self.X_train_second, self.X_test, self.train_features, self.encode_type)

        if save_result:
            self.dump_pickle_mixed('train_first')
            self.dump_pickle_mixed('train_second')
            self.dump_pickle_mixed('test_first')
        return self.X_train, self.X_test

    def dump_pickle(self, data_type):
        """
        Dump encoded data to a pickle file.
        :param data_type: 'train' or 'test'
        :return: None
        """
        pickle_name = '{}_{}.pkl'.format(self.campaign, data_type)
        with open(self.pickle_path+pickle_name, 'wb') as f:
            if data_type == 'train':
                pickle_data = EncodeData(self.X_train, self.Y_train, self.Z_train, self.B_train)
            else:
                pickle_data = EncodeData(self.X_test, self.Y_test, self.Z_test, self.B_test)
            pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_pickle_mixed(self, data_type):
        """
        Dump encoded data to a pickle file.
        :param data_type: 'train' or 'test'
        :return: None
        """
        pickle_name = '{}_{}.pkl'.format(self.campaign, data_type)
        with open(self.pickle_path+pickle_name, 'wb') as f:
            if data_type == 'train_first':
                pickle_data = EncodeData(self.X_train_first, self.Y_train_first, self.Z_train_first, self.B_train_first)
            elif data_type == 'train_second':
                pickle_data = EncodeData(self.X_train_second, self.Y_train_second, self.Z_train_second, self.B_train_second)
            elif data_type == 'test_first':
                pickle_data = EncodeData(self.X_test, self.Y_test, self.Z_test, self.B_test)
            pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load_pickle(self, data_type):
        """
        Load encoded data from the pickle file.
        :param data_type: 'train' or 'test'
        :return: pickle_data: optional
        """
        pickle_name = '{}_{}.pkl'.format(self.campaign, data_type)
        with open(self.pickle_path+pickle_name, 'rb') as f:
            pickle_data = pickle.load(f, encoding='bytes')
        return pickle_data

    def load_encode(self, data_type):
        """
        Load data, including encoded feature X, click label Y and market price Z.
        :param data_type: 'train' or 'test': str
        :param camp: campaign name: str
        :return: encoded data: custom Encode class object (including X, Y, Z attributes)
        """
        logger.info("Start loading {} data for campaign {}".format(data_type, self.campaign))
        start_time = time.time()
        encode = self.load_pickle(data_type)
        encode.X = encode.X[self.train_features]
        end_time = time.time()
        logger.info("Loading data done.Time cost:{}".format(end_time - start_time))
        return encode

    def load_encode_mixed(self, data_type):
        """
        Load data, including encoded feature X, click label Y and market price Z.
        :param data_type: 'train' or 'test': str
        :param camp: campaign name: str
        :return: encoded data: custom Encode class object (including X, Y, Z attributes)
        """
        logger.info("Start loading mixed {} data for campaign {}".format(data_type, self.campaign))
        start_time = time.time()
        encode = self.load_pickle(data_type)
        encode.X = encode.X[self.train_features]
        end_time = time.time()
        logger.info("Loading data done.Time cost:{}".format(end_time - start_time))
        return encode


if __name__ == '__main__':
    dataset = 'ipinyou'
    # campaign_list = base_config.campaign_list[dataset]
    # take campaign 1458 as an example
    campaign_list = ['3476']
    for camp in campaign_list:
        processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
        # processor.encode(save_result=True)
        processor.encode_mixed(save_result=True)
