from util.data_format import CensoredBatchData, CensoredBatchFirstSecondData, MixedBatchData
from util.truthful_bidder import TruthfulBidder
from util.processor import Processor, EncodeData
from config import base_config
import logging
import numpy as np
import sys
sys.path.append('../')


np.random.seed(2023)
logger = logging.getLogger('tensorflow')


class CensoredProcessor:
    def __init__(self, processor, bidder, bid_prop):
        self.processor = processor
        self.bidder = bidder
        self.win_data = None
        self.lose_data = None
        self.bid_prop = bid_prop

    def load_encode(self, data_type='train', load_bid=True, random_bid=False, mixed=False):
        data = self.processor.load_encode(data_type)

        if random_bid:
            bids = np.random.randint(1, 301, size=data.B.shape) # TODO add predict_z to control the random range
        else:
            if load_bid:
                bids = self.bidder.load_bids(type=data_type)
            else:
                bids = data.B * self.bid_prop

        if mixed:
            return EncodeData(X=data.X, Y=data.Y,
                              Z=data.Z, B=bids)
        else:
            win_index = data.Z < bids
            lose_index = data.Z >= bids
            win_data = EncodeData(X=data.X[win_index], Y=data.Y[win_index],
                                  Z=data.Z[win_index], B=bids[win_index])
            lose_data = EncodeData(X=data.X[lose_index], Y=data.Y[lose_index],
                                   Z=data.Z[lose_index], B=bids[lose_index])
            return win_data, lose_data

    def generate_batch_index(self, size, batch_size=10240):
        start_indexs = [
            i * batch_size for i in range(int(np.ceil(size / batch_size)))]
        end_indexs = [
            i * batch_size for i in range(1, int(np.ceil(size / batch_size)))]
        end_indexs.append(size)
        return start_indexs, end_indexs

    def load_rtb_dataset_pipeline(self, batch_size=10240):
        '''
        load the dataset from scratch which can be directly used.
        return: censored_train, censored_val, censored_test
        '''
        train_data = self.processor.data_process_rtb()
        train_size = len(train_data.Z)
        start_indexes, end_indexes = self.generate_batch_index(
            train_size, batch_size)
        censored_train = self.generate_mixed_batches(
            train_data, start_indexes, end_indexes, first_flag=False)
        np.random.shuffle(censored_train)
        for i in range(len(censored_train)):
            censored_train[i].batch_id = i
        return censored_train, train_data

    def load_sdk_dataset_pipeline(self, batch_size=10240, stats_only=False):
        '''
        load the dataset from scratch which can be directly used.
        return: censored_train, censored_val, censored_test

        '''
        train_data, val_data, test_data = self.processor.data_process_sdk(
            stats_only=stats_only)
        train_size = len(train_data.Z)
        start_indexes, end_indexes = self.generate_batch_index(
            train_size, batch_size)
        censored_train = self.generate_mixed_batches(
            train_data, start_indexes, end_indexes, first_flag=True)
        np.random.shuffle(censored_train)
        for i in range(len(censored_train)):
            censored_train[i].batch_id = i
        return censored_train, train_data, val_data, test_data

    def load_dataset(self, batch_size=10240, data_type='train'):
        win_data, lose_data = self.load_encode(data_type=data_type)
        win_size, lose_size = len(win_data.Z), len(lose_data.Z)
        win_start_indexs, win_end_indexs = self.generate_batch_index(
            win_size, batch_size)
        lose_start_indexs, lose_end_indexs = self.generate_batch_index(
            lose_size, batch_size)
        censored_win_dataset = self.generate_censored_batches(
            win_data, win_start_indexs, win_end_indexs, True)
        censored_lose_dataset = self.generate_censored_batches(
            lose_data, lose_start_indexs, lose_end_indexs, False)
        censored_full_dataset = censored_win_dataset + censored_lose_dataset
        np.random.shuffle(censored_full_dataset)
        for i in range(len(censored_full_dataset)):
            censored_full_dataset[i].batch_id = i
        return censored_full_dataset

    def load_mixed_dataset(self, batch_size=10240, data_type='train'):
        mixed_data = self.load_encode(data_type=data_type, mixed=True)
        mixed_size = len(mixed_data.Z)
        mixed_start_indexs, mixed_end_indexs = self.generate_batch_index(
            mixed_size, batch_size)
        censored_mixed_dataset = self.generate_mixed_batches(
            mixed_data, mixed_start_indexs, mixed_end_indexs)
        np.random.shuffle(censored_mixed_dataset)
        for i in range(len(censored_mixed_dataset)):
            censored_mixed_dataset[i].batch_id = i
        return censored_mixed_dataset

    def load_encode_mixed(self, data_type='train', load_bid=False, random_bid=False, mixed_win_lose=False):
        # both first and second data
        first_data = self.processor.load_encode_mixed(data_type + '_first')
        second_data = self.processor.load_encode_mixed(data_type + '_second')

        if random_bid:
            first_bids = np.random.randint(1, 301, size=first_data.B.shape)
            second_bids = np.random.randint(1, 301, size=second_data.B.shape)
        else:
            if load_bid:
                first_bids = self.bidder.load_bids('first')
                second_bids = self.bidder.load_bids('second')
            else:
                first_bids = first_data.B * self.bid_prop
                second_bids = second_data.B * self.bid_prop

            # first_bids = first_data.B
            # second_bids = second_data.B

        if mixed_win_lose:
            first_data = EncodeData(X=first_data.X, Y=first_data.Y,
                                    Z=first_data.Z, B=first_bids)
            # newly add 当在mixed_win_lose模式下时，按照9:1分割训练集和验证集，来作为earlystop的依据
            length = len(first_data.X)
            indexes = np.arange(0, length)
            val_indexes = np.random.choice(indexes, int(length * 0.1), replace=False)
            train_indexes = np.setdiff1d(indexes, val_indexes)

            first_train_data = EncodeData(X=first_data.X.iloc[train_indexes], Y=first_data.Y[train_indexes],
                                    Z=first_data.Z[train_indexes], B=first_bids[train_indexes])
            first_val_data = EncodeData(X=first_data.X.iloc[val_indexes], Y=first_data.Y[val_indexes],
                                    Z=first_data.Z[val_indexes], B=first_bids[val_indexes])

            second_data = EncodeData(X=second_data.X, Y=second_data.Y,
                                     Z=second_data.Z, B=second_bids)
            
            return first_train_data, first_val_data, second_data
        else:
            first_win_index = first_data.Z < first_bids
            first_lose_index = first_data.Z >= first_bids
            # print('first winning rate: ', np.count_nonzero(first_win_index) / len(first_data.Z))

            second_win_index = second_data.Z < second_bids
            second_lose_index = second_data.Z >= second_bids
            # print('second winning rate: ', np.count_nonzero(second_win_index) / len(second_data.Z))

            first_win_data = EncodeData(X=first_data.X[first_win_index], Y=first_data.Y[first_win_index],
                                        Z=first_data.Z[first_win_index], B=first_bids[first_win_index])
            first_lose_data = EncodeData(X=first_data.X[first_lose_index], Y=first_data.Y[first_lose_index],
                                         Z=first_data.Z[first_lose_index], B=first_bids[first_lose_index])
            second_win_data = EncodeData(X=second_data.X[second_win_index], Y=second_data.Y[second_win_index],
                                         Z=second_data.Z[second_win_index], B=second_bids[second_win_index])
            second_lose_data = EncodeData(X=second_data.X[second_lose_index], Y=second_data.Y[second_lose_index],
                                          Z=second_data.Z[second_lose_index], B=second_bids[second_lose_index])
            return second_win_data, first_win_data, first_lose_data, second_lose_data

    def load_dataset_mixed(self, batch_size=10240, data_type='train', mixed_batch=False, only_first=False):
        if mixed_batch:
            first_mixed_data, _, second_mixed_data = self.load_encode_mixed(
                data_type=data_type, mixed_win_lose=True)
            # first
            first_mixed_size = len(first_mixed_data.Z)
            first_mixed_start_indexs, first_mixed_end_indexs = self.generate_batch_index(
                first_mixed_size, batch_size)
            first_censored_mixed_dataset = self.generate_mixed_batches(
                first_mixed_data, first_mixed_start_indexs, first_mixed_end_indexs, first_flag=True)
            # second
            second_mixed_size = len(second_mixed_data.Z)
            second_mixed_start_indexs, second_mixed_end_indexs = self.generate_batch_index(
                second_mixed_size, batch_size)
            second_censored_mixed_dataset = self.generate_mixed_batches(
                second_mixed_data, second_mixed_start_indexs, second_mixed_end_indexs, first_flag=False)
            if only_first:
                censored_full_dataset = first_censored_mixed_dataset
            else:
                censored_full_dataset = first_censored_mixed_dataset + second_censored_mixed_dataset

        else:
            second_win_data, first_win_data, first_lose_data, second_lose_data = self.load_encode_mixed(
                data_type=data_type)
            first_win_size, first_lose_size, second_win_size, second_lose_size = len(
                first_win_data.Z), len(first_lose_data.Z), len(second_win_data.Z), len(second_lose_data.Z)

            # mixed first and second
            first_win_start_indexs, first_win_end_indexs = self.generate_batch_index(
                first_win_size, batch_size)
            first_lose_start_indexs, first_lose_end_indexs = self.generate_batch_index(
                first_lose_size, batch_size)
            second_win_start_indexs, second_win_end_indexs = self.generate_batch_index(
                second_win_size, batch_size)
            second_lose_start_indexs, second_lose_end_indexs = self.generate_batch_index(
                second_lose_size, batch_size)

            second_win_dataset = self.generate_censored_first_second_batches(
                second_win_data, second_win_start_indexs, second_win_end_indexs, True, False)
            second_lose_dataset = self.generate_censored_first_second_batches(
                second_lose_data, second_lose_start_indexs, second_lose_end_indexs, False, False)
            first_win_dataset = self.generate_censored_first_second_batches(
                first_win_data, first_win_start_indexs, first_win_end_indexs, False, True)
            first_lose_dataset = self.generate_censored_first_second_batches(
                first_lose_data, first_lose_start_indexs, first_lose_end_indexs, False, True)
            if only_first:
                censored_full_dataset = first_win_dataset + first_lose_dataset
            else:
                censored_full_dataset = second_win_dataset + \
                    second_lose_dataset + first_win_dataset + first_lose_dataset
        np.random.shuffle(censored_full_dataset)
        for i in range(len(censored_full_dataset)):
            censored_full_dataset[i].batch_id = i
        return censored_full_dataset

    def generate_censored_batches(self, data, start_indexs, end_indexs, win_flag):
        censored_batch_dataset = []
        for i in range(len(start_indexs)):
            start = start_indexs[i]
            end = end_indexs[i]
            batch_data = self.data_slice(data, start, end)
            censored_batch_data = CensoredBatchData(
                batch_data, batch_id=-1, win_flag=win_flag)
            censored_batch_dataset.append(censored_batch_data)
        return censored_batch_dataset

    def generate_mixed_batches(self, mixed_data, mixed_start_indexs, mixed_end_indexs, first_flag=True):
        censored_batch_dataset = []
        for i in range(len(mixed_start_indexs)):
            start = mixed_start_indexs[i]
            end = mixed_end_indexs[i]
            batch_data = self.data_slice(mixed_data, start, end)
            censored_batch_data = MixedBatchData(
                batch_data, batch_id=-1, first_flag=first_flag)
            censored_batch_dataset.append(censored_batch_data)
        return censored_batch_dataset

    def generate_censored_first_second_batches(self, data, start_indexs, end_indexs, win_flag, first_flag):
        censored_batch_dataset = []
        for i in range(len(start_indexs)):
            start = start_indexs[i]
            end = end_indexs[i]
            batch_data = self.data_slice(data, start, end)
            censored_batch_data = CensoredBatchFirstSecondData(
                batch_data, batch_id=-1, win_flag=win_flag, first_flag=first_flag)
            censored_batch_dataset.append(censored_batch_data)
        return censored_batch_dataset

    @staticmethod
    def data_slice(data, start_index, end_index):
        return EncodeData(data.X[start_index:end_index],
                          None,
                          data.Z[start_index:end_index],
                          data.B[start_index:end_index])


if __name__ == '__main__':
    dataset = 'ipinyou'
    # campaign_list = base_config.campaign_list[dataset]
    # take campaign 1458 as an example
    campaign_list = ['1458']
    for camp in campaign_list:
        # load data
        processor = Processor(campaign=camp, dataset=dataset,
                              encode_type='label_encode')
        clk_value = base_config.camp_cpc_dict[dataset][camp]
        truthful_bidder = TruthfulBidder(dataset=dataset, campaign=camp)
        censored_processor = CensoredProcessor(processor, truthful_bidder)
        censored_dataset = censored_processor.load_dataset(batch_size=10240)
        win_train, lose_train = censored_processor.load_encode('train')
