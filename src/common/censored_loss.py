import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K
from scipy.stats import entropy, wasserstein_distance
from config.base_config import B_START, B_LIMIT, B_DELTA

def wd_loss(y_true, y_pred):
    _, max_length = y_pred.shape.as_list()
    split_pos = max_length // 2
    first_pdf = tf.slice(y_pred, [0, 0], [-1, split_pos])
    second_pdf = tf.slice(y_pred, [0, split_pos], [-1, max_length - split_pos])
    return K.mean(first_pdf * second_pdf)


def cross_entropy_loss_second(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)        # (None_all, price_step)
    y_true = math_ops.cast(y_true, y_pred.dtype)  # (None_all, 3)
    price_step = tf.cast(tf.shape(y_pred)[-1], tf.int32)

    # split y_true
    y_true_label_1d = K.flatten(tf.slice(y_true, [0,0], [-1,1]))  # (None_all,)
    # caculate the bidding price bucket index
    y_true_b = tf.slice(y_true, [0,1], [-1,1])  # (None_all, 1)
    y_true_b = tf.clip_by_value(y_true_b, B_START, B_LIMIT)
    y_true_b_idx_2d = tf.cast(tf.floor((y_true_b - B_START) / B_DELTA), dtype='int32')  # (None_all, 1)
    y_true_b_idx_1d = K.flatten(y_true_b_idx_2d)  # (None_all,)
    # caculate the winning price bucket index
    y_true_z = tf.slice(y_true, [0,2], [-1,1])  # (None_all, 1)
    y_true_z = tf.clip_by_value(y_true_z, B_START, B_LIMIT)
    y_true_z_idx_2d = tf.cast(tf.floor((y_true_z - B_START) / B_DELTA), dtype='int32')  # (None_all, 1)
    y_true_z_idx_1d = K.flatten(y_true_z_idx_2d)  # (None_all,)

    # Calculate masks
    ## on All bids
    mask_win = y_true_label_1d  # (None,)
    mask_lose = 1 - mask_win  # (None,)

    mask_z_cdf = tf.sequence_mask(
                    y_true_z_idx_1d + 1, 
                    price_step)  # (None, price_step)
    mask_z_pdf = tf.math.logical_xor(
                    mask_z_cdf, 
                    tf.sequence_mask(
                        y_true_z_idx_1d,
                        price_step))  # (None, price_step)

    mask_b_cdf = tf.sequence_mask(
                    y_true_b_idx_1d + 1, 
                    price_step)  # (None, price_step)
    mask_b_pdf = tf.math.logical_xor(
                    mask_b_cdf, 
                    tf.sequence_mask(
                        y_true_b_idx_1d, 
                        price_step))  # (None, price_step)
    ## on Winning bids
    mask_win_z_cdf = tf.boolean_mask(mask_z_cdf, mask_win)  # (None_win, price_step)
    mask_win_z_pdf = tf.boolean_mask(mask_z_pdf, mask_win)  # (None_win, price_step)
    mask_win_b_cdf = tf.boolean_mask(mask_b_cdf, mask_win)  # (None_win, price_step)
    mask_win_b_pdf = tf.boolean_mask(mask_b_pdf, mask_win)  # (None_win, price_step)
    ## on Losing bids
    mask_lose_b_cdf = tf.boolean_mask(mask_z_cdf, mask_lose)  # (None_lose, price_step)
    mask_lose_b_pdf = tf.boolean_mask(mask_z_pdf, mask_lose)  # (None_lose, price_step)

    # Price Distribution
    y_pred_win = tf.boolean_mask(y_pred, mask_win)  # (None_win, price_step)
    y_pred_lose = tf.boolean_mask(y_pred, mask_lose)  # (None_lose, price_step)

    # Loss
    zeros = tf.zeros(tf.shape(y_pred), tf.float32)  # (None, price_step)
    zeros_win = tf.zeros(tf.shape(y_pred_win), tf.float32)  # (None_win, price_step)
    zeros_lose = tf.zeros(tf.shape(y_pred_lose), tf.float32)  # (None_lose, price_step)
    ones = tf.ones(tf.shape(y_pred), tf.float32)  # (None, price_step)
    ones_win = tf.ones(tf.shape(y_pred_win), tf.float32)  # (None_win, price_step)
    ones_lose = tf.ones(tf.shape(y_pred_lose), tf.float32)  # (None_lose, price_step)

    # loss_1
    loss_1 = - K.sum(
                tf.math.log(tf.clip_by_value(
                    tf.boolean_mask(
                        y_pred_win,
                        mask_win_z_pdf),
                    K.epsilon(),
                    1.)))

    # loss_2_win
    left_neighbourhood_offset = y_true_b_idx_1d - y_true_z_idx_1d
    left_neighbourhood_idx = tf.math.maximum(y_true_z_idx_1d - left_neighbourhood_offset, 0)
    mask_z_neighbourhood_cdf = tf.math.logical_xor(
                                    mask_b_cdf, 
                                    tf.sequence_mask(
                                        left_neighbourhood_idx,
                                        price_step))
    mask_win_z_neighbourhood_cdf = tf.boolean_mask(mask_z_neighbourhood_cdf, mask_win)
    loss_2_win = - K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_win_z_neighbourhood_cdf, 
                                y_pred_win, 
                                zeros_win),
                            axis=1),
                        K.epsilon(),
                        1.)))
    # loss_2_lose
    right_neighbourhood_offset = 40
    right_neighbourhood_idx = tf.math.minimum(y_true_b_idx_1d + right_neighbourhood_offset, price_step - 1)
    mask_b_neighbourhood_cdf = tf.math.logical_xor(
                                    tf.math.logical_not(mask_b_cdf), 
                                    tf.math.logical_not(
                                        tf.sequence_mask(right_neighbourhood_idx, price_step)))
    mask_lose_b_neighbourhood_cdf = tf.boolean_mask(mask_b_neighbourhood_cdf, mask_lose)
    loss_2_lose = - K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_lose_b_neighbourhood_cdf, 
                                y_pred_lose, 
                                zeros_lose),
                            axis=1),
                        K.epsilon(),
                        1.)))
    # loss_2
    beta = 0.2
    loss_2 = beta * loss_2_win + (1 - beta) * loss_2_lose

    # total loss
    alpha = 0.5
    return alpha * loss_1 + (1 - alpha) * loss_2


def cross_entropy_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)        # (None_all, price_step)
    y_true = math_ops.cast(y_true, y_pred.dtype)  # (None_all, 3)

    price_step = tf.cast(tf.shape(y_pred)[-1], tf.int32)
    y_true_label_1d = K.flatten(tf.slice(y_true, [0,0], [-1,1]))  # (None_all,)

    y_true_b = tf.slice(y_true, [0,1], [-1,1])  # (None_all, 1) # Tensor("cross_entropy_loss/Cast:0", shape=(10240, 1), dtype=float32)

    y_true_b = tf.clip_by_value(y_true_b, B_START, B_LIMIT)
    y_true_b_idx_2d = tf.cast(tf.floor((y_true_b - B_START) / B_DELTA), dtype='int32')  # (None_all, 1)
    y_true_b_idx_1d = K.flatten(y_true_b_idx_2d)  # (None_all,)
    # caculate the winning price bucket index
    y_true_z = tf.slice(y_true, [0,2], [-1,1])  # (None_all, 1)
    y_true_z = tf.clip_by_value(y_true_z, B_START, B_LIMIT)
    y_true_z_idx_2d = tf.cast(tf.floor((y_true_z - B_START) / B_DELTA), dtype='int32')  # (None_all, 1)
    y_true_z_idx_1d = K.flatten(y_true_z_idx_2d)  # (None_all,)

    # Calculate masks
    ## on All bids
    mask_win = y_true_label_1d  # (None,)
    mask_lose = 1 - mask_win  # (None,)
    mask_b_cdf = tf.sequence_mask(
                    y_true_b_idx_1d + 1, 
                    price_step)  # (None, price_step)


    # Price Distribution
    y_pred_win = tf.boolean_mask(y_pred, mask_win)  # (None_win, price_step)
    y_pred_lose = tf.boolean_mask(y_pred, mask_lose)  # (None_lose, price_step)

    # Loss
    zeros = tf.zeros(tf.shape(y_pred), tf.float32)  # (None, price_step)
    zeros_win = tf.zeros(tf.shape(y_pred_win), tf.float32)  # (None_win, price_step)
    zeros_lose = tf.zeros(tf.shape(y_pred_lose), tf.float32)  # (None_lose, price_step)

    # loss_win 没有wp的竞胜样本交叉熵loss计算
    left_neighbourhood_offset = B_LIMIT
    left_neighbourhood_idx = tf.math.maximum(y_true_b_idx_1d - left_neighbourhood_offset, 0)
    mask_b_left_neighbourhood_cdf = tf.math.logical_xor(
                                            mask_b_cdf, 
                                            tf.sequence_mask(left_neighbourhood_idx, price_step))

    mask_win_b_neighbourhood_cdf = tf.boolean_mask(mask_b_left_neighbourhood_cdf, mask_win)
    loss_win = -K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_win_b_neighbourhood_cdf,
                                y_pred_win,
                                zeros_win),
                            axis=1),
                        K.epsilon(),
                    1.)))

    # loss_lose 没有wp的竞败样本交叉熵loss计算
    right_neighbourhood_offset = B_LIMIT
    right_neighbourhood_idx = tf.math.minimum(y_true_b_idx_1d + right_neighbourhood_offset, price_step - 1)
    mask_b_right_neighbourhood_cdf = tf.math.logical_xor(
                                    tf.math.logical_not(mask_b_cdf), 
                                    tf.math.logical_not(
                                        tf.sequence_mask(right_neighbourhood_idx, price_step)))
    mask_lose_b_neighbourhood_cdf = tf.boolean_mask(mask_b_right_neighbourhood_cdf, mask_lose)
    loss_lose = - K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_lose_b_neighbourhood_cdf, 
                                y_pred_lose, 
                                zeros_lose),
                            axis=1),
                        K.epsilon(),
                        1.)))

    # 交叉熵loss
    alpha = 0.5
    return alpha * loss_win + (1 - alpha) * loss_lose



def cross_entropy_loss_delta(y_true, y_pred):
    first_pdf = y_pred[0]
    second_pdf = y_pred[1]
    y_pred = y_pred[0]
    left_point = tf.keras.backend.argmax(first_pdf, axis=1) # (None, 1)
    right_point = tf.keras.backend.argmax(second_pdf, axis=1) # (None, 1)
    min_values = tf.keras.backend.minimum(left_point, right_point)
    max_values = tf.keras.backend.minimum(left_point, right_point)
    lefts = tf.keras.backend.switch(tf.keras.backend.less(left_point, right_point), min_values, max_values)
    rights = tf.keras.backend.switch(tf.keras.backend.less(left_point, right_point), max_values, min_values)

    # 将y的数据类型转换为int32
    lefts = K.cast(lefts, dtype='int32')
    rights = K.cast(rights, dtype='int32')
    price_step = tf.cast(tf.shape(y_pred)[-1], tf.int32)

    y_true_label_1d = K.flatten(tf.slice(y_true, [0,0], [-1,1]))  # (None_all,)

    y_true_b = tf.slice(y_true, [0,1], [-1,1])  # (None_all, 1) # Tensor("cross_entropy_loss/Cast:0", shape=(10240, 1), dtype=float32)
    # print(y_true_b)
    y_true_b = tf.clip_by_value(y_true_b, B_START, B_LIMIT)
    y_true_b_idx_2d = tf.cast(tf.floor((y_true_b - B_START) / B_DELTA), dtype='int32')  # (None_all, 1)
    y_true_b_idx_1d = K.flatten(y_true_b_idx_2d)  # (None_all,)

    # Calculate masks
    ## on All bids
    mask_win = y_true_label_1d  # (None,)
    mask_lose = 1 - mask_win  # (None,)
    mask_b_cdf = tf.sequence_mask(
                    y_true_b_idx_1d + 1, 
                    price_step)  # (None, price_step)

    # Price Distribution
    y_pred_win = tf.boolean_mask(y_pred, mask_win)  # (None_win, price_step)
    y_pred_lose = tf.boolean_mask(y_pred, mask_lose)  # (None_lose, price_step)

    # Loss
    # zeros = tf.zeros(tf.shape(y_pred), tf.float32)  # (None, price_step)
    zeros_win = tf.zeros(tf.shape(y_pred_win), tf.float32)  # (None_win, price_step)
    zeros_lose = tf.zeros(tf.shape(y_pred_lose), tf.float32)  # (None_lose, price_step)

    # loss_win 没有wp的竞胜样本交叉熵loss计算
    left_neighbourhood_offset = lefts
    left_neighbourhood_idx = tf.math.maximum(y_true_b_idx_1d - left_neighbourhood_offset, 0)
    mask_b_left_neighbourhood_cdf = tf.math.logical_xor(
                                            mask_b_cdf, 
                                            tf.sequence_mask(left_neighbourhood_idx, price_step))

    mask_win_b_neighbourhood_cdf = tf.boolean_mask(mask_b_left_neighbourhood_cdf, mask_win)
    loss_win = -K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_win_b_neighbourhood_cdf,
                                y_pred_win,
                                zeros_win),
                            axis=1),
                        K.epsilon(),
                    1.)))

    # loss_lose 没有wp的竞败样本交叉熵loss计算
    right_neighbourhood_offset = rights
    right_neighbourhood_idx = tf.math.minimum(y_true_b_idx_1d + right_neighbourhood_offset, price_step - 1)
    mask_b_right_neighbourhood_cdf = tf.math.logical_xor(
                                    tf.math.logical_not(mask_b_cdf), 
                                    tf.math.logical_not(
                                        tf.sequence_mask(right_neighbourhood_idx, price_step)))
    mask_lose_b_neighbourhood_cdf = tf.boolean_mask(mask_b_right_neighbourhood_cdf, mask_lose)
    loss_lose = - K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_lose_b_neighbourhood_cdf, 
                                y_pred_lose, 
                                zeros_lose),
                            axis=1),
                        K.epsilon(),
                        1.)))

    # 交叉熵loss
    alpha = 0.5
    return alpha * loss_win + (1 - alpha) * loss_lose


def kld(p1, p2):
    exp_p1 = K.exp(p1)
    return K.sum(exp_p1*(p1-p2)) # , axis=-1

def jsd(p1, p2):
    p3 = (p1 + p2) / 2 
    return 0.5 * kld(p1, p3) + 0.5 * kld(p2, p3)

def ce(p1, p2):
    return tf.reduce_sum(-p1 * tf.math.log(p2))


# @tf.function
def get_naive_lr_points(first_pdf, second_pdf, max_length):
    # for wd
    min_wd = K.mean(first_pdf * second_pdf) 

    # for ce
    min_kld = ce(first_pdf, second_pdf)

    max_step = int(max_length/10)
    step = 0
    target_step = -5
    while step < max_length: 
        step += 1
        pdf = tf.roll(second_pdf, shift=step, axis=1)
        new_kld = ce(first_pdf, pdf)
        min_kld = tf.keras.backend.switch(tf.keras.backend.less(new_kld, min_kld), new_kld, min_kld)
        target_step = tf.keras.backend.switch(tf.keras.backend.less(new_kld, min_kld), step, target_step)

    second_pdf = tf.roll(second_pdf, shift=-target_step, axis=1)
    inter_pdf =  tf.keras.backend.minimum(first_pdf, second_pdf)
    threshold = B_DELTA / (B_LIMIT - B_START) * 0.001

    greater_mask = tf.greater(inter_pdf, threshold)

    first_true_indices = tf.argmax(greater_mask, axis=1)

    last_true_indices = max_length - tf.argmax(tf.reverse(greater_mask, axis=[1]), axis=1) - 1

    left_point = first_true_indices
    right_point = last_true_indices

    min_values = tf.keras.backend.minimum(left_point, right_point)
    max_values = tf.keras.backend.maximum(left_point, right_point)

    lefts = tf.keras.backend.switch(tf.keras.backend.less(left_point, right_point), min_values, max_values)
    rights = tf.keras.backend.switch(tf.keras.backend.less(left_point, right_point), max_values, min_values)

    lefts = K.cast(lefts, dtype='int32')
    rights = K.cast(rights, dtype='int32')

    return lefts, rights



def cross_entropy_loss_mmcp(y_true, y_pred):
    # 获取x的形状
    batch_size, max_length = y_pred.shape.as_list()
    # if batch_size == None:
    #     batch_size = 20480

    # 计算切片位置
    split_pos = max_length // 2

    # 切片获取前半部分张量
    first_pdf = tf.slice(y_pred, [0, 0], [-1, split_pos])
    # 切片获取后半部分张量
    second_pdf = tf.slice(y_pred, [0, split_pos], [-1, max_length - split_pos])

    y_pred = first_pdf


    # 将y的数据类型转换为int32
    y_pred = ops.convert_to_tensor(y_pred)        # (None_all, price_step)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    price_step = tf.cast(tf.shape(y_pred)[-1], tf.int32)

    # split y_true
    y_true_label_1d = K.flatten(tf.slice(y_true, [0,0], [-1,1]))  # (None_all,)
    # caculate the bidding price bucket index

    y_true_b = tf.slice(y_true, [0,1], [-1,1])  # (None_all, 1) # Tensor("cross_entropy_loss/Cast:0", shape=(10240, 1), dtype=float32)

    y_true_b = tf.clip_by_value(y_true_b, B_START, B_LIMIT)
    y_true_b_idx_2d = tf.cast(tf.floor((y_true_b - B_START) / B_DELTA), dtype='int32')  # (None_all, 1)
    y_true_b_idx_1d = K.flatten(y_true_b_idx_2d)  # (None_all,)

    # Calculate masks
    ## on All bids
    mask_win = y_true_label_1d  # (None,)
    mask_lose = 1 - mask_win  # (None,)
    mask_b_cdf = tf.sequence_mask(
                    y_true_b_idx_1d + 1, 
                    price_step)  # (None, price_step)

    # Price Distribution
    y_pred_win = tf.boolean_mask(y_pred, mask_win)  # (None_win, price_step)
    y_pred_lose = tf.boolean_mask(y_pred, mask_lose)  # (None_lose, price_step)

    # Loss
    zeros_win = tf.zeros(tf.shape(y_pred_win), tf.float32)  # (None_win, price_step)
    zeros_lose = tf.zeros(tf.shape(y_pred_lose), tf.float32)  # (None_lose, price_step)

    lefts, rights = get_naive_lr_points(first_pdf, second_pdf, split_pos)

    # loss_win 没有wp的竞胜样本交叉熵loss计算
    left_neighbourhood_offset = tf.abs(lefts - y_true_b_idx_1d)
    left_neighbourhood_idx = tf.math.maximum(y_true_b_idx_1d - left_neighbourhood_offset, 0)
    mask_b_left_neighbourhood_cdf = tf.math.logical_xor(
                                            mask_b_cdf, 
                                            tf.sequence_mask(left_neighbourhood_idx, price_step))

    mask_win_b_neighbourhood_cdf = tf.boolean_mask(mask_b_left_neighbourhood_cdf, mask_win)
    loss_win = -K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_win_b_neighbourhood_cdf,
                                y_pred_win,
                                zeros_win),
                            axis=1),
                        K.epsilon(),
                    1.)))
    

    # loss_lose 没有wp的竞败样本交叉熵loss计算
    right_neighbourhood_offset = tf.abs(rights - y_true_b_idx_1d)
    right_neighbourhood_idx = tf.math.minimum(y_true_b_idx_1d + right_neighbourhood_offset, price_step - 1)
    mask_b_right_neighbourhood_cdf = tf.math.logical_xor(
                                    tf.math.logical_not(mask_b_cdf), 
                                    tf.math.logical_not(
                                        tf.sequence_mask(right_neighbourhood_idx, price_step)))
    mask_lose_b_neighbourhood_cdf = tf.boolean_mask(mask_b_right_neighbourhood_cdf, mask_lose)
    loss_lose = - K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_lose_b_neighbourhood_cdf, 
                                y_pred_lose, 
                                zeros_lose),
                            axis=1),
                        K.epsilon(),
                        1.)))

    # 交叉熵loss
    alpha = 0.5
    return alpha * loss_win + (1 - alpha) * loss_lose + K.mean(first_pdf * second_pdf)

def get_stats(p, b):
    mean = np.sum(p * b)
    var = np.sum((b - mean)**2 * p)
    std = np.sqrt(var)
    return mean, var, std

def dn(p1, p2, b):
    mean1, var1, std1 = get_stats(p1, b)
    mean2, var2, std2 = get_stats(p2, b)

