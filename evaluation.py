import numpy as np   
import tensorflow as tf 

def evaluate(model, inputtensor, test_idx, all_ite, ts, yfs):
    test_t = tf.gather(ts, test_idx)
    true_ite = tf.gather(all_ite, test_idx)
    test_yf = tf.gather(yfs, test_idx)
    pre_t, pre_c = model(tf.cast(inputtensor, tf.float32), test_idx)
    pre_test_yf = np.array(test_t) * np.array(pre_t) + np.array((1 - test_t)) * np.array(pre_c)

    pred_ite = np.array(pre_t) - np.array(pre_c)
    ate = np.array(tf.reduce_mean(pred_ite))
    true_ate = np.array(tf.reduce_mean(true_ite))

    pehe = np.mean((pred_ite - true_ite) ** 2)
    msey = np.mean((pre_test_yf - test_yf) ** 2)
    err_ate = np.abs(ATE - true_ate)
    return msey, pehe, err_ate


