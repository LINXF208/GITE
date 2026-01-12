import math
import os
import time

import numpy as np 
import tensorflow as tf
import pandas as pd
import evaluation
from tensorflow import keras
from scipy.sparse import csr_matrix


def wasserstein(X, out_1, out_0, t, yf, p, gamma_hy,lam=10, its=10, sq=True, backpropT=False):
    it = tf.where(t > 0)[:,0]
    ic = tf.where(t < 1)[:,0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.cast(tf.shape(Xc)[0], tf.float32)
    nt = tf.cast(tf.shape(Xt)[0], tf.float32)

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = tf.sqrt(tf.clip_by_value(pdist2sq(Xt, Xc), SQRT_CONST, np.inf))    

    if gamma_hy > 0:
        pred_0_cf = tf.gather(out_1, tf.where(t == 0)[:, 0])  # Predicted outcomes for control group with t=1
        pred_1_cf = tf.gather(out_0, tf.where(t == 1)[:, 0])  # Predicted outcomes for treated group with t=0
        yf_1 = tf.gather(yf, tf.where(t == 1)[:, 0]) ## factual outcomes with t=1
        yf_0 = tf.gather(yf, tf.where(t == 0)[:, 0]) ## factual outcomes with t=0

        dist_10 = pdist2sq(pred_0_cf, yf_1) ## 11
        dist_01 = pdist2sq(yf_0, pred_1_cf) ## 00
        dist_10 = tf.transpose(dist_10)
        dist_01 = tf.transpose(dist_01)
        M += gamma_hy * (dist_10 + dist_01)

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam / (M_mean + 1e-10))

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:, 0:1])),tf.zeros((1, 1))],0)
    Mt = tf.concat([M,row],0)
    Mt = tf.concat([Mt,col],1)

    ''' Compute marginal vectors '''
    a = tf.concat([p * tf.ones(tf.shape(tf.where(t > 0)[:, 0:1]))/nt, (1 - p) * tf.ones((1, 1))], 0)
    b = tf.concat([(1 - p) * tf.ones(tf.shape(tf.where(t < 1)[:, 0:1]))/nc, p * tf.ones((1, 1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-10
    U = K * Mt
    ainvK = K / (a + 1e-10)
    u = a
    for i in range(0,its):
        u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K))) + 1e-10) + 1e-10)
    v = b/(tf.transpose(tf.matmul(tf.transpose(u), K)) + 1e-10)
    T = u*(tf.transpose(v) * K)
    if not backpropT:
        T = tf.stop_gradient(T)
    D = 2 * tf.reduce_sum(T * Mt)
    return D


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
    D = tf.maximum((C + tf.transpose(ny)) + nx, 0.0)
    return D    


def divide_t_c(concated_data, input_t):
        """Divide units into Treated and Control groups."""
    i0 = tf.cast((tf.where(input_t < 1)[:, 0]), tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:, 0]), tf.int32)

    group_T = tf.gather(concated_data, i1)
    group_C = tf.gather(concated_data, i0)

    return group_T, group_C, i0, i1


def split_train_val_test(data, train_ratio, val_ratio, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)
    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size + val_set_size]
    test_indices = shuffled_indices[train_set_size + val_set_size:]
    return train_indices, val_indices, test_indices


def save_my_model(save_path, save_name, need_save_model):
	cur_path = save_path + '/' + save_name
	need_save_model.save_weights(cur_path )
	print("Already saved the model's weights in file" + cur_path  )


def load_mymodel(load_path, load_name, need_load_model, config_hyperparameters, activation, adj):
	cur_model = need_load_model(config_hyperparameters, activation, adj)
	cur_path = load_path + '/' + load_name
	cur_model.load_weights(cur_path)
	print("load model")
	return cur_model


def save_results(save_result, save_name):
	np.save(save_name, save_result) 
	print("saved all results ")


def config_pare(
        iterations,
        lr_rate,
        lr_weigh_decay,
        flag_early_stop,
        use_batch, 
        st_hidden_layer, 
        proxy_layers, 
        self_loss_alpha,
        rep_alpha,
        out_dropout,
        GNN_dropout,
        rep_dropout,
        inp_dropout,
        proxy_dropout,
        st_inf_dropout,
        rep_hidden_layer,
        rep_hidden_shape,
        GNN_hidden_layer,
        GNN_hidden_shape,
        out_T_layer,
        out_C_layer,
        out_hidden_shape,
        activation,
        wass_lambda,
        flag_act_proxy,
        st_hidden_shape,
        reg_lambda,
        st_alpha,
        proxy_shape,
        eps
    ):

        cur_activation = activation
        config = {}
        config["iterations"] = iterations
        config["lr_rate"] = lr_rate
        config["lr_weigh_decay"] = lr_weigh_decay
        config["flag_early_stop"] = flag_early_stop
        config["flag_act_proxy"] = flag_act_proxy
        config['use_batch'] = use_batch
        config['rep_alpha'] = rep_alpha
        config['reg_lambda'] = reg_lambda
        config['st_alpha'] = st_alpha 
        config['wass_lambda'] = wass_lambda
        config['self_loss_alpha'] = self_loss_alpha
        config['eps'] = eps
        config['out_dropout'] = out_dropout
        config['GNN_dropout'] = GNN_dropout
        config['rep_dropout'] = rep_dropout
        config['inp_dropout'] = inp_dropout
        config['proxy_dropout'] = proxy_dropout
        config['st_inf_dropout'] = st_inf_dropout
        config['rep_hidden_layer'] = rep_hidden_layer
        config['GNN_hidden_layer'] = GNN_hidden_layer
        config['proxy_layers'] = proxy_layers
        config['out_T_layer'] = out_T_layer
        config['out_C_layer'] = out_C_layer
        config['st_hidden_layer'] = st_hidden_layer
        config['rep_hidden_shape'] = rep_hidden_shape
        config['GNN_hidden_shape'] = GNN_hidden_shape
        config['st_hidden_shape'] = st_hidden_shape
        config['out_hidden_shape'] = out_hidden_shape
        config['proxy_shape'] = proxy_shape
        return config, cur_activation


def load_data(data_name):
    data = []
    if data_name == 'flickr':
        xs = np.load("data/flk/flk_x.npy")
        adjs= np.load("data/flk/flk_A.npy")
        Ts = np.load("data/flk/flk_t.npy")
        all_yfs = np.load("data/flk/flk_yf.npy")
        all_y1s = np.load("data/flk/flk_y1.npy")
        all_y0s = np.load("data/flk/flk_y0.npy")
    elif data_name == 'blog':
        xs = np.load("data/blog/blog_x.npy")
        adjs= np.load("data/blog/blog_A.npy")
        Ts = np.load("data/blog/blog_t.npy")
        all_yfs = np.load("data/blog/blog_yf.npy")
        all_y1s = np.load("data/blog/blog_y1.npy")
        all_y0s = np.load("data/blog/blog_y0.npy")
    elif data_name == "AMZ-N":
        load_data = pd.read_csv("../data/AmazonItmFeatures_neg.csv",header=None,prefix="col")
        data_n = load_data.to_numpy()
        prod_G = np.load("../data/new_product_graph_neg.npz")
        datas = prod_G['data']
        indices = prod_G['indices']
        indptr = prod_G['indptr']
        csr_mat = csr_matrix((datas, indices, indptr), dtype=int)
        arr = csr_mat.toarray()
        shape = prod_G['shape']
        adjs = arr[:14538, :14538]
        Ts = data_n[:, 0]
        all_y1s = data_n[:, 1]
        all_y0s = data_n[:, 2]
        yfs = len(all_y1s) * [0]
        for i in range(len(all_y1s)):
            if Ts[i]>0:
                yfs[i] = all_y1s[i]
            else:
                yfs[i] = all_y0s[i]

        all_yfs = np.array(yfs)
        xs = data_n[:, 5:]
    all_yfs = all_yfs.reshape(len(all_yfs), 1)
    Ts = Ts.reshape(len(Ts), 1)
    all_y1s = all_y1s.reshape(len(all_y1s), 1)
    all_y0s = all_y0s.reshape(len(all_y0s), 1)
    data.append(xs)
    data.append(adjs)
    data.append(Ts)
    data.append(all_yfs)
    data.append(all_y1s)
    data.append(all_y0s)
    return data


def data_preparation(data_name, data):
    xs = data[0]
    adjs= data[1]
    Ts = data[2]
    all_yfs = data[3]
    all_y1s = data[4]
    all_y0s = data[5]
    all_y1s = all_y1s.reshape(len(all_y1s), 1)
    all_y0s = all_y0s.reshape(len(all_y0s), 1)
    cur_ites_true = all_y1s - all_y0s
    cur_all_inputs = np.concatenate([xs, Ts], axis=1)
    return cur_all_inputs, cur_ites_true 


def train(model_name, cur_all_inputs, data, config, val_idx, train_idx, data_name, activation=tf.nn.relu):
	losslist=[]
	cur_init_A = data[1]
	self_loop  = tf.ones(len(cur_init_A), dtype=tf.float32)
	cur_A = tf.cast(tf.linalg.set_diag(tf.cast((cur_init_A.T > 0 + 0.0), tf.float32), self_loop), tf.float32)
	all_yfs = data[3]
	cur_model = model_name(config, activation=activation, init_adj=cur_A) 
	count = 0
	losslist_val = []
	sum_loss = 0
	sum_val_loss = 0
	losslist = []
	start_time = time.time()
	start_early_stop = 400

	for i in range(config['iterations']):
		print("iter", i)
		loss = cur_model.val_y(tf.cast(cur_all_inputs, tf.float32),tf.cast(all_yfs, tf.float32), train_idx)
		total_loss = cur_model.network_learn(tf.cast(cur_all_inputs, tf.float32),tf.cast(all_yfs, tf.float32), train_idx)
		val_loss = cur_model.val_y(tf.cast(cur_all_inputs, tf.float32),  tf.cast(all_yfs, tf.float32), val_idx)
		sum_loss += loss
		sum_val_loss += val_loss
		if (i+1) % 20 == 0:
			if len(losslist_val) > 0 and sum_val_loss / 20 >= losslist_val[-1]:
				count += 1
			else:
				count = 0
			if config['flag_early_stop']:
				if i > start_early_stop and count >= 1:
					break
			losslist.append(sum_loss/20)
			losslist_val.append(sum_val_loss/20)
			sum_loss = 0
			sum_val_loss = 0 
	return cur_model


def implement(config, data_name, model_name, f_activation, start_i, end_i):
    data = load_data(data_name)
    train_idx, val_idx, test_idx = split_train_val_test(data[0], 0.7, 0.15, 0.15)
    if data_name == "AMZ-N":
        mu1 = data[4] * 1.0
        mu0 = data[5] * 1.0
        train_indices = train_idx
        val_indices = val_idx
        test_indices = test_idx

        mean_m1_train = np.mean(mu1[train_indices])
        std_m1_train = np.std(mu1[train_indices])
        mu1[train_indices] = (mu1[train_indices]-mean_m1_train) / (std_m1_train )

        mean_m1_val = np.mean(mu1[val_indices])
        std_m1_val = np.std(mu1[val_indices])
        mu1[val_indices] = (mu1[val_indices]-mean_m1_val) / (std_m1_val )

        mean_m1_test = np.mean(mu1[test_indices])
        std_m1_test = np.std(mu1[test_indices])
        mu1[test_indices] = (mu1[test_indices]-mean_m1_test) / (std_m1_test)

        mean_m0_train = np.mean(mu0[train_indices])
        std_m0_train = np.std(mu0[train_indices])
        mu0[train_indices] = (mu0[train_indices]-mean_m0_train) / (std_m0_train )

        mean_m0_val = np.mean(mu0[val_indices])
        std_m0_val = np.std(mu0[val_indices])
        mu0[val_indices] = (mu0[val_indices]-mean_m0_val) / (std_m0_val)
        mean_m0_test = np.mean(mu0[test_indices])
        std_m0_test = np.std(mu0[test_indices])
        mu0[test_indices] = (mu0[test_indices]-mean_m0_test) / (std_m0_test)
        data[5] = mu0
        data[4] = mu1

        cur_yf = data[3] * 1.0
        mean_yf_train = np.mean(cur_yf[train_indices])
        std_yf_train = np.std(cur_yf[train_indices])
        cur_yf[train_indices] = (cur_yf[train_indices]-mean_yf_train) / (std_yf_train )
        mean_yf_val = np.mean(cur_yf[val_indices])
        std_yf_val = np.std(cur_yf[val_indices])
        cur_yf[val_indices] = (cur_yf[val_indices]-mean_yf_val) / (std_yf_val)
        mean_yf_test = np.mean(cur_yf[test_indices])
        std_yf_test = np.std(cur_yf[test_indices])
        cur_yf[test_indices] = (cur_yf[test_indices]-mean_yf_test) / (std_yf_test)
        data[3] = cur_yf
    cur_all_inputs, cur_ites_true = data_preparation(data_name, data)

    for i in range(start_i, end_i):
        cur_model = train(model_name, cur_all_inputs, data, config, val_idx, train_idx, data_name=data_name, activation=f_activation)
        cur_save_model_name = data_name + str(model_name)[8:-2]  + "alpha_rep" + str(config['rep_alpha']) + "drop" + str(config['out_dropout']) + "_" + "split_" + str(i)
        cur_save_path = './models/Model_'+ data_name + "_" + str(model_name)[8:-2] + "_split_" + str(i) 
        os.makedirs(cur_save_path, exist_ok=True)
        save_my_model(cur_save_path, cur_save_model_name, cur_model)

        cur_val_results = []
        mse_y, pehe, err_ate = evaluation.evaluate(cur_model, cur_all_inputs, val_idx, cur_ites_true, data[2], data[3])
        cur_val_results.append(mse_y)
        cur_val_results.append(pehe)
        cur_val_results.append(err_ate)
        cur_val_results_name = './results/results_' + data_name + "_" + str(model_name)[8:-2] + "_val" + "_split_" + str(i) 
        save_results(cur_val_results, cur_val_results_name)

        cur_test_results = []
        mse_y, pehe, err_ate = evaluation.evaluate(cur_model, cur_all_inputs, test_idx, cur_ites_true, data[2], data[3])
        cur_test_results.append(mse_y)
        cur_test_results.append(pehe)
        cur_test_results.append(err_ate)
        cur_test_results_name = './results/test_results_' + data_name + "_" + str(model_name)[8:-2] + "_test" + "_split_" + str(i) 
        save_results(cur_test_results, cur_test_results_name)
