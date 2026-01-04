import layers
import utils
import numpy as np 
import tensorflow as tf

from sklearn import metrics
from sklearn.utils import shuffle

import pandas as pd
import math
from tensorflow import keras
import random


class GITEModel(keras.Model):
    def __init__(self, config, init_adj, activation):
        super(GITEModel, self).__init__()
        print("Initialization ...")

        self.init_adj = init_adj
        self.degrees = tf.reshape(tf.reduce_sum(init_adj, axis=1), (len(init_adj), 1))
        self.rep_layers = []
        self.gnn_layers = []
        self.att_layers = []
        self.att_layers_t = []
        self.proxy_layers = [] 
        self.out_T_layers = []
        self.out_C_layers = []
        self.st_model_layers = []
        self.activation = activation
        self.use_batch = config['use_batch']
        self.rep_alpha = config['rep_alpha']
        self.reg_lambda = config['reg_lambda']
        self.st_alpha = config['st_alpha']
        self.wass_lambda = config['wass_lambda']
        self.self_loss_alpha = config['self_loss_alpha'] 
        self.out_dropout =  config['out_dropout']
        self.st_inf_dropout = config['st_inf_dropout']
        self.GNN_dropout = config['GNN_dropout']
        self.rep_dropout = config['rep_dropout'] 
        self.proxy_dropout = config['proxy_dropout']
        self.inp_drop = config['inp_dropout']
        self.input_rep_layers = []
        self.train_loss = None
        self.result = None

        self.optimizer= keras.optimizers.Adam(lr=config["lr_rate"], decay = config["lr_weigh_decay"])
        self.layer_norm = keras.layers.LayerNormalization()
        self.pi_eta = self.add_variable("eps", shape=[1], initializer=tf.constant_initializer(1.0), trainable=False, dtype=tf.float32)

        for i in range(config['rep_hidden_layer']):
            r = keras.layers.Dense(config['rep_hidden_shape'][i], activation=self.activation)
            self.input_rep_layers.append(r)

        for i in range(config['st_hidden_layer']):
            st = layers.GINLayer(config['st_hidden_shape'][i], config['eps'], activation=self.activation)
            self.st_model_layers.append(st)

        for i in range(config['rep_hidden_layer']):
            att = ISGATTLayer(rep_hidden_shape[i], head_num=1, activation=self.activation)
            att_t = ISGATTLayer(rep_hidden_shape[i], head_num=1, activation=self.activation, flag_st=False)
            self.att_layers.append(att)
            self.att_layers_t.append(att_t)

        for i in range(config['rep_hidden_layer']):
            h = layers.WGNNLayer(config['GNN_hidden_shape'][i], activation=self.activation)
            self.rep_layers.append(h)

        for i in range(config['GNN_hidden_layer']):
            g = layers.WGNNLayer(config['GNN_hidden_shape'][i], activation=self.activation)
            self.gnn_layers.append(g)

        for i in range(config['proxy_layers']):
            J = layers.proxylayer(config['proxy_shape'][i], activation=self.activation, act_flag=config["flag_act_proxy"])
            self.proxy_layers.append(J) 

        for i in range(config['out_T_layer']):
            out_T = keras.layers.Dense(
                config['out_hidden_shape'][i], 
                activation=self.activation, 
                kernel_initializer=tf.keras.initializers.glorot_uniform(), 
                kernel_regularizer=keras.regularizers.l2(0.0)
            )
            self.out_T_layers.append(out_T)

        for i in range(config['out_C_layer']):
            out_C = keras.layers.Dense(
                config['out_hidden_shape'][i], 
                activation=self.activation,
                kernel_initializer=tf.keras.initializers.glorot_uniform(), 
                kernel_regularizer=keras.regularizers.l2(0.0)
            )
            self.out_C_layers.append(out_C)

        self.out_layer_t = keras.layers.Dense(1)
        self.out_layer_c = keras.layers.Dense(1)

    def call(self, inputtensor, idx, training=False):
        print("Call ...")
        input_tensor = inputtensor[:, :-1]
        input_t = tf.constant(inputtensor[:, -1], shape=[input_tensor.shape[0], 1])
        features = input_tensor

        st_inf = tf.ones_like(self.degrees) 
        aggregated_sts = []
        for i in range(len(self.st_model_layers)):
            st_inf = self.st_model_layers[i](st_inf, st_inf, self.init_adj)
            aggregated_sts.append(st_inf)

        log_degree = tf.math.log(self.degrees + 1)
        aggregated_st_cal = self.pi_eta * log_degree / tf.reduce_sum(log_degree)

        input_rep = input_tensor
        for i in range(len(self.rep_layers)):
            input_rep = self.input_rep_layers[i](input_rep)

        hidden = input_tensor
        GNN = input_t
        concat_t_rep = tf.concat([hidden, GNN], axis=1)

        for i in range(len(self.rep_layers)):
            att_weight, att_weight_st = self.att_layers[i]([hidden, aggregated_sts[i]], self.init_adj, aggregated_st_cal)
            att_weight_t = self.att_layers_t[i]([concat_t_rep, aggregated_sts[i]], self.init_adj, aggregated_st_cal)
            if i == 0:
                GNN = self.gnn_layers[i](GNN, aggregated_st_cal, att_weight_t, att_weight_st, first_flag=True)
                hidden = self.rep_layers[i](hidden, aggregated_st_cal, att_weight, att_weight_st, first_flag=False)                             
            else:
                GNN = self.gnn_layers[i](GNN, aggregated_st_cal, att_weight_t, att_weight_st, first_flag=False)
                hidden = self.rep_layers[i](hidden, aggregated_st_cal,att_weight, att_weight_st, first_flag=False)
            concat_t_rep = tf.concat([hidden, GNN], axis=1)

        concated_data = tf.concat([input_rep, hidden, GNN], axis=1)
        cur_concated_data = tf.gather(concated_data, idx)

        J_in = self.layer_norm(cur_concated_data)
        for i in range(len(self.proxy_layers)):
            J_in = self.proxy_layers[i](J_in)

        cur_input_t = tf.gather(input_t, idx)
        cur_hidden = tf.gather(hidden, idx)
        cur_GNN = tf.gather(GNN, idx)
        group_t, group_c = cur_concated_data, cur_concated_data

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.out_layer_t(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.out_layer_c(outnn_C)
        
        return output_T, output_C

    def get_loss(self, inputtensor, all_y, train_idx, training=True):
        input_tensor = inputtensor[:, :-1]
        input_t = tf.constant(inputtensor[:, -1], shape=[input_tensor.shape[0], 1])
        features = input_tensor
        reg_loss = 0.0

        st_inf = tf.ones_like(self.degrees) 
        aggregated_sts = []
        for i in range(len(self.st_model_layers)):
            st_inf = self.st_model_layers[i](st_inf, st_inf, self.init_adj)
            st_inf = tf.nn.dropout(st_inf, self.st_inf_dropout)
            aggregated_sts.append(st_inf)
            reg_loss += tf.nn.l2_loss(self.st_model_layers[i].kernel)
            reg_loss += tf.nn.l2_loss(self.st_model_layers[i].eps)

        log_degree = tf.math.log(self.degrees+1)
        train_log_degree = tf.gather(log_degree, train_idx)
        aggregated_st_cal = log_degree / tf.reduce_sum(train_log_degree)

        input_rep = input_tensor
        for i in range(len(self.rep_layers)):
            input_rep = self.input_rep_layers[i](input_rep)
            input_rep = tf.nn.dropout(input_rep,self.rep_dropout)
            reg_loss += tf.nn.l2_loss(self.input_rep_layers[i].kernel)
        
        hidden = input_tensor
        GNN = input_t
        concat_t_rep = tf.concat([hidden, GNN], axis=1)

        for i in range(len(self.rep_layers)):
            att_weight, att_weight_st = self.att_layers[i]([hidden, aggregated_sts[i]], self.init_adj, aggregated_st_cal)
            att_weight_t = self.att_layers_t[i]([concat_t_rep, aggregated_sts[i]], self.init_adj, aggregated_st_cal)
            if i == 0:
                GNN = self.gnn_layers[i](GNN, aggregated_st_cal, att_weight_t, att_weight_st, first_flag=True)
                hidden = self.rep_layers[i](hidden, aggregated_st_cal, att_weight, att_weight_st, first_flag=False)                             
            else:
                GNN = self.gnn_layers[i](GNN, aggregated_st_cal, att_weight_t, att_weight_st, first_flag=False)
                hidden = self.rep_layers[i](hidden, aggregated_st_cal,att_weight, att_weight_st, first_flag=False)
            hidden = tf.nn.dropout(hidden, self.rep_dropout)
            GNN = tf.nn.dropout(GNN, self.GNN_dropout)
            concat_t_rep = tf.concat([hidden, GNN], axis=1)
            
            reg_loss += tf.nn.l2_loss(self.gnn_layers[i].kernel)                        
            reg_los += tf.nn.l2_loss(self.rep_layers[i].kernel)
            reg_los += tf.nn.l2_loss(self.gnn_layers[i].kernel_2)                        
            reg_los += tf.nn.l2_loss(self.rep_layers[i].kernel_2)                  
            reg_los += tf.nn.l2_loss(self.att_layers[i].weight)
            reg_los += tf.nn.l2_loss(self.att_layers[i].att_self_weight)
            reg_los += tf.nn.l2_loss(self.att_layers[i].att_neighs_weight)
            reg_los += tf.nn.l2_loss(self.att_layers[i].weight_st)
            reg_los += tf.nn.l2_loss(self.att_layers[i].att_self_weight_st)
            reg_los += tf.nn.l2_loss(self.att_layers[i].att_neighs_weight_st)
            reg_los += tf.nn.l2_loss(self.att_layers_t[i].weight)
            reg_los += tf.nn.l2_loss(self.att_layers_t[i].att_self_weight)
            reg_los += tf.nn.l2_loss(self.att_layers_t[i].att_neighs_weight)   

        concated_data = tf.concat([input_rep, hidden, GNN], axis = 1)
        train_concated_data = tf.gather(concated_data, train_idx)
        train_input_t = tf.gather(input_t, train_idx)
        train_y = tf.gather(all_y, train_idx)
        train_hidden = tf.gather(hidden, train_idx)
        train_GNN = tf.gather(GNN, train_idx)

        pt_trian = tf.divide(tf.reduce_sum(train_input_t), train_input_t.shape[0]) 
        if self.use_batch:
            I = random.sample(range(0, len(train_concated_data)), self.use_batch)
            train_concated_data = tf.gather(train_concated_data, I)
            train_input_t = tf.gather(train_input_t, I)
            train_y = tf.gather(train_y, I)
            train_hidden = tf.gather(train_hidden, I)
            train_GNN = tf.gather(train_GNN, I)
            
        norm_concated = self.layer_norm(train_concated_data)
        J_in = norm_concated * 1.0
        for i in range(len(self.proxy_layers)):
            J_in = self.proxy_layers[i](J_in)
            J_in = tf.nn.dropout(J_in, self.proxy_dropout)
            reg_loss += tf.nn.l2_loss(self.proxy_layers[i].kernel)
        if self.self_loss_alpha > 0:
            self_loss = tf.reduce_mean(tf.square(J_in - norm_concated))
        else:
            self_loss = 0.0

        group_t, group_c, i_0, i_1 = utils.divide_t_c(train_concated_data, train_input_t)
        
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            outnn_T = tf.nn.dropout(outnn_T,self.out_dropout)
            reg_loss += tf.nn.l2_loss(self.out_T_layers[i].kernel)
        output_T = self.out_layer_t(outnn_T)
        
        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            outnn_C = tf.nn.dropout(outnn_C,self.out_dropout)
            reg_loss += tf.nn.l2_loss(self.out_C_layers[i].kernel)
        output_C = self.out_layer_c(outnn_C)  
        y_pre = tf.dynamic_stitch([i_0,i_1],[output_C,output_T])

        factual_loss = tf.reduce_mean(tf.square((train_y - y_pre))) 
        rep_loss = self.rep_alpha * utils.wasserstein(
            J_in,
            output_T,
            output_C,
            train_input_t,
            train_y,
            pt_trian,
            self.wass_lambda,
            5.0,
            50
        )
        if np.isnan(rep_loss):
            rep_loss = 0.0

        loss =   rep_loss + factual_loss  +  self.reg_lambda * reg_loss + self.self_loss_alpha * self_loss
        return loss

    def get_grad(self, inputtensor, y, train_idx):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            L = self.get_loss(inputtensor, y, train_idx)
            self.train_loss = L
            g = tape.gradient(L,self.variables)
        return g

    def network_learn(self, inputtensor, y, train_idx):
        g = self.get_grad(inputtensor, y, train_idx)
        self.optimizer.apply_gradients(zip(g, self.variables))
        return self.train_loss

    def val_y(self,inputtensor, y, train_idx, training=False):
        input_tensor = inputtensor[:, :-1]
        input_t = tf.constant(inputtensor[:, -1], shape=[input_tensor.shape[0], 1])
        features = input_tensor

        st_inf = tf.ones_like(self.degrees) 
        aggregated_sts = []
        for i in range(len(self.st_model_layers)):
            st_inf = self.st_model_layers[i](st_inf, st_inf, self.init_adj)
            aggregated_sts.append(st_inf)
        
        log_degree = tf.math.log(self.degrees + 1)
        train_log_degree = tf.gather(log_degree, train_idx)
        aggregated_st_cal = self.pi_eta*log_degree / tf.reduce_sum(train_log_degree)
        
        input_rep = input_tensor
        for i in range(len(self.rep_layers)):
            input_rep = self.input_rep_layers[i](input_rep)

        hidden = input_tensor
        GNN = input_t
        concat_t_rep = tf.concat([hidden, GNN], axis=1)

        for i in range(len(self.rep_layers)):
            att_weight, att_weight_st = self.att_layers[i]([hidden, aggregated_sts[i]], self.init_adj, aggregated_st_cal)
            att_weight_t = self.att_layers_t[i]([concat_t_rep, aggregated_sts[i]], self.init_adj, aggregated_st_cal)
            if i == 0:
                GNN = self.gnn_layers[i](GNN, aggregated_st_cal, att_weight_t, att_weight_st, first_flag=True)
                hidden = self.rep_layers[i](hidden, aggregated_st_cal, att_weight, att_weight_st, first_flag=False)                             
            else:
                GNN = self.gnn_layers[i](GNN, aggregated_st_cal, att_weight_t, att_weight_st, first_flag=False)
                hidden = self.rep_layers[i](hidden, aggregated_st_cal,att_weight, att_weight_st, first_flag=False)
            concat_t_rep = tf.concat([hidden, GNN], axis=1)
            
        concated_data = tf.concat([input_rep, hidden, GNN], axis=1)
        train_concated_data = tf.gather(concated_data, train_idx)
        train_input_t = tf.gather(input_t, train_idx)
        train_y = tf.gather(y, train_idx)
        train_hidden = tf.gather(hidden, train_idx)
        train_GNN = tf.gather(GNN, train_idx)
        pt_trian = tf.divide(tf.reduce_sum(train_input_t), train_input_t.shape[0])
        train_input_rep = tf.gather(input_rep, train_idx)

        J_in = self.layer_norm(train_concated_data)
        for i in range(len(self.proxy_layers)):
            J_in = self.proxy_layers[i](J_in)

        group_t, group_c, i_0, i_1= utils.divide_t_c(train_concated_data, train_input_t)
        
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.out_layer_t(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.out_layer_c(outnn_C)

        y_pre = tf.dynamic_stitch([i_0, i_1], [output_C, output_T])
        factual_loss = tf.reduce_mean(tf.square((train_y - y_pre)))  
        print("pre_y val loss", factual_loss)
        return factual
