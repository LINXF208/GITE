import numpy as np 
import tensorflow as tf
from tensorflow import keras

import utils
import layers
import evaluation
import GITE


def main(dataname):
    configs, activation = utils.config_pare(
        iterations=2000,
        lr_rate=0.001,
        lr_weigh_decay=0.001,
        flag_early_stop=True,
        use_batch=1024, 
        st_hidden_layer=3, 
        proxy_layers=1,
        rep_alpha=0.1,
        out_dropout=0.1,
        GNN_dropout=0.1,
        rep_dropout=0.1,
        proxy_dropout=0.1,
        st_inf_dropout=0.1,
        inp_dropout = 0.0,
        rep_hidden_layer=3,
        rep_hidden_shape=[100, 100, 100],
        self_loss_alpha=0.0,
        eps=0.0,
        GNN_hidden_layer=3,
        GNN_hidden_shape=[100, 100, 100],
        st_hidden_shape=[100, 100, 100],
        out_T_layer=3,
        out_C_layer=3,
        out_hidden_shape=[100, 100, 100],
        proxy_shape=[300],
        activation=tf.nn.relu,
        reg_lambda=0.01,
        flag_act_proxy=False,
        st_alpha=1.0,
        wass_lambda=1.0
    )
    utils.implement(
        config=configs,
        data_name=dataname,
        model_name=GITE.GITEModel,
        f_activation=activation,
        start_i=0,
        end_i=10
    )


if __name__ == '__main__':
    main('AMZ-N')
