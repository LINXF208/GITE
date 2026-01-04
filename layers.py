import tensorflow as tf
from tensorflow import keras


class GINLayer(keras.layers.Layer):
    def __init__(self, hidden_units, eps, activation=tf.nn.relu):
        super(GINLayer,self).__init__()
        self.hidden_units = hidden_units
        self.eps = eps   
        self.activation = activation

    def build(self,input_shape):
        self.eps = self.add_variable("eps", shape=[1], initializer=tf.constant_initializer(self.eps), trainable=True, dtype=tf.float32)
        self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.hidden_units], dtype=tf.float32, initializer=tf.keras.initializers.glorot_uniform())
        self.bias = self.add_variable("bias", shape=[self.hidden_units])
        
    def call(self, input, self_features, adj):
        neighbors_features = tf.matmul(adj, input)
        support = (1+self.eps) * self_features + neighbors_features
        output = self.activation(tf.matmul(support, self.kernel) + self.bias)
        return output


class ISGATLayer(keras.layers.Layer):
    def __init__(self, att_embedding_size=8, head_num=8, activation=tf.nn.relu, eps=0.5, reduction='mean', use_bias=False, flag_st=True, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.eps=eps   
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.activation = activation
        self.act = activation
        self.reduction = reduction
        self.use_bias = use_bias
        self.flag_st = flag_st
        super(ISGATLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        X, st = input_shape
        embedding_size = int(X[-1])
        st_emb_size = int(st[-1])
        self.eps = self.add_variable(
            name="eps", 
            shape=[1], 
            initializer=tf.constant_initializer(self.eps), 
            trainable=True, 
            dtype=tf.float32
        )
        self.weight = self.add_weight(
            name='weight', 
            shape=[embedding_size, self.att_embedding_size], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.att_self_weight = self.add_weight(
            name='att_self_weight',
            shape=[1, self.att_embedding_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.att_neighs_weight = self.add_weight(
            name='att_neighs_weight',
            shape=[1, 
            self.att_embedding_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.weight_st_f = self.add_weight(
            name='weight_st_f', 
            shape=[embedding_size, self.att_embedding_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )
        if self.flag_st:
            self.weight_st = self.add_weight(
                name='weight_st',
                shape=[st_emb_size, self.att_embedding_size],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_uniform()
            )
            self.att_self_weight_st = self.add_weight(
                name='att_self_weight_st',
                shape=[1, self.att_embedding_size],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_uniform()
            )
            self.att_neighs_weight_st = self.add_weight(
                name='att_neighs_weight_st',
                shape=[1, self.att_embedding_size],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_uniform()
            )
        super(ISGATLayer, self).build(input_shape)

    def call(self, input, A, cal_degrees, training=None, **kwargs):
        X, aggregated_degree = input
        features = tf.matmul(X, self.weight)
        features = tf.reshape(features, [-1,  self.att_embedding_size]) 
        attn_for_self = tf.reduce_sum(features * self.att_self_weight, axis=-1, keepdims=True) 
        attn_for_neighs = tf.reduce_sum(features * self.att_neighs_weight, axis=-1, keepdims=True)
        dense =  attn_for_self + tf.transpose(attn_for_neighs)
        dense = tf.nn.leaky_relu(dense, alpha=0.2)
        mask = -10e9 * (1.0 - tf.sign(A))
        dense += mask
        normalized_att_scores = tf.nn.softmax(dense, axis=-1,)
        att_score = tf.squeeze(normalized_att_scores)

        if self.flag_st:
            st_features = tf.matmul(aggregated_degree, self.weight_st)  
            st_features = tf.reshape(st_features, [-1,  self.att_embedding_size]) 
            attn_st_for_self = tf.reduce_sum(st_features * self.att_self_weight_st, axis=-1, keepdims=True)  
            attn_st_for_neighs = tf.reduce_sum(st_features * self.att_neighs_weight_st, axis=-1, keepdims=True)
            dense_st =  attn_st_for_self + tf.transpose(attn_st_for_neighs)
            dense_st = tf.nn.leaky_relu(dense_st, alpha=0.2)
            mask_st = -10e9 * (1.0 - tf.sign(A))
            dense_st += mask_st
            normalized_att_scores_st = tf.nn.softmax(dense_st, axis=-1,) 
            att_score_st = tf.squeeze(normalized_att_scores_st)

        if self.flag_st:
            return att_score, att_score_st
        else:
            return att_score

class WGNNLayer(keras.layers.Layer):
    def __init__(self, hidden_units, activation=tf.nn.relu, eps=0.5):
        super(WGNNLayer,self).__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.eps=eps

    def build(self,input_shape):
        self.kernel = self.add_variable(
            "kernel", 
            shape = [int(input_shape[-1]), 
            self.hidden_units], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.eps = self.add_variable(
            "eps", 
            shape=[1], 
            initializer=tf.constant_initializer(self.eps), 
            trainable=True, 
            dtype=tf.float32
        )
        self.bias = self.add_variable(
            "bias",
            shape=[self.hidden_units]
        )
        self.kernel_2 = self.add_variable(
            "kernel_2", 
            shape=[int(input_shape[-1]), 
            self.hidden_units], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.bias_2 = self.add_variable(
            "bias",
            shape=[self.hidden_units]
        )

    def call(self, input, cal_degrees, weights_adj, weights_adj_st, first_flag):
        if first_flag:
            support = tf.matmul(input,self.kernel) + self.bias
            support_s = tf.matmul(input,self.kernel_2) + self.bias_2
            weights_deleted = tf.linalg.set_diag(weights_adj, tf.zeros(weights_adj.shape[0]))
            weights_deleted_st = tf.linalg.set_diag(weights_adj_st, tf.zeros(weights_adj_st.shape[0]))
            result = tf.matmul(weights_deleted,support)
            result_st = tf.matmul(weights_deleted_st,support_s)
            results = self.eps*result + result_st * (1-self.eps)
            results = (results+cal_degrees * results)
            output = self.activation(results)
        else:
            support = tf.matmul(input,self.kernel) + self.bias     
            support_s = tf.matmul(input,self.kernel_2) + self.bias_2
            result = tf.matmul(weights_adj, support)
            result_st = tf.matmul(weights_adj_st, support_s)
            results = self.eps * result + result_st * (1-self.eps)
            results = (results + cal_degrees * results)
            output = self.activation(results)
        return output


class proxylayer(keras.layers.Layer):
    def __init__(self, hidden_units, activation, act_flag=False):
        super(proxylayer, self).__init__()
        self.hidden_units = hidden_units
        self.act_flag = act_flag
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.hidden_units])
        self.bias = self.add_variable("bias", shape=[self.hidden_units])

    def call(self, input):
        output = tf.matmul(input, self.kernel)+self.bias
        if self.act_flag:
            output = self.activation(output)
        return output


