import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax
import numpy as np

query_term_maxlen = 5
hist_size = 30
# vocab_size = 1000
# embed_size = 1
# embed = np.float32(np.random.uniform(-0.2, 0.2, [vocab_size, embed_size]))
num_layers = 2
hidden_sizes = [5, 1]

initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)


#
# returns the raw keras model object
#
def build_keras_model():

    #
    # input layers (query and doc)
    #

    # -> the query idf input (1d array of float32)
    query = Input(name='query', shape=(query_term_maxlen,1))

    # -> the histogram (2d array: every query gets 1d histogram
    doc = Input(name='doc', shape=(query_term_maxlen, hist_size))

    #
    # the histogram handling part (feed forward network)
    #

    z = doc
    for i in range(num_layers):
        z = Dense(hidden_sizes[i], kernel_initializer=initializer_fc)(z)
        z = Activation('tanh')(z)

    z = Permute((2, 1))(z)
    z = Reshape((query_term_maxlen,))(z)

    #
    # the query term idf part
    #

    q_w = Dense(1, kernel_initializer=initializer_gate, use_bias=False)(query) # what is that doing here ??
    q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(query_term_maxlen,))(q_w)
    q_w = Reshape((query_term_maxlen,))(q_w) # isn't that redundant ??

    #
    # combination of softmax(query term idf) and feed forward result per query term
    #
    out_ = Dot(axes=[1, 1])([z, q_w])

    model = Model(inputs=[query, doc], outputs=[out_])

    return model
