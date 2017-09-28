import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax
import numpy as np

text1_maxlen = 5
hist_size = 30
vocab_size = 1000
embed_size = 1
embed = np.float32(np.random.uniform(-0.2, 0.2, [vocab_size, embed_size]))
num_layers = 2
hidden_sizes = [5, 1]

initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)



def build_keras_model():
    #def tensor_product(x):
    #    a = x[0]
    #    b = x[1]
    #    y = K.batch_dot(a, b, axis=1)
    #    y = K.einsum('ijk, ikl->ijl', a, b)
    #    return y

    # -> the query word, so we can get the idf (?) from the embedding !
    query = Input(name='query', shape=(text1_maxlen,))
    print('[Input] query:\t%s' % str(query.get_shape().as_list()))

    # -> the histogram
    doc = Input(name='doc', shape=(text1_maxlen, hist_size))
    print('[Input] doc:\t%s' % str(doc.get_shape().as_list()))

    # embedding is the idf of every query word
    embedding = Embedding(vocab_size, embed_size, weights=[embed],
                          trainable=False)
    # query gat handling
    q_embed = embedding(query)
    print('[Embedding] q_embed:\t%s' % str(q_embed.get_shape().as_list()))

    q_w = Dense(1, kernel_initializer=initializer_gate, use_bias=False)(q_embed)
    print('[Dense] q_gate:\t%s' % str(q_w.get_shape().as_list()))

    q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(text1_maxlen,))(q_w)
    print('[Softmax] q_gate:\t%s' % str(q_w.get_shape().as_list()))
    # end query handling

    # histogram feed forward layers
    z = doc
    for i in range(num_layers):
        z = Dense(hidden_sizes[i], kernel_initializer=initializer_fc)(z)
        z = Activation('tanh')(z)
        print('[Dense] z (full connection):\t%s' % str(z.get_shape().as_list()))

    z = Permute((2, 1))(z)
    z = Reshape((text1_maxlen,))(z)
    print('[Reshape] z (matching) :\t%s' % str(z.get_shape().as_list()))

    q_w = Reshape((text1_maxlen,))(q_w)
    print('[Reshape] q_w (gating) :\t%s' % str(q_w.get_shape().as_list()))

    # combination of softmax(query term idf) and feed forward result per query term (!?)
    out_ = Dot(axes=[1, 1])([z, q_w])
    print('[Dot] out_ :\t%s' % str(out_.get_shape().as_list()))

    model = Model(inputs=[query, doc], outputs=[out_])
    return model