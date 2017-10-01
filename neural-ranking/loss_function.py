# from https://github.com/faneshion/MatchZoo/blob/master/matchzoo/losses/rank_losses.py
from keras.backend import tf
from keras.losses import *
from keras.layers import Lambda
from keras.utils.generic_utils import deserialize_keras_object

#
# y_true is IGNORED (!), you don't have to set a label to train (?)
# y_pred contains the complete batch (!)
#  -> the slicing splits the tensors in even and odd (pos and negative from the input)
#  -> VERY IMPORTANT: The input data must not be shuffled !! shuffle = False
#
def rank_hinge_loss(y_true, y_pred):

    #y_pred = tf.Print(y_pred, [y_pred], 'all',summarize=10)
    #y_true = tf.Print(y_true, [y_true], 'y_true',summarize=10)

    y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)

    #y_true_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_true)

    #y_pos = tf.Print(y_pos, [y_pos], 'y_pos',summarize=10)
    #y_neg = tf.Print(y_neg, [y_neg], 'y_neg',summarize=10)

    loss = K.maximum(0., 1. + y_neg - y_pos)
    return K.mean(loss)

def serialize(rank_loss):
    return rank_loss.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)

# val = np.random.random((10, 1))
# t = K.variable(value=val)
# t1 = t[::2, :]
# t2 = t[1::2, :]
#
# print('t:\n', K.eval(t))
# print('t1:\n', K.eval(t1))
# print('t2:\n', K.eval(t2))
#