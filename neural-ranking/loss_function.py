 # from https://github.com/faneshion/MatchZoo/blob/master/matchzoo/losses/rank_losses.py

from keras.losses import *
from keras.layers import Lambda
from keras.utils.generic_utils import deserialize_keras_object

#
# y_true is IGNORED (!), you don't have to set a label to train (?)
# y_pred needs to contain 2 runs (?)
#
def rank_hinge_loss(y_true, y_pred):

    y_pos = Lambda(lambda
                       a
                   : a[::2, :]
                   , output_shape= (1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)

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