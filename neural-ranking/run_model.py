import keras
from keras.utils import plot_model
from keras_model import build_keras_model
from load_data import load_data
from loss_function import *
import numpy as np

train_data = load_data('../data/5-folds/30_histogram_fold_1.test')

model = build_keras_model()
model.summary()
model.compile(loss = rank_hinge_loss, optimizer='adam')

plot_model(model, to_file='model.png', show_shapes=True)

batch_size = 1
test1 = train_data['653'][0]
test2 = train_data['653'][7]

# np histogram
X2 = np.zeros((batch_size*2, 5, 30), dtype=np.float32)

X2[0][0] = test1[3][0]
X2[0][1] = test1[3][1]
X2[0][2] = test1[3][2]

X2[1][0] = test2[3][0]
X2[1][1] = test2[3][1]
X2[1][2] = test2[3][2]

# topic idf
X1 = np.zeros((batch_size*2, 5,1), dtype=np.float32)

X1[0] = test1[2]
X1[1] = test2[2]

# empty label array
Y = np.zeros((batch_size*2,), dtype=np.int32)
Y[::2] = 1

c1=keras.callbacks.TensorBoard(log_dir='./tensorboard-logs', histogram_freq=0,
          write_graph=True, write_images=True)


model.fit({'query': X1, 'doc': X2}, Y, 2,callbacks=[c1])

test = 0
