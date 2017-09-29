import keras
from keras.utils import plot_model
from keras_model import build_keras_model
from load_data import load_data
from loss_function import *
import numpy as np

train_data = load_data('../data/5-folds/30_histogram_fold_1.train')

model = build_keras_model()
model.summary()
model.compile(loss = rank_hinge_loss, optimizer='adam')

plot_model(model, to_file='model.png', show_shapes=True)

#
# create train dataset (1 batch)
#

batch_size = 50000

# np histogram
X2 = np.zeros((batch_size*2, 5, 30), dtype=np.float32)

# topic idf
X1 = np.zeros((batch_size*2, 5,1), dtype=np.float32)

# empty label array
Y = np.zeros((batch_size*2,), dtype=np.int32)
Y[::2] = 1

test1 = train_data['653'][0]
test2 = train_data['653'][7]

i = 0
for topic in train_data:

    list_of_docs = train_data[topic]

    # get pairs
    p1 = None
    for entry in list_of_docs:

        if i == batch_size:
            break

        if p1 and p1[1] > entry[1]: # compare the scores make sure we are higher now

            # high = p1
            # low = entry

            # histograms
            for t in range(len(p1[3])):
                X2[i][t] = p1[3][t]

            for t in range(len(entry[3])): # should be the same count (same query), but just to clarify
                X2[i+1][t] = entry[3][t]

            # idf
            X1[i] = p1[2]
            X1[i+1] = entry[2]

            i += 2

        p1 = entry

    if i == batch_size:
        break

c1=keras.callbacks.TensorBoard(log_dir='./tensorboard-logs', histogram_freq=0,
          write_graph=True, write_images=True)


model.fit({'query': X1, 'doc': X2}, Y, batch_size=10, epochs=20 ,callbacks=[c1])

test = 0
