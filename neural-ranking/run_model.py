import keras
import sys
from keras.utils import plot_model
from keras_model import build_keras_model
from load_data import load_data
from loss_function import *
import numpy as np
import os


# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 4:
    print('Needs 3 arguments - 1. run name, 2. train file, 3.test file')
    exit(0)

run_name = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]

#
# build and train model
#

train_data, data_count = load_data(train_file)

model = build_keras_model()
model.summary()
model.compile(loss=rank_hinge_loss, optimizer='adam') #adagrad

#plot_model(model, to_file='model.png', show_shapes=True)
#
#
# create train dataset (1 batch)
#

batch_size = int(data_count/10)*10 #int((data_count/3)*2)

# np histogram
X2 = np.zeros((batch_size, 5, 30), dtype=np.float32)

# topic idf
X1 = np.zeros((batch_size, 5,1), dtype=np.float32)

# empty label array
Y = np.zeros((batch_size,), dtype=np.int32)
Y[::2] = 1

i = 0
for topic in train_data:

    list_of_docs = train_data[topic]

    half = len(list_of_docs)//2
    upper = list_of_docs[:half]
    lower = list_of_docs[half:]

    # get pairs
    #p1 = None
    for t in range(min(len(upper), len(lower))):

        if i == batch_size:
            break

        d_high = upper[t]
        d_low = lower[t]

        # filter out inputs that are not different enough
        #if d_high[1] < d_low[1] + 2:
        #   continue

        #print(str(d_high[1]-d_low[1]))

        # histograms
        for w in range(len(d_high[3])): # 5
            X2[i][w] = d_high[3][w] # np.ones(30,dtype=np.float32)
            X2[i+1][w] = d_low[3][w] # np.zeros(30,dtype=np.float32)

        # idf
        X1[i] = d_high[2] # np.ones((5,1),dtype=np.float32) #
        X1[i+1] = d_low[2] # np.zeros((5,1),dtype=np.float32) #

        i += 2

    if i == batch_size:
        break

# use this to reshape input arrays if not all samples are taken
#i = int(i/10)*10
#print('using '+str(i)+' samples')
#
#print('X1 shape before: ',X1.shape)
#print('X2 shape before: ',X2.shape)
#
#X1 = np.resize(X1,(i,5,1))
#X2 = np.resize(X2,(i,5,30))
#Y = np.resize(Y,(i,))
#
#print('X1 shape after resize : ',X1.shape)
#print('X2 shape after resize: ',X2.shape)
#print('Y shape after resize: ',Y.shape)

c1=keras.callbacks.TensorBoard(log_dir='./tensorboard-logs', histogram_freq=0,
          write_graph=True, write_images=True)


model.fit({'query': X1, 'doc': X2}, Y, batch_size=10, verbose=2, shuffle=False, epochs=500, callbacks=[c1])

if not os.path.exists('models/'):
    os.makedirs('models/')

model.save_weights('models/'+run_name+'.weights')

#
#
# prediction
#
#

train_data, data_count = load_data(test_file)


batch_size = data_count

# np histogram
X2 = np.zeros((batch_size, 5, 30), dtype=np.float32)

# topic idf
X1 = np.zeros((batch_size, 5,1), dtype=np.float32)

i = 0
for topic in train_data:

    for line in train_data[topic]:

        for w in range(len(line[3])):
            X2[i][w] = line[3][w]

        X1[i] = line[2]

        i += 1

predictions = model.predict({'query': X1, 'doc': X2}, batch_size = 20)

if not os.path.exists('result/'):
    os.makedirs('result/')
with open('result/'+run_name+".result", 'w') as outFile:
    i = 0
    for topic in train_data:

        for line in train_data[topic]:

            outFile.write(topic + ' '+line[0]+' '+str(predictions[i][0])+'\n')

            i += 1