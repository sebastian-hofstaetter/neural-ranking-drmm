from keras.utils import plot_model
from keras_model import build_keras_model


model = build_keras_model()

plot_model(model, to_file='model.png',show_shapes=True)