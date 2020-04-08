import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow_model_optimization

import neptune


import gin
import gin.config
import gin.tf
from absl import flags, app

flags.DEFINE_multi_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS


BATCH_SIZE = 32
EPOCHS = 10

@gin.configurable()
def input_fn(dataset='mnist'):

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if not hasattr(layer, 'prunable_weights'):
            continue
        #if isinstance(layer, tf.keras.engine.Network):
        #    reset_weights(layer)
        #    continue
        for v in layer.prunable_weights:
            if hasattr(v, 'initializer'):
                v.initializer.run(session=session)

def vis_weights(model, epoch):
    ws = []
    session = K.get_session()
    i = 0

    i2 = 0
    for layer in model.layers:
        if not hasattr(layer, 'prunable_weights'):
            continue
        for v in layer.prunable_weights:
            i += 1
            w = session.run(v)
            s = w.shape
            plt.imshow(w.reshape((-1, s[-1])), cmap='hot', interpolation=None)
            plt.savefig("e{}_l{}.png".format(epoch, i), dpi=200)
        print(layer.pruning_vars)
        for v in [layer.pruning_vars[0][1]]:
            i2 += 1
            w = session.run(v)
            s = w.shape
            plt.clf()
            plt.imshow(w.reshape((-1, s[-1])), cmap='hot', interpolation=None)
            plt.savefig("mask_e{}_l{}.png".format(epoch, i), dpi=200)

            plt.clf()
            print(w.shape)
            conns_per_filter = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
            plt.hist(conns_per_filter)
            plt.savefig("conns_per_filter_e{}_l{}".format(epoch, i), dpi=200)
            if len(w.shape) == 4:
              plt.clf()
              print(w.shape)
              filters_per_filter = np.sum(np.max(w, axis=(0,1)), axis=(0, ))
              plt.hist(filters_per_filter)
              plt.savefig("filters_per_filter_e{}_l{}".format(epoch, i), dpi=200)

              plt.clf()
              print(w.shape)
              c2c_num_weights = np.reshape(np.sum(w, axis=(0,1)), (-1,))
              plt.hist(c2c_num_weights)
              plt.savefig("c2c_num_weights_e{}_l{}".format(epoch, i), dpi=200)

    return ws


def save_weights(model):
    ws = []
    session = K.get_session()
    for layer in model.layers:
        if not hasattr(layer, 'prunable_weights'):
            continue
        for v in layer.prunable_weights:
            ws.append(np.copy(session.run(v)))
    return ws

def load_weights(model, ws):
    session = K.get_session()
    i = 0
    for layer in model.layers:
        if not hasattr(layer, 'prunable_weights'):
            continue
        for v in layer.prunable_weights:
            tf.assign(v, ws[i]).eval(session=session)
            i += 1

l = tf.keras.layers


# Create an instance of the model
num_classes = 10

def get_model(input_shape=(28, 28, 1), pruning_params=None):
    assert pruning_params is not None, "You must specify pruning params."

    pruned_model = tf.keras.Sequential([
        sparsity.prune_low_magnitude(l.Conv2D(32, 5, padding='same', activation='relu'), **pruning_params),
        l.MaxPooling2D((2, 2), (2, 2), padding='same'),
        
        sparsity.prune_low_magnitude(l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
        l.MaxPooling2D((2, 2), (2, 2), padding='same'),
        
        sparsity.prune_low_magnitude(l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
        l.MaxPooling2D((2, 2), (2, 2), padding='same'),
        
        l.Flatten(),
        #    sparsity.prune_low_magnitude(l.Dense(1024, activation='relu'), **pruning_params),
        sparsity.prune_low_magnitude(l.Dense(num_classes, activation='softmax'), **pruning_params)
    ])

    pruned_model.build(input_shape=(None,) + input_shape)

    return pruned_model


class ResetCallback(tf.keras.callbacks.Callback):

    def __init__(self, ws, every_xth_epoch):
        super().__init__()
        self.ws = ws
        self.every_xth_epoch = every_xth_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and epoch % self.every_xth_epoch == 0:
            vis_weights(self.model, epoch)
            load_weights(self.model, self.ws)


def train():

    (x_train, y_train), (x_test, y_test) = input_fn()

    num_train_samples = x_train.shape[0]
    end_step = np.ceil(1.0 * num_train_samples / BATCH_SIZE).astype(np.int32) * EPOCHS
    print('End step: ' + str(end_step))
    print('Train input shape: ', x_train.shape)

    pruning_params = {
          'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.20,
                                                   final_sparsity=0.95,
                                                   begin_step=500,
                                                   end_step=end_step,
                                                   frequency=100)
    }

    pruned_model = get_model(input_shape=x_train.shape[1:], pruning_params=pruning_params)

    pruned_model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])
    pruned_model.summary()

    initial_ws = save_weights(pruned_model)

    logdir = 'logs'

    # Add a pruning step callback to peg the pruning step to the optimizer's
    # step. Also add a callback to add pruning summaries to tensorboard
    callbacks = [
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
        ResetCallback(initial_ws, 2)
    ]

    pruned_model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

    score = pruned_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])





def main(argv):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  use_neptune = "NEPTUNE_API_TOKEN" in os.environ
  if use_neptune:
    neptune.init(project_qualified_name="csadrian/ltp")
    print(gin.operative_config_str())
    exp = neptune.create_experiment(params={}, name="exp")
    #for tag in opts['tags'].split(','):
    #  neptune.append_tag(tag)

  train()

if __name__ == '__main__':
  app.run(main)