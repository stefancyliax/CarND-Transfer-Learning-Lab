import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Activation
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    # Input shape from size of X_train
    input_shape = X_train.shape[1:]
    print('Input shape: ', input_shape)
    nb_classes = len(np.unique(y_train))
    print('output shape: ', nb_classes)

    model = Sequential()
    model.add(Dense(nb_classes, input_shape=input_shape))
    model.add(Activation('softmax'))
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])

    # TODO: train your model here

    model.fit(X_train, y_train, batch_size=256, nb_epoch=50, validation_data=(X_val, y_val))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
