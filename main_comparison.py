import random
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import keras.datasets as datasets
import tensorflow as tf
import pickle
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy


def randomize_seed():
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def tensorflow_shutup():
    """
    https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
    Make Tensorflow less verbose
    """
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyPackageRequirements
        import tensorflow as tf
        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        deprecation.deprecated = deprecated

    except ImportError:
        pass


def threshold_01(x, threshold=0.5):
    return np.where(x >= threshold, 1, 0)


def plot_contour(x_min, x_max, y_min, y_max, model, delta=1e-2):
    xx = np.arange(x_min, x_max, delta)
    delta = ((x_max - x_min)*delta) / (y_max - y_min)
    yy = np.arange(y_min, y_max, delta)
    X, Y = np.meshgrid(xx, yy)
    XY = np.c_[X.ravel(), Y.ravel()]
    Z = model.predict(XY).reshape(X.shape)
    Z = threshold_01(Z)
    plt.contourf(X, Y, Z, cmap='viridis', alpha=0.2)
    return X, Y, Z

tensorflow_shutup()

def print_help():
    print("Usage: python3 main_comparison.py [training_case]")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    case = sys.argv[1]
    if len(sys.argv) == 3:
        iters = int(sys.argv[2])
    else:
        iters = 1
    if case == 'xor':
        def XOR(csv_file, i):

            print('### XOR ###')

            # Create the XOR dataset
            X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(-1, 2).astype(np.float64)
            y = np.array([0, 1, 1, 0]).reshape(-1, 1).astype(np.float64)

            # Define hyper-parameters
            layers = [2, 2, 1]

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.1,
                decay_steps=50,
                decay_rate=0.2,
                staircase=True)
            epochs = 300

            randomize_seed()

            # Create the model
            nn = Sequential([
                Dense(2, input_shape=(layers[0],), activation='sigmoid'),
                Dense(1, activation='sigmoid'),
            ])

            nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                       loss='mean_squared_error', metrics=['accuracy'])

            # Train the model
            history = nn.fit(X, y, epochs=epochs, verbose=0, validation_data=(X, y))
            history = history.history
            # Predict the data
            yp = threshold_01(nn.predict(X))
            print(f'Accuracy: {float(tf.reduce_mean(binary_accuracy(y, yp)))}')

            # Create a loss and accuracy plot
            if csv_file is None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.set_title("Xor Loss")
                ax1.plot(range(len(history['loss'])), history['loss'], label='Training Loss')
                ax1.plot(range(len(history['val_loss'])), history['val_loss'], label='Validation Loss')
                ax1.legend()

                ax2.set_title("Xor Accuracy")
                ax2.plot(range(len(history['accuracy'])), history['accuracy'], label='Training Accuracy')
                ax2.plot(range(len(history['val_accuracy'])), history['val_accuracy'], label='Validation Accuracy')
                ax2.legend()
                plt.show()

                plt.title('XOR Decision Boundary')
                _x, _y, _z = plot_contour(-.5, 1.5, -.5, 1.5, nn, delta=1e-2)
                pickle.dump((_x, _y, _z), open('data/nn_xor_boundary.pkl', 'wb'))
                plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
                plt.savefig('img/nn_xor_decision_boundary.png')
                plt.show()
            else:
                with open(csv_file, 'a') as f:
                    for e in range(len(history["loss"])):
                        f.write(f'{i},train,{e},{history["loss"][e]},{history["accuracy"][e]}\n')
                    for e in range(len(history["val_loss"])):
                        f.write(f'{i},val,{e},{history["val_loss"][e]},{history["val_accuracy"][e]}\n')

        if iters <= 1:
            XOR(None, 0)
        else:
            for i in range(iters):
                XOR('data/nn_xor.csv', i)

    elif case == 'and':
        def AND(csv_file, i):

            print('### AND ###')

            # Create the XOR dataset
            X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(-1, 2).astype(np.float64)
            y = np.array([0, 0, 0, 1]).reshape(-1, 1).astype(np.float64)

            # Define hyper-parameters
            layers = [2, 1]
            lr = 0.1
            epochs = 150

            randomize_seed()

            # Create the model
            nn = Sequential([
                Dense(1, input_shape=(layers[0],), activation='sigmoid'),
            ])

            nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                       loss='mean_squared_error', metrics=['accuracy'])

            # Train the model
            history = nn.fit(X, y, epochs=epochs, verbose=1, validation_data=(X, y))
            history = history.history
            # Predict the data
            yp = threshold_01(nn.predict(X))

            print(f'Accuracy: {float(tf.reduce_mean(binary_accuracy(y, yp)))}')

            # Create a loss and accuracy plot
            if csv_file is None:
                print('Plotting Loss and Accuracy')
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.set_title("And Loss")
                ax1.plot(range(len(history['loss'])), history['loss'], label='Training Loss')
                ax1.plot(range(len(history['val_loss'])), history['val_loss'], label='Validation Loss')
                ax1.legend()

                ax2.set_title("And Accuracy")
                ax2.plot(range(len(history['accuracy'])), history['accuracy'], label='Training Accuracy')
                ax2.plot(range(len(history['val_accuracy'])), history['val_accuracy'], label='Validation Accuracy')
                ax2.legend()
                plt.show()
                print('Plotting Decision Boundary')
                plt.title('AND Decision Boundary')
                _x, _y, _z = plot_contour(-.5, 1.5, -.5, 1.5, nn, delta=1e-2)
                pickle.dump((_x, _y, _z), open('data/nn_and_boundary.pkl', 'wb'))
                plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
                plt.savefig('img/nn_and_decision_boundary.png')
                plt.show()
            else:
                with open(csv_file, 'a') as f:
                    for e in range(len(history["loss"])):
                        f.write(f'{i},train,{e},{history["loss"][e]},{history["accuracy"][e]}\n')
                    for e in range(len(history["val_loss"])):
                        f.write(f'{i},val,{e},{history["val_loss"][e]},{history["val_accuracy"][e]}\n')


        if iters <= 1:
            AND(None, 0)
        else:
            for i in range(iters):
                AND('data/nn_and.csv', i)

    elif case == 'synth':
        # Load in the dataset
        # https://www.stats.ox.ac.uk/pub/PRNN/
        train_fn = os.path.join(os.path.dirname(__file__), 'dataset', 'synth.tr')
        test_fn = os.path.join(os.path.dirname(__file__), 'dataset', 'synth.te')


        def load_data(fn):
            data = np.loadtxt(fn)
            X = data[:, :-1]
            y = data[:, -1]
            return X, y


        X_train, y_train = load_data(train_fn)
        X_val, y_val = load_data(test_fn)

        def SYNTH(csv_file, i):
            print('### Synthetic ###')

            # Define hyper-parameters
            layers = [X_train.shape[1], 5, 5, 1]
            lr = 0.03
            epochs = 300
            randomize_seed()

            # Create the model
            nn = Sequential([
                Dense(layers[1], input_shape=(layers[0],), activation='sigmoid'),
                Dense(layers[2], activation='sigmoid'),
                Dense(layers[3], activation='sigmoid')
            ])

            nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                       loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model
            history = nn.fit(X_train, y_train, epochs=epochs, verbose=0, validation_data=(X_val, y_val))
            history = history.history
            # Predict the data
            yp = threshold_01(nn.predict(X_val))

            print(f'Accuracy: {float(tf.reduce_mean(binary_accuracy(y_val, yp)))}')


            if csv_file is None:
                # Create a loss and accuracy plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.set_title("Synth Loss")
                ax1.plot(range(len(history['loss'])), history['loss'], label='Training Loss')
                ax1.plot(range(len(history['val_loss'])), history['val_loss'], label='Validation Loss')
                ax1.legend()

                ax2.set_title("Synth Accuracy")
                ax2.plot(range(len(history['accuracy'])), history['accuracy'], label='Training Accuracy')
                ax2.plot(range(len(history['val_accuracy'])), history['val_accuracy'], label='Validation Accuracy')
                ax2.legend()
                plt.show()

                print('Plotting Decision Boundary')
                plt.title('SYNTH Decision Boundary')
                _x, _y, _z = plot_contour(-1.5, 1.3, -.5, 1.5, nn, delta=1e-2)
                pickle.dump((_x, _y, _z), open('data/nn_synth_boundary.pkl', 'wb'))
                plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, s=50)
                plt.savefig('img/nn_synth_decision_boundary.png')
                plt.show()
            else:
                with open(csv_file, 'a') as f:
                    for e in range(len(history["loss"])):
                        f.write(f'{i},train,{e},{history["loss"][e]},{history["accuracy"][e]}\n')
                    for e in range(len(history["val_loss"])):
                        f.write(f'{i},val,{e},{history["val_loss"][e]},{history["val_accuracy"][e]}\n')


        if iters <= 1:
            SYNTH(None, 0)
        else:
            for i in range(iters):
                SYNTH('data/nn_synth.csv', i)
    else:
        print_help()
        sys.exit()
