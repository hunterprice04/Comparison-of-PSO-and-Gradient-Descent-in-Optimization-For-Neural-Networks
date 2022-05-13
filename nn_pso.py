import numpy as np
import tensorflow as tf
from tqdm import tqdm
from math import ceil
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy


def sigmoid(net):
    return 1.0 / (1.0 + np.exp(-net))


def relu(net):
    return np.maximum(0, net)


def linear(net):
    return net


class Particle:
    def __init__(self, position):
        """
        initializes the particle
        :param position: the weight vector
        """
        # here position needs to a flattened array of all weights in the network
        self.position = position
        self.v = np.zeros_like(position, dtype=np.float64)
        self.best_position = self.position.copy()
        self.best_loss = None

    def update(self, pso, batch):
        """
        updates the particle position and velocity
        :param pso: pso object
        :return: None
        """
        # calculate the new velocity
        self.v = pso.inertia * self.v + \
                 pso.phi_1 * np.random.random(pso.weight_len) * np.subtract(self.best_position, self.position) + \
                 pso.phi_2 * np.random.random(pso.weight_len) * np.subtract(pso.global_best, self.position)

        # check against the maximum velocity
        sum_sqr_v = np.sum(self.v ** 2)
        if sum_sqr_v > pso.max_vel ** 2:
            self.v = (pso.max_vel / sum_sqr_v) * self.v

        # update the position
        self.position = np.add(self.position, self.v)

        # get the loss
        loss = pso.Q(self.position, batch)

        # update particle best position if better
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_position = self.position.copy()

        # update global best position if better
        if loss < pso.global_best_loss:
            pso.global_best_loss = loss
            pso.global_best = self.position.copy()


class NN_PSO:
    def __init__(self, num_particles, inertia, phi_1, phi_2, max_vel, layers, X, y, loss_func, activations=None):
        """
        Initialize the PSO
        :param num_particles:
        :param inertia:
        :param phi_1:
        :param phi_2:
        :param max_vel:
        :param layers:
        :param X:
        :param y:
        """
        # save class variables
        self.num_particles = num_particles
        self.inertia = inertia
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.max_vel = max_vel
        self.layers = layers
        self.X = X
        self.y = y
        self.weight_len = 0
        self.loss_func = loss_func
        self.activations = activations

        for i in range(len(layers) - 1):
            self.weight_len += layers[i] * layers[i + 1]

        self.global_best = np.zeros(self.weight_len, dtype=np.float64)
        self.global_best_loss = None
        self.particles = []
        self.batch_size = None

        # initialize all particles
        for i in range(num_particles):
            w = np.random.randn(self.weight_len)
            particle = Particle(w)
            particle.best_loss = self._calc_loss(particle.position, self.X, self.y)
            if self.global_best_loss is None or self.global_best_loss > particle.best_loss:
                self.global_best_loss = particle.best_loss
                self.global_best[:] = w[:]
            self.particles.append(particle)

        # variables to keep track of loss and accuracy
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def update(self):
        """
        Update all particles
        :return: None
        """
        for b in range(ceil(len(self.X) / self.batch_size)):
            for i in range(self.num_particles):
                self.particles[i].update(self, b)

    def _feed_forward(self, inp, weights):
        """
        :param inp: input data
        :param weights: weights to use
        :return: returns prediction
        """
        for i in range(len(weights)):
            net = inp.dot(weights[i])
            if self.activations is None:
                inp = sigmoid(net)
            else:
                inp = self.activations[i](net)
        return inp

    def Q(self, position, batch):
        """
        Objective function for the network
        :param position: list of weights of a particle
        :return: returns the loss
        """
        X, y = self.get_batch(batch)
        return self._calc_loss(position, X, y)

    def _calc_loss(self, position, X, y):
        """
        Calculates the loss for the given weights, X and y
        :param position: the weights to use
        :param X: X data
        :param y: y data
        :return: returns the loss
        """
        # reshape the given weights
        weights = self._reshape_weights(position)
        # feed the data through the network
        y_pred = self._feed_forward(X, weights)
        # get the overall loss
        loss = float(tf.reduce_mean(self.loss_func(y, y_pred)))
        return loss

    def _calc_acc(self, position, X, y):
        """
        Calculates the accuracy of the network given the weights, X and y
        :param position: the weights of the network
        :param X: X data
        :param y: y data
        :return: returns the accuracy
        """
        # reshape the given weights
        weights = self._reshape_weights(position)
        # feed the data through the network
        y_pred = self._feed_forward(X, weights)
        if y_pred.shape[1] == 1:
            acc = float(tf.reduce_mean(binary_accuracy(y, y_pred)))
        else:
            acc = float(tf.reduce_mean(categorical_accuracy(y, y_pred)))
        return acc

    def _reshape_weights(self, weights):
        """
        takes in a flat array of all the weights and reshapes them into the proper configuration
        :param weights:
        :return:
        """
        # offset for slicing
        i = 0
        w = []

        # iterate through each layer
        for l in range(1, len(self.layers)):
            # grab and calculate needed sizes
            prev_input_size = self.layers[l - 1]
            num_layer_nuerons = self.layers[l]
            offset = prev_input_size * num_layer_nuerons

            # get and save the weights
            w.append(weights[i:i + offset].reshape(prev_input_size, num_layer_nuerons))

            # increment the offset
            i += offset

        return w

    def predict(self, X):
        """
        predict the given data with the best weights found so far
        :param X: data
        :return: prediction
        """
        w = self._reshape_weights(self.global_best)
        return self._feed_forward(X, w)

    def train(self, epochs, val_data: tuple = None, loss_threshold=1e-4, batch_size=None, shuffle=True):
        """
        training loop to abstract away the training process from the user
        :param epochs: number of epochs to train for
        :param val_data: a tuple holding the x and y values for validation
        :param loss_threshold: the threshold for the loss to stop training
        :return: None
        """
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        # if batch size is not given, use the whole dataset
        self.batch_size = self.X.shape[0] if batch_size is None else batch_size

        pbar = tqdm(range(epochs))
        for _ in pbar:

            # shuffle the data if needed
            if shuffle:
                self.shuffle_data()

            # update the weights
            self.update()

            # keep track of training loss
            self.train_loss.append(self._calc_loss(self.global_best, self.X, self.y))
            self.train_acc.append(self._calc_acc(self.global_best, self.X, self.y))
            post_fix = {'train_loss': self.train_loss[-1], 'train_acc': self.train_acc[-1]}
            # keep track of validation loss
            if val_data is not None:
                X_val, y_val = val_data
                self.val_loss.append(self._calc_loss(self.global_best, X_val, y_val))
                self.val_acc.append(self._calc_acc(self.global_best, X_val, y_val))
                post_fix['val_loss'] = self.val_loss[-1]
                post_fix['val_acc'] = self.val_acc[-1]

            pbar.set_postfix(post_fix)
            # break if loss under threshold
            if loss_threshold is not None and self.train_loss[-1] < loss_threshold:
                break


    def shuffle_data(self):
        mask = np.random.permutation(np.arange(len(self.X)))
        self.X = self.X[mask]
        self.y = self.y[mask]


    def get_batch(self, b):
        X = self.X[b * self.batch_size: (b + 1) * self.batch_size]
        y = self.y[b * self.batch_size: (b + 1) * self.batch_size]
        return X, y