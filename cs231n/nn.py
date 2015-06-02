#!/usr/bin/env python
# coding: utf-8

import numpy as np


class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an exmaple. Y is 1-dimension of size N """
        self.X_train = X
        self.y_train = y
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def compute_norm1_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i, j] = np.sum(np.abs(self.X_train[j, :] - X[i, :]))
        return dists

    def compute_norm1_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)
        return dists

    def compute_norm2_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i, j] = np.sum((self.X_train[j, :] - X[i, :])**2)
        return dists

    def compute_norm2_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        train_2 = np.sum(self.X_train**2, axis=1).T
        for i in xrange(num_test):
            test_2 = np.tile(np.sum((X[i, :])**2), [1, num_train])
            test_train = X[i, :].dot(self.X_train.T)
            dists[i, :] = train_2 + test_2 - 2 * test_train
        return dists

    def compute_norm2_distances_no_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        train_2 = np.tile(np.sum(self.X_train**2, axis=1), [num_test, 1])
        test_2 = np.tile(np.sum(X**2, axis=1), [num_train, 1]).T
        test_train = X.dot(self.X_train.T)
        dists = train_2 + test_2 - 2 * test_train
        return dists

    def predict(self, X, k=1, norm2=True):
        """ X is N x D where each row is an exmaple we wish to predict """
        if norm2:
            dists = self.compute_norm2_distances_no_loop(X)
        else:
            dists = self.compute_norm1_distances_one_loop(X)
        return self.predict_labels(dists, k=k)

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=np.dtype)
        for i in xrange(num_test):
            closest_idx = np.argpartition(dists[i, :], kth=k)[:k]
            closest_y = self.y_train[closest_idx]
            # count the frequency of those closest labels
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)
        return y_pred
