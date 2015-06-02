#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np


def unpickle(path):
    """Load data and labels from cifar-10 and cifar-100 data file.
    @return
        data: 10000x3072 numpy array of unint8s, 3x32x32 RGB pixels.
        lables: a list of 10000 numbers in the range 0-9 or 0-99.
    """
    import cPickle
    with open(path, 'rb') as f:
        d = cPickle.load(f)
        return d['data'].astype('float'), np.array(d['labels'])


def load_CIFAR10(root):
    """ Load all data of cifar-10
    :param root: root directory path
    :return:
    """
    ds = [unpickle(p) for p in glob.glob(os.path.join(root, 'data_batch_*'))]
    X_train = np.concatenate([X for X, _ in ds], axis=0)
    y_train = np.concatenate([y for _, y in ds], axis=0)
    X_test, y_test = unpickle(os.path.join(root, 'test_batch'))
    return X_train, y_train, X_test, y_test


def read_bin(path):
    """ Read from binary format """
    import struct
    with open(path, 'rb') as f:
        def read_one():
            t = struct.unpack('b', f.read(1))[0]
            m = np.fromstring(f.read(3072), dtype=np.uint8).reshape(3, 32, 32)
            return t, m
        return [read_one() for i in xrange(10000)]


if __name__ == '__main__':
    load_CIFAR10('data/cifar-10-batches-py', verboase=True)
