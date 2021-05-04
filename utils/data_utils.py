import os
import pickle
import torch
import random

def load_cifar10_batch(file):
    """ Unpickles a CIFAR-10 batch file. https://www.cs.toronto.edu/~kriz/cifar.html

    Inputs:
    - file: path to the batch file

    Returns:
    - tuple of (data, labels):
        - data: a PyTorch tensor of flattened images of shape (N, D, W, H)
        - labels: a PyTorch tensor of shape (N,) containing N labels in the range 0-9, such that
                labels[i] corresponds to data[i]
    """
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')

    data = torch.tensor(d[b'data']).float() / 255
    data = data.view(data.shape[0], 3, 32, 32)  # (N, W * H * 3)
    labels = torch.tensor(d[b'labels'])
    return data, labels


def load_cifar10(batch_paths):
    """ Loads a list of CIFAR-10 batches.

    Inputs:
    - batch_paths: a list of paths to the batches to load

    Returns:
    - tuple of (data, labels):
        - data: a PyTorch tensor of flattened images of shape (N, D, W, H) where N is the total number of
                images across all batches
        - labels: a PyTorch tensor of N labels in the range 0-9
    """
    batches = [load_cifar10_batch(path) for path in batch_paths]
    data = torch.cat([batch[0] for batch in batches])
    labels = torch.cat([batch[1] for batch in batches])
    return data, labels

def sample_classes(data, labels, per_class = 100):
    zipped = list(zip(data, labels))
    class_dict = {}
    for i in range(10):
        class_dict[i] = torch.stack(random.sample([x[0] for x in zipped if x[1] == i], per_class))
    return class_dict

