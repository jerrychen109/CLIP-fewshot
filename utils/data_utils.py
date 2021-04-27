import os
import pickle
import torch

def load_cifar10_batch(file):
    """ Unpickles a CIFAR-10 batch file. https://www.cs.toronto.edu/~kriz/cifar.html

    Inputs:
    - file: path to the batch file

    Returns:
    - tuple of (data, labels):
        - data: a PyTorch tensor of flattened images of shape (N, D) where N is the number of images
                and D = W * H * 3
        - labels: a PyTorch tensor of shape (N,) containing N labels in the range 0-9, such that
                labels[i] corresponds to data[i]
    """
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return torch.tensor(d[b'data']), torch.tensor(d[b'labels'])

def load_cifar10(batch_paths):
    """ Loads a list of CIFAR-10 batches.

    Inputs:
    - batch_paths: a list of paths to the batches to load

    Returns:
    - tuple of (data, labels):
        - data: a PyTorch tensor of flattened images of shape (N, D) where N is the total number of
                images across all batches
        - labels: a PyTorch tensor of N labels in the range 0-9
    """
    batches = [load_cifar10_batch(path) for path in batch_paths]
    data = torch.cat([batch[0] for batch in batches])
    labels = torch.cat([batch[1] for batch in batches])
    return data, labels