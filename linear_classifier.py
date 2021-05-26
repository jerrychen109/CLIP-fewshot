import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import notebook

from utils.image_utils import imagesToVector

def linear_classifier(train_data, train_labels, num_classes, epochs = 10, lr = 1e-4, device=None, print_every = 200):
    """ Trains a simple linear classifier on the given training data.
    Assumed for use in a few-shot context and does not batch the data.

    Inputs:
    - train_data: an (N, F) tensor where N is the number of examples and F is the feature size
    - train_labels: an (N, 1) tensor containing the class labels for each training example
    - num_classes: the number of 
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = nn.Linear(train_data.shape[1], num_classes).to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_labels);
        loss.backward()
        optimizer.step()

        if print_every > 0 and epoch % print_every == 0:
            print ("epoch {}: loss = {}".format(epoch, loss.item() / len(train_data)))

    return model


def linear_classifier_cifar10(train_dataset, clip_model, epochs = 10, lr = 1e-3, device=None, print_every = 200, batch_size=64):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = nn.Linear(512, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=2)

    try:
        for epoch in notebook.tqdm(range(epochs), desc="epoch"):
            prog = notebook.tqdm(total = len(train_dataloader) * batch_size, desc="train samples")
            for images, labels in train_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                _, imageVectors = imagesToVector(images, clip_model.encode_image, device=device)
                # print(imageVectors.shape)

                optimizer.zero_grad()
                scores = model(imageVectors)
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

                prog.update(batch_size)

            print("epoch {}: loss = {}".format(epoch, loss.item() / batch_size))
    except:
        print("stopping early")

    return model