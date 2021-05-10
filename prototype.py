import utils
import numpy as np
import torch
from utils.text_utils import *
from utils.image_utils import standardize, encodeImageWithFunc, normalize, getImagesFromFiles


class Prototype():
    def __init__(self, imageEncodeFunc, device, label=None, k=None, seed=1729):
        self.imageEncodeFunc = imageEncodeFunc
        self.device = device
        self.label = label
        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.images = None
        self.STDImages = None  # tensor
        self.vectors = None  # non-normalized vectors, tensor
        self.norm_vectors = None # normalized vectors, tensor
        self.classVector = None  # average of k vectors, tensor

    def addLabel(self, label):
        self.label = label

    def changeLabel(self, label):
        self.addLabel(label)

    def changeK(self, k):
        self.k = k

    def changeSeed(self, seed):
        self.seed = seed
        
    def calcClassVector(self, k=None):
        if k is None:
            k = self.k
        return torch.tensor(self.rng.choice(self.norm_vectors.cpu().numpy(), k, replace=False).mean(axis=0), device=self.device)

    #     # Normalizes random k vectors and sets classVectors
    #     if k is None:
    #         k = self.k
    #     indices = self.rng(range(len(images)), k)
    #     indices = torch.tensor(indices)
    #     self.classVectors = normalize(self.vectors[indices]).numpy()

    # def addImage(self, image):
    #     # Standardizes and encodes image, updates data structures
    #     self.images.append(image)
    #     image = [image]
    #     STDImage = standardize(image)
    #     self.addSTDImage(STDImage)
    #     image_vector = self.imageEncodeFunc(STDImage).float()
    #     self.addVector(image_vector)
    def addImagesWithFilenames(self, startDir, filenames):
        images = getImagesFromFiles(startDir, filenames)
        self.addImages(images)

    def addImages(self, images, image_mean, image_std):
        if self.images is None:
            self.images = torch.tensor(np.stack(images), device=self.device)
        else:
            self.images = torch.tensor(np.stack(self.images, images), device=self.device)
        self.STDImages = standardize(self.images, self.device)
        self.vectors = encodeImageWithFunc(
            self.imageEncodeFunc, self.STDImages)
        self.norm_vectors = normalize(self.vectors)
        self.classVector = self.calcClassVector()
        # self.images.append(images)

    # def addSTDImage(self, STDImage):
    #     if self.STDImages is None:
    #         self.STDImages = STDImage
    #     else:
    #         self.STDImages = torch.cat((self.STDImages, STDImage), 0)

    # def addVector(self, image_vector):
    #     if self.vectors is None:
    #         self.vectors = image_vector
    #     else:
    #         self.vectors = torch.cat((self.vectors, image_vector), 0)

    def getImages(self):
        return self.images

    def getSTDImages(self):
        return self.STDImages

    def getVectors(self):
        return self.vectors

    def getNormVectors(self):
        return self.norm_vectors

    def getClassVector(self, k=None):
        # Calculates and returns classVector
        if k is None:
            k = self.k
        return self.calcClassVector(k)
