import utils
import numpy as np
import torch
from utils.text_utils import *
from utils.image_utils import standardize, encodeImageWithFunc, normalize, getImagesFromFiles, imagesToVector

class Prototype():
    def __init__(self, imageEncodeFunc, device, label=None, k=None, seed=1729):
        self.imageEncodeFunc = imageEncodeFunc
        self.device = device
        self.label = label
        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed)
#         self.images = None
#         self.STDImages = None  # tensor
        self.vectors = None  # non-normalized vectors, tensor
        self.norm_vectors = None # normalized vectors, tensor
        self.classVector = None  # average of k vectors, tensor
        self.kVectors = None # the k vectors that make up self.classVector, tensor

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
        self.kVectors = torch.tensor(self.rng.choice(self.norm_vectors.cpu().numpy(), k, replace=False), device=self.device)
        self.classVector = self.kVectors.mean(axis=0)
        return self.kVectors, self.classVector

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

    def addImages(self, images):
        """ Adds a batch of images, shape (N, 3, H, W), to the Prototype.
        """
        self.vectors, self.norm_vectors = imagesToVector(images, self.imageEncodeFunc, device=self.device)
        self.kVectors, self.classVector = self.calcClassVector()

    def getVectors(self):
        return self.vectors

    def getNormVectors(self):
        return self.norm_vectors
    
    def getRandomNormVector(self):
        return self.rng.choice(self.norm_vectors)
    
    def getKVectors(self, k=None):
        # Calculates and returns kVectors
        if k is None:
            k = self.k
        return self.calcClassVector(k)[0]

    def getClassVector(self, k=None):
        # Calculates and returns classVector
        if k is None:
            k = self.k
        return self.calcClassVector(k)[1]
    
    def getKClassVectors(self, k=None):
        # Calculates and returns classVector
        if k is None:
            k = self.k
        return self.calcClassVector(k)

