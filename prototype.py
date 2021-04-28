import utils
import numpy as np
import torch
from utils import *

class Prototype():
    def __init__(self, imageEncodeFunc, seed=1729, k=None, label=None):
        self.imageEncodeFunc = imageEncodeFunc
        self.seed = seed
        self.k = k
        self.label = label
        self.images = []
        self.STDImages = None # tensor
        self.vectors = None # non-normalized, tensor
        self.classVectors = None # normalized k vectors, numpy array
        self.classVector = None # average of k vectors, numpy array

    def addLabel(self, label):
        self.label = label

    def changeLabel(self, label):
        self.addLabel(label)
        
    def changeK(self, k):
        self.k = k
        
    def changeSeed(self, seed):
        self.seed = seed

    def addImage(self, image):
        # Standardizes and encodes image, updates data structures
        self.images.append(image)
        image = [image]
        STDImage = standardize(image)
        self.addSTDImage(STDImage)
        image_vector = self.imageEncodeFunc(STDImage).float()
        self.addVector(image_vector)

    def addImages(self, images):
        self.images.append(images)

    def addSTDImage(self, STDImage):
        if self.STDImages is None:
            self.STDImages = STDImage
        else:
            self.STDImages = torch.cat((self.STDImages, STDImage),0)

    def addVector(self, image_vector):
        if self.vectors is None:
            self.vectors = image_vector
        else:
            self.vectors = torch.cat((self.vectors, image_vector),0)

    def getImages(self):
        return self.images

    def getSTDImages(self):
        return self.STDImages

    def getVectors(self):
        return self.vectors
    
    def getClassVectors(self):
        return self.classVectors
    
    def setClassVectors(self, k=None):
        # Normalizes random k vectors and sets classVectors
        if k is None:
            k = self.k
        random.seed(self.seed)
        indices = random.sample(range(len(images)), k)
        indices = torch.tensor(indices)
        self.classVectors = normalize(self.vectors[indices]).numpy()
    
    def getClassVector(self):
        # Calculates and returns classVector
        self.classVector = self.classVectors.mean(axis=0)
        return self.classVector
        
        
        
