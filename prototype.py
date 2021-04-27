import utils
import numpy as np

class Prototype():
    def __init__(self, imageEncodeFunc, seed=1729, k=None, label=None):
        self.imageEncodeFunc = imageEncodeFunc
        self.seed = seed
        self.k = k
        self.label = label
        self.images = []
        self.STDImages = []
        self.vectors = []
        self.classVector = []

    def addLabel(self, label):
        self.label = label

    def changeLabel(self, label):
        self.addLabel(label)

    def addImage(self, filename):
        ''' TODO: Update self.images, self.STDimages, self.vectors '''
        pass

    def addImages(self, filenames):
        for filename in filenames:
            addImage(filename)

    def addSTDImage(self, image):
        ''' TODO: Add standardized image to STD images '''
        pass

    def addVector(self, image):
        '''TODO: Do something with the vector (add to vectors + maybe use it to find average?) '''
        image_vector = self.imageEncodeFunc(image)

    def getImages(self):
        return self.images

    def getSTDImages(self):
        return self.STDImages

    def getVectors(self):
        return self.vectors
    
    def getImages(self, k=None):
        if k is None:
            k = self.k
        ''' TODO: Extract k images and change class vectors '''
