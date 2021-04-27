import utils
import numpy as np
import prototype


class PrototypeVector():
    def __init__(self, imageEncodeFunc, k=None):
        self.imageEncodeFunc = imageEncodeFunc
        self.labels = []
        self.labelsToPrototypes = {}
        self.allImages = []
        self.allSTDImages = []
        self.allVectors = []
        self.classVectors = []
        self.k = k

    def addPrototype(self, label, filenames):
        '''TODO: create a new class i.e. append something to labels, allImages, allSTDImages, allVectors, classVectors'''
        # newProto = Prototype(imageEncodeFunc, k=self.k)
        # newProto.addImages(filenames)
        # self.labels.append(newProto.getImages())
        pass

    def addPrototypes(self, labels, allFilenames):
        for label, filenames in zip(labels, allFilenames):
            self.addPrototype(label, filenames=filenames)
            
    def classify(self, images):
        '''TODO: classify list of images'''
        pass
