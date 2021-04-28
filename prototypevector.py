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
        newProto = Prototype(self.imageEncodeFunc, k=self.k)
        newProto.addLabel(label)
        self.labels.append(label)
        newProto.addImages(filenames)
        self.allImages.append(newProto.getImages())
        for image in newProto.getImages():
            newProto.addSTDImage(image)
        self.allSTDImages.append(newProto.getSTDImages())
        self.allVectors.append(newProto.getVectors())
        newProto.setClassVectors(self, k=self.k)
        self.classVectors.append(newProto.getClassVector())
        self.labelsToPrototypes[label] = newProto

    def addPrototypes(self, labels, allFilenames):
        for label, filenames in zip(labels, allFilenames):
            self.addPrototype(label, filenames=filenames)
            
    def classify(self, images):
        '''TODO: classify list of images'''
        pass
