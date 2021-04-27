import utils
import numpy as np
import prototype


class PrototypeVector():
    def __init__(self, imageEncodeFunc):
        self.imageEncodeFunc = imageEncodeFunc
        self.labels = []
        self.allImages = []
        self.allSTDImages = []
        self.allVectors = []
        self.classVectors = []

    def addClass(self, label, filenames):
        '''TODO: create a new class i.e. append something to labels, allImages, allSTDImages, allVectors, classVectors'''
        pass

    def addClasses(self, labels, allFilenames):
        for label, filenames in zip(labels, allFilenames):
            addClass(label, filenames)
            
    def classify(self, images):
        '''TODO: classify list of images'''
        pass