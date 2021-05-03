import utils
import numpy as np
import torch
from prototype import Prototype
from utils.image_utils import *
from utils.text_utils import *

class PrototypeVector():
    def __init__(self, imageEncodeFunc, device, k=None, seed=1729):
        self.imageEncodeFunc = imageEncodeFunc
        self.device = device
        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.labelsToPrototypes = {}
        self.allImages = []
        self.allSTDImages = []
        self.allVectors = []
        self.allNormVectors = []
        self.allClassVectors = []

    def addPrototypeWithFilenames(self, startdir, filenames, label):
        newPrototype = Prototype(self.imageEncodeFunc,
                                 self.device, label, self.k, self.seed)
        newPrototype.addImagesWithFilenames(startdir, filenames)
        self.addInfo(newPrototype, label)

    def addPrototypesWithFilenames(self, allStartdirs, allFilenames, labels):
        for startdir, filenames, label in zip(allStartdirs, allFilenames, labels):
            self.addPrototypeWithFilenames(startdir, filenames, label)
            
    def addPrototype(self, images, label):
        newPrototype = Prototype(self.imageEncodeFunc,
                                 self.device, label, self.k, self.seed)
        newPrototype.addImages(images)
        self.addInfo(newPrototype, label)

    def addInfo(self, newPrototype, label):
        self.allImages.append(newPrototype.getImages().cpu().numpy())
        self.allSTDImages.append(newPrototype.getSTDImages().cpu().numpy())
        self.allVectors.append(newPrototype.getVectors().cpu().numpy())
        self.allNormVectors.append(newPrototype.getNormVectors().cpu().numpy())
        self.allClassVectors.append(newPrototype.getClassVector(self.k).cpu().numpy())
        self.labelsToPrototypes[label] = newPrototype

    def addPrototypes(self, setImages, labels):
        for images, label in zip(setImages, labels):
            self.addPrototype(images, label)
            
    def getLabelsToPrototypes(self):
        return self.labelsToPrototypes

    def getClassVectors(self, k=None):
        if k is None:
            return self.allClassVectors
        return [key.getClassVector() for key in self.labelsToPrototypes.keys()]

    def classify(self, similarityFunc, imageVector, k=None):
        if k is None:
            k = self.k
        tuples = []
        for key in self.labelsToPrototypes.keys():
            similarity = similarityFunc(self.labelsToPrototypes[key].getClassVector(k), imageVector)
            tuples.append((similarity, key))
        tuples.sort(reverse=True)
        return tuples[0][1], tuples
                
            
            
