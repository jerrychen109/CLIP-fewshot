import utils
import numpy as np
import torch
from prototype import Prototype
from utils.image_utils import *
from utils.text_utils import *

class PrototypeVector():
    def __init__(self, imageEncodeFunc, device, image_mean, image_std, k=None, seed=1729):
        self.imageEncodeFunc = imageEncodeFunc
        self.device = device
        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.labelsToPrototypes = {}
#         self.allImages = []
#         self.allSTDImages = []
        self.allVectors = []
        self.allNormVectors = []
        self.allClassVectors = {}
        self.allKVectors = {}
        self.image_mean = image_mean
        self.image_std = image_std

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
        newPrototype.addImages(images, self.image_mean, self.image_std)
        self.addInfo(newPrototype, label)

    def addInfo(self, newPrototype, label):
#         self.allImages.append(newPrototype.getImages().cpu().numpy())
#         self.allSTDImages.append(newPrototype.getSTDImages().cpu().numpy())
        self.allVectors.append(newPrototype.getVectors().cpu().numpy())
        self.allNormVectors.append(newPrototype.getNormVectors().cpu().numpy())
        #self.allClassVectors.append(newPrototype.getClassVector(self.k).cpu().numpy())
        self.labelsToPrototypes[label] = newPrototype

    def addPrototypes(self, setImages, labels):
        for images, label in zip(setImages, labels):
            self.addPrototype(images, label)

    def addPrototypesFromDict(self, imageDict):
        for c in imageDict:
            self.addPrototype(imageDict[c], c)
            
    def getLabelsToPrototypes(self):
        return self.labelsToPrototypes

    def getClassVectors(self, k=None):
        if k is None:
            k = self.k
        self.allClassVectors[k] = {}
        self.allKVectors[k] = {}
        for key, value in self.labelsToPrototypes.items():
            self.allKVectors[k][key], self.allClassVectors[k][key] = value.getKClassVectors(k)
        return self.allKVectors[k], self.allClassVectors[k]

#     def classify(self, similarityFunc, imageVector, k=None):
#         if k is None:
#             k = self.k
#         tuples = []
#         for key, vec in self.getClassVectors(k)[1]:
#             similarity = similarityFunc(vec, imageVector)
#             tuples.append((similarity, key))
#         tuples.sort(reverse=True)
#         return tuples[0][1], tuples
    
    def classifyImagesWithClassVector(self, similarityFunc, imageVectors, k=None, recalc=False):
        if k is None:
            k = self.k
    
        # Add new {k: dict} if doesn't already exist or replace old if recalculating
        if k not in self.allClassVectors or recalc == True:
            self.getClassVectors(k)
                
        tupleList = []
        for imageVector in imageVectors:
            maxsim = 0.0
            maxlabel = ""
            for label, classvec in self.allClassVectors[k].items():
                similarity = similarityFunc(classvec, imageVector)
                if similarity > maxsim:
                    maxsim = similarity
                    maxlabel = label
            tupleList.append((maxlabel, maxsim))
        return tupleList
    
    def distancesWithKVectors(self, similarityFunc, imageVectors, k=None, recalc=False):
        if k is None:
            k = self.k
    
        # Add new {k: dict} if doesn't already exist or replace old if recalculating
        if k not in self.allClassVectors or recalc == True:
            self.getClassVectors(k)
                
        tupleList = []
        for imageVector in imageVectors:
            simlabels = []
            for label, kvecs in self.allKVectors[k].items():
                for kvec in kvecs:
                    similarity = similarityFunc(kvec, imageVector)
                    simlabels.append((sim, label))
            simlabels = sorted(simlabels, reverse=True)
            tupleList.append(simlabels)
        return tupleList
                
            
            
