import utils
import numpy as np
import torch
from torch.utils.data import DataLoader
from prototype import Prototype
from utils.image_utils import *
from utils.text_utils import *
from tqdm import notebook

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
        self.allTextVectors = {}
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
        newPrototype.addImages(images)
        self.addInfo(newPrototype, label)

    def addInfo(self, newPrototype, label):
#         self.allImages.append(newPrototype.getImages().cpu().numpy())
#         self.allSTDImages.append(newPrototype.getSTDImages().cpu().numpy())
        self.allVectors.append(newPrototype.getVectors().cpu().numpy())
        self.allNormVectors.append(newPrototype.getNormVectors().cpu().numpy())
        #self.allClassVectors.append(newPrototype.calcClassVector(self.k).cpu().numpy())
        self.labelsToPrototypes[label] = newPrototype

    def addPrototypes(self, setImages, labels):
        for images, label in zip(setImages, labels):
            self.addPrototype(images, label)
            
    def addTextVectors(self, textDict):
        self.allTextVectors = textDict

    def addPrototypesFromDict(self, imageDict):
        for c in imageDict:
            self.addPrototype(imageDict[c], c)
            
    def getLabelsToPrototypes(self):
        return self.labelsToPrototypes

    def calcClassVectors(self, k=None):
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
    
    def classifyImagesWithClassVector(self, similarityFunc, encoded_images,
        k=None, recalc=False, bimodal=False, biweight=0.5, batch_size=512):
        """ Classifies the given image vectors using the closest class template based on the
        provided similarity function.

        Inputs:
        - similarityFunc: the similarity function for comparing two vectors with the same dims
        - dataset: Dataset of images to classify
        - k: the number of training images to use to generate class templates
        - recalc: whether or not to generate new class templates
        - bidomal: whether or not to use text vector weighted 50/50

        Returns:
        - a list of (label, score) tuples of the same length as imageVectors
        """
        if k is None:
            k = self.k
    
        # Add new {k: dict} if doesn't already exist or replace old if recalculating
        if k not in self.allClassVectors or recalc == True:
            self.calcClassVectors(k)
        # trueLabels = []

        # dataloader = DataLoader(dataset, batch_size = batch_size)
        # for images, labels in notebook.tqdm(dataloader, desc="eval"):
            # _, imageVectors = imagesToVector(images, self.imageEncodeFunc, device=self.device)
            # print(imageVectors.device)
            # trueLabels.extend(list(labels))
        classVecs = [tup[1] for tup in sorted(self.allClassVectors[k].items())]
        if bimodal:
            classVecs = torch.stack(classVecs)*(1-biweight) + torch.stack(list(self.allTextVectors.values()))*biweight
        else:
            classVecs = torch.stack(classVecs)
        similarity = similarityFunc(classVecs, encoded_images)
        return np.argmax(similarity, axis=0), np.amax(similarity, axis=0)
    
    def distancesWithKVectors(self, similarityFunc, imageVectors, k=None, recalc=False):
        if k is None:
            k = self.k
    
        # Add new {k: dict} if doesn't already exist or replace old if recalculating
        if k not in self.allClassVectors or recalc == True:
            self.calcClassVectors(k)
                
        tupleList = []
        for imageVector in imageVectors:
            simlabels = []
            for label, kvecs in self.allKVectors[k].items():
                sims = similarityFunc(kvecs, imageVector)
                sims = [(sim, label) for sim in sims]
                simlabels += sims
            simlabels = sorted(simlabels, reverse=True)
            tupleList.append(simlabels)
        return tupleList
    
    def getRandomNormVectors(self):
        images = []
        for proto in self.labelsToPrototypes.values():
            images.append(proto.getRandomNormVector())
        return images
                
            
            
