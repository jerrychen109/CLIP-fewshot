#Inspired from https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
#TODO: USE GPU + other things from here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes 
# + https://github.com/facebookresearch/faiss/blob/2d380e992b782d96774881c3fef11be41a51bc43/tests/test_clustering.py
import faiss
import numpy as np

class FaissKNeighbors:
  def __init__(self, k=5, idxType=faiss.IndexFlatIP):
    self.index = None
    self.y = None
    self.k = k
    self.idxType=idxType

  def fit(self, X, y):
    self.index = self.idxType(X.shape[1])
    self.index.add(X.astype(np.float32))
    self.y = y

  def predict(self, X):
    self.distances, indices = self.index.search(X.astype(np.float32), k=self.k)
    votes = self.y[indices]
    predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
    return predictions
