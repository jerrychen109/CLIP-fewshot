def importLibs():
  !pip install ftfy regex
  !wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz -O bpe_simple_vocab_16e6.txt.gz
  !apt install libomp-dev
  !pip install faiss-gpu

  import subprocess

  CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
  print("CUDA version:", CUDA_version)

  if CUDA_version == "10.0":
      torch_version_suffix = "+cu100"
  elif CUDA_version == "10.1":
      torch_version_suffix = "+cu101"
  elif CUDA_version == "10.2":
      torch_version_suffix = ""
  else:
      torch_version_suffix = "+cu110"

  from collections import OrderedDict
  import IPython.display
  import itertools
  import os
  from tqdm.notebook import tqdm

  from collections import Counter
  import matplotlib.pyplot as plt
  import matplotlib.ticker as ticker
  import numpy as np
  import pandas as pd
  from PIL import Image
  import seaborn as sns
  import skimage #Has some images in here - check original "Interacting with CLIP.ipynb" document
  import torch
  from torch.utils.data import DataLoader

  import faiss
  from faissKNeighbors import FaissKNeighbors
  from prototype import Prototype
  from prototypevector import PrototypeVector
  from torchvision.datasets import CIFAR10, CIFAR100
  from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
  from utils.data_utils import *
  from utils.image_utils import *
  from utils.text_utils import *
  from linear_classifier import *
  
  
def importSets():
  print("Torch version:", torch.__version__)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  %matplotlib inline
  %config InlineBackend.figure_format = 'retina'
  sns.set_theme(style="whitegrid")

  plt.rcParams['figure.figsize'] = (10.0, 8.0) # Set default size of plots.
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap'] = 'gray'

  %load_ext autoreload
  %autoreload 2
  %reload_ext autoreload
  %autoreload 2
