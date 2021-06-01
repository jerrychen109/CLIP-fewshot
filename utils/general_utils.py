import subprocess

def install():
  bashCommand = "pip install ftfy regex faiss-gpu"
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()
  print(output, error)
  bashCommand = "wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz -O bpe_simple_vocab_16e6.txt.gz"
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()
  print(output, error)
  bashCommand = "apt install libomp-dev"
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()
  print(output, error)
  
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
  return device
