import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import gzip
import html
from functools import lru_cache
import ftfy
import regex as re
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

####### CODE FROM COLAB START #######


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"),
                                                      ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>',
                      '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors="replace").replace('</w>', ' ')
        return text
####### CODE FROM COLAB END #######


def preprocess(input_resolution=224):
    ''' Defines preprocessing function according to input resolution '''
    return Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])


def getImageMean(images):
    ''' TODO: Gets image mean given a set of images '''


def getImageStd(images):
    ''' TODO: Gets image mean given a set of images '''

defImageMean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
defImageStd = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def standardize(images, image_mean=defImageMean, image_std=defImageStd):
    ''' Standardizes list of images'''
    image_input = torch.tensor(np.stack(images)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    return image_input

def getImageFromFile(startDir, filename, input_resolution=224):
    return preprocess(input_resolution)(Image.open(os.path.join(
        startDir, filename)).convert("RGB"))


def getImagesFromFiles(startDir, filenames, input_resolution=224):
    return [getImageFromFile(startDir, filename, input_resolution) for filename in filenames]


def graphFiles(startDir, filenames, input_resolution=224, texts=None, descriptions=None):
    images = getImagesFromFiles(startDir, filenames, input_resolution)
    if texts is None:
        texts = extractNames(filenames)
    return graphImages(images, texts, descriptions)


def graphImages(images, texts=None, descriptions=None):
    plt.figure(figsize=(16, 16))
    if texts is None:
        texts = list(range(len(images)))

    subplot_size = extractSubplotSize(len(images))
    for idx, image in enumerate(images):
        plt.subplot(subplot_size, idx)
        plt.imshow(image.permute(1, 2, 0))
        if descriptions is None:
            plt.title(f"{texts[idx]}")
        else:
            plt.title(f"{texts[idx]}\n{descriptions[texts[idx]]}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    return images


def normalize(features):
    features /= features.norm(dim=-1, keepdim=True)
    return features


def cosineSimilarity(features1, features2):
    norm_features1 = normalize(features1)
    norm_features2 = normalize(features2)
    similarity = norm_features1.cpu().numpy() @ norm_features2.cpu().numpy().T


def softmax(features1, features2):
    text_probs = (100.0 * normalize(features1) @
                  normalize(features2).T).softmax(dim=-1)
    return text_probs


def graphImagesCosineSim(images, descriptions, image_features, text_features):
    similarity = cosineSimilarity(image_features, text_features)
    count = len(descriptions)

    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), list(descriptions.values()), fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image.permute(1, 2, 0), extent=(
            i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}",
                     ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    return similarity


def graphImagesSoftmax(images, labels, image_features, label_features):
    plt.figure(figsize=(16, 16))
    height, width = extractSubplotSize(len(images))
    text_probs = (image_features, label_features)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    for i, image in enumerate(images):
        plt.subplot(height, width*2, 2 * i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [labels[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()


def extractNames(filenames):
    ''' Finds last part of list of filenames '''
    return [os.path.basename(os.path.normpath(filename)) for filename in filenames]


def extractSubplotSize(imageNum):
    ''' Finds optimal width and height '''
    start = int(math.sqrt(imageNum))
    while True:
        if (imageNum % start) == 0:
            return (start, imageNum/start)
        start -= 1
    return (1, imageNum)
