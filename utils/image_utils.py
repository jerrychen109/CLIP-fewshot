import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resize_images(images, height=224, width=224):
    """ Up/downscales images to the given width and height.
    Inputs:
    - images: a PyTorch tensor shaped (N, D, H, W)
    - height: the new height
    - width: the new width
    Returns:
    - the resized images, shaped (N, D, height, width)
    """
    return torch.nn.functional.interpolate(images, size=(height, width))


def preprocess(input_resolution=224):
    ''' Defines preprocessing function according to input resolution '''
    return Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])


def getImageMean(images):
    ''' TODO: Gets image mean given a set of images.
    Inputs:
    - images: a tensor of shape (N, D, H, W)
    Returns:
    - the mean pixel value across all images. Shape (D,)
    '''
    return torch.mean(images, dim=(0, 2, 3)).cuda()


def getImageStd(images):
    ''' TODO: Gets image standard deviation given a set of images
    Inputs:
    - images: a tensor of shape (N, D, H, W)
    Returns:
    - the pixel standard deviation across all images. Shape (D,)
    '''
    return torch.std(images, dim=(0, 2, 3)).cuda()


defImageMean = np.array([0.48145466, 0.4578275, 0.40821073])
defImageStd = np.array([0.26862954, 0.26130258, 0.27577711])

def standardize(images, device=device, image_mean=None, image_std=None):
    ''' Standardizes list of images'''
    if image_mean is None:
        image_mean = getImageMean(images)
    if image_std is None:
        image_std = getImageStd(images)
#     image_input = torch.tensor(np.stack(images), device=device)
#     image_input -= image_mean[:, None, None]
#     image_input /= image_std[:, None, None]
    images = images.clone()
    images -= image_mean[:, None, None]
    images /= image_std[:, None, None]
    return images


def getImageFilesFromDir(startDir):
    return [filename for filename in os.listdir(startDir) if filename.endswith(
        ".png") or filename.endswith(".jpg")]


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
        plt.subplot(*subplot_size, idx + 1)
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
    features /= np.linalg.norm(features, axis=0) #features.norm(dim=-1, keepdim=True)
    return features


def cosineSimilarity(features1, features2):
    norm_features1 = normalize(features1)
    norm_features2 = normalize(features2)
    similarity = norm_features1.cpu().numpy() @ norm_features2.cpu().numpy().T
    return similarity

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

        plt.subplot(height, width*2, 2 * i + 2)
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
            return (start, int(imageNum/start))
        start -= 1
    return (1, imageNum)


def encodeImageInModel(model, imageInput):
    return encodeImageWithFunc(model.encode_image, imageInput)


def encodeImageWithFunc(imageEncodeFunc, imageInput):
    with torch.no_grad():
        image_features = imageEncodeFunc(imageInput).float()
    return image_features


def imageToVector(image, imageEncodeFunc, device=device, image_mean=None, image_std=None):
    image = torch.tensor(np.stack(image), device=device)
    image = resize_images(standardize(image, device=device, image_mean=image_mean, image_std=image_std).unsqueeze(0))
    imageVectors = encodeImageWithFunc(imageEncodeFunc, image).squeeze()
    normImageVectors = normalize(imageVector.cpu().numpy())
    return imageVector, normImageVector


def imagesToVector(images, imageEncodeFunc, device=device, image_mean=None, image_std=None):
    images = torch.tensor(np.stack(images.clone()), device=device) # Allow this copy of images to be removed after function exits
    images = resize_images(standardize(images, device=device, image_mean=image_mean, image_std=image_std))
    imageVectors = encodeImageWithFunc(imageEncodeFunc, images)
    normImageVectors = np.array(list(map(lambda imageVector: normalize(imageVector.cpu().numpy()), imageVectors)))
    return imageVectors, normImageVectors
