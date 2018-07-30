import torch
import torchvision
from torchvision import transforms, datasets
from random import randrange
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb

IMAGE_SIZE = (50, 50)
backgrounds = [Image.open(f).convert('RGBA') for f in Path('data/sprites/backgrounds/').glob('*.png')]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

def apply_to_background(img):
    w0, h0 = img.size  # PIL does this backwards
    h1, w1 = IMAGE_SIZE  # We do it the right way

    # Pick a random background and crop it
    i = randrange(0,len(backgrounds))
    bg = backgrounds[i].copy()
    bg = transforms.RandomCrop(IMAGE_SIZE)(bg)

    # If bg is bigger than img
    if w1>=w0 and h1>=h0:
        pass

    # If img is bigger than bg
    elif w0>w1 and h0>h1:
        img = transforms.RandomCrop(IMAGE_SIZE)(img)

    # If img width is bigger than bg width, crop to image hight, bg width
    elif w0>w1:
        img = transforms.RandomCrop((h0,w1))(img)

    # If img hight is bigger than bg hight, crop to image width, bg hight
    elif h0>h1:
        img = transforms.RandomCrop((h1,w0))(img)

    w0, h0 = img.size  # PIL does this backwards
    x0 = randrange(0,w1-w0+1)
    y0 = randrange(0,h1-h0+1)
    bg.paste(img, mask=img, box=(x0,y0))

    return bg.convert("RGB")

sprites_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        apply_to_background,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
sprites_dataset = datasets.ImageFolder(root='data/sprites/classes/',
                                       transform=sprites_transform,
                                       loader=pil_loader)
sprites_loader = torch.utils.data.DataLoader(sprites_dataset,
                                             batch_size=8, shuffle=True,
                                             num_workers=4)
class_names = sprites_dataset.classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

inputs, classes = next(iter(sprites_loader))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
