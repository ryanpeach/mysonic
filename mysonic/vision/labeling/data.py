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

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

class ApplyToBackground():
    def __init__(self, path):
        self.backgrounds = [Image.open(f).convert('RGBA') for f in Path(path).glob('*.png')]

    def __call__(self, img):
        w0, h0 = img.size  # PIL does this backwards

        # Pick a random background and crop it
        i = randrange(0,len(self.backgrounds))
        bg = self.backgrounds[i].copy()
        bg = transforms.RandomCrop((h0, w0))(bg)

        return Image.alpha_composite(bg, img).convert("RGB")


sprites_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMAGE_SIZE, pad_if_needed=True),
        ApplyToBackground('data/sprites/backgrounds/'),
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
