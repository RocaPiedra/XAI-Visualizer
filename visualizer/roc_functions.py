"""

@author: Pablo Roca - github.com/RocaPiedra
"""
import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models
from misc_functions import apply_colormap_on_image, save_image

import pickle
from urllib.request import urlopen

def get_image_path(path, filename):
    if filename == None:
        onlyimages = [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) & f.endswith(('.jpg','.png'))]
        return onlyimages
    else:
        image_path = path + filename
        return image_path

def choose_model(input_argument = None, model_name = None):
    if model_name == 'resnet':
        model = models.resnet50(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif input_argument is not None:
        if int(input_argument) in (1,2):
            option = int(input_argument)
    else:
        option = int(input('Model is not defined choose from the available list:\n'
    '1. Alexnet\n2. ResNet\n'))
    if option == 1:
        model = models.alexnet(pretrained=True)
        print('Alexnet is the chosen classifier')
    elif option == 2:
        model = models.resnet18(pretrained=True)
        print('ResNet is the chosen classifier')
    else:
        print('Option incorrect, set default model: Alexnet')
        model = models.alexnet(pretrained=True)

    return model

def get_top_classes(output, number_of_classes = 5):
    idx = np.argpartition(output, -number_of_classes)[-number_of_classes:]
    return idx

def get_class_name_imagenet(idx):
    try:
        url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'
        imagenet = pickle.load(urlopen(url))
    except:
        imagenet = pickle.load(r'C:\Users\pablo\source\repos\pytorch-cnn-visualizations\ImageNet_utils\imagenet1000_clsid_to_human.pkl')

    return imagenet[idx]

def get_imagenet_dictionary(url=None):
    if url is None:
        url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'
    try:
        path_name = os.path.realpath(os.path.join(os.path.dirname(__file__),'../ImageNet_utils/imagenet1000_clsid_to_human.pkl'))
        # print(f'path is: {path_name}\nexists? -> {os.path.isfile(path_name)}')
        imagenet = pickle.load(path_name)
    except:
        print('pickle file was not found, downloading...')
        imagenet = pickle.load(urlopen(url))

    return imagenet
