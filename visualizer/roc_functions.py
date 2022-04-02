"""

@author: Pablo Roca - github.com/RocaPiedra
"""
import os
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models
from misc_functions import apply_colormap_on_image, save_image
import parameters

import pickle
from urllib.request import urlopen

import subprocess
from time import sleep

def preprocess_image(pil_im, sendToGPU=True, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    # if not im_as_var.is_cuda() and not sendtoGPU:
    if sendToGPU:
        im_as_var.to('cuda')
    return im_as_var

def get_image_path(path, filename):
    if filename == None:
        onlyimages = [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) & f.endswith(('.jpg','.png'))]
        return onlyimages
    else:
        image_path = path + filename
        return image_path

def choose_model(input_argument = None, model_name = None):
    if model_name == 'resnet':
        return models.resnet50(pretrained=True)
    elif model_name == 'alexnet':
        return models.alexnet(pretrained=True)
    elif model_name == 'yolov5':
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    elif input_argument is not None:
        if int(input_argument) in (1,2,3):
            option = int(input_argument)
    else:
        option = int(input('Model is not defined choose from the available list:\n'
    '1. Alexnet\n2. ResNet\n3. YoloV5\n'))
    if option == 1:
        print('Alexnet is the chosen classifier')
        return models.alexnet(pretrained=True)
    elif option == 2:
        print('ResNet is the chosen classifier')
        return models.resnet18(pretrained=True)
    elif option == 3:
        print('YoloV5 is the chosen classifier')
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    else:
        print('Option incorrect, set default model: Alexnet')
        return models.alexnet(pretrained=True)

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

def launch_carla_simulator_locally(unreal_engine_path = None):
    if unreal_engine_path is None:
        unreal_engine_path = parameters.unreal_engine_path
    print('Launching Unreal Engine Server...')
    if os.name == 'nt':
        unreal_engine = subprocess.Popen(unreal_engine_path, stdout=subprocess.PIPE)
    else:
        unreal_engine = subprocess.Popen([unreal_engine_path], stdout=subprocess.PIPE)
    sleep(5)
    print('Generating traffic...')
    generate_traffic = subprocess.Popen(["python", "../carlacomms/generate_traffic.py"], stdout=subprocess.PIPE)
    return unreal_engine, generate_traffic

